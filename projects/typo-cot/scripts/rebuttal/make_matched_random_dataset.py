#!/usr/bin/env python3
"""Rebuttal 実験④: Matched-Random 対照摂動データセット作成 (AxQH Weakness 3 / R1 Comment 2).

Random 条件を「LXT-4 が選んだトークンと (a) 語長・(b) 内容語/機能語クラスをマッチさせた
ランダム選択」に置き換えた摂動データセットを作成する。

選択ロジックは perturbation/dataset.py の random モード (dataset.py:405-434) をミラーする:
- 重要度降順ソート → 上位 k 個 (=LXT-4 のターゲット) を除いた残りを候補プールとする
- 上位 k 個の各トークンについて、プールから (クラス一致, 語長差最小) のトークンを
  乱数タイブレークで非復元抽出する
- 摂動の適用は既存の _apply_perturbations の適用ループ (per-token seed, offset 調整,
  perturb() 失敗時の次候補フォールバック) と同一の処理を用いる

出力形式は既存 perturbed_dataset.json と完全互換 (perturbation_mode="matched_random")。

使用例:
  PYTHONHASHSEED=42 uv run --no-sync python scripts/rebuttal/make_matched_random_dataset.py \
    --baseline_dir outputs/baseline/gemma-3-4b-it_mmlu \
    --num_perturbations 4 --output_dir datasets/rebuttal
"""

import argparse
import json
import random
from pathlib import Path


from typo_cot.perturbation.dataset import PerturbedDatasetCreator

# 英語機能語リスト (冠詞・前置詞・代名詞・接続詞・助動詞・基本副詞等)。
# spaCy モデル依存を避けるためハードコード。判定は小文字化後の完全一致。
FUNCTION_WORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "each", "every", "either",
    "neither", "some", "any", "no", "all", "both", "few", "many", "much", "more",
    "most", "other", "another", "such", "what", "which", "whose",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "themselves", "who", "whom", "someone", "anyone", "everyone", "nothing",
    "something", "anything", "everything", "one", "none",
    # prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "of", "off", "over", "under", "again", "further", "than", "as", "per",
    "via", "within", "without", "upon", "among", "across", "behind", "beyond",
    "near", "onto", "toward", "towards",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "if", "because", "although", "though",
    "while", "when", "whenever", "where", "wherever", "since", "unless", "until",
    "whether", "once",
    # auxiliaries / copula
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "ought", "need", "dare",
    # negation / common adverbs of degree
    "not", "n't", "only", "just", "also", "too", "very", "then", "there", "here",
    "how", "why", "out",
}

SUBWORD_MARKERS = ("▁", "Ġ")


def strip_token(token: str) -> str:
    """サブワードマーカー・空白を除去した表層形を返す."""
    t = token.strip()
    for m in SUBWORD_MARKERS:
        t = t.replace(m, "")
    return t


def token_class(token: str) -> str:
    """内容語/機能語の2値クラスを返す."""
    surface = strip_token(token).lower().strip("'\"()[].,:;!?")
    return "function" if surface in FUNCTION_WORDS else "content"


def token_len(token: str) -> int:
    """マーカー除去後の文字長."""
    return len(strip_token(token))


class MatchedRandomDatasetCreator(PerturbedDatasetCreator):
    """Matched-Random モード: 語長・内容語/機能語クラスをマッチさせたランダム選択.

    _apply_perturbations の候補順序決定のみを差し替え、適用ループは既存実装
    (dataset.py:436-504) と同一の処理を踏襲する。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.match_log: list[dict] = []

    def _apply_perturbations(
        self,
        question: str,
        question_tokens: list[tuple[int, str, float]],
        question_char_start: int,
        offset_mapping: list[tuple[int, int]],
        sample_id: str,
    ):
        if not question_tokens:
            return question, []

        # --- 選択部: random モード (dataset.py:405-428) のミラー + マッチング ---
        sorted_by_importance = sorted(question_tokens, key=lambda x: x[2], reverse=True)

        if len(sorted_by_importance) > self.num_perturbations:
            top_k = sorted_by_importance[: self.num_perturbations]
            pool = list(sorted_by_importance[self.num_perturbations :])
        else:
            # random モードと同じフォールバック: 全トークンから選択
            top_k = sorted_by_importance
            pool = list(sorted_by_importance)

        rng = random.Random(hash((self.seed, sample_id, "matched_selection")))

        matched: list[tuple[int, str, float]] = []
        match_records = []
        for _t_idx, t_tok, _t_score in top_k:
            if not pool:
                break
            t_cls = token_class(t_tok)
            t_len = token_len(t_tok)
            # (クラス一致, 語長差最小) の候補集合
            same_cls = [c for c in pool if token_class(c[1]) == t_cls]
            search_space = same_cls if same_cls else pool
            min_diff = min(abs(token_len(c[1]) - t_len) for c in search_space)
            best = [c for c in search_space if abs(token_len(c[1]) - t_len) == min_diff]
            choice = rng.choice(best)
            pool.remove(choice)
            matched.append(choice)
            match_records.append(
                {
                    "template_token": t_tok,
                    "template_class": t_cls,
                    "template_len": t_len,
                    "matched_token": choice[1],
                    "matched_class": token_class(choice[1]),
                    "matched_len": token_len(choice[1]),
                    "len_diff": abs(token_len(choice[1]) - t_len),
                    "class_match": token_class(choice[1]) == t_cls,
                }
            )

        # 適用ループでの perturb() 失敗に備え、残りプールをシャッフルして後置
        backup = list(pool)
        rng.shuffle(backup)
        candidate_tokens = matched + backup

        self.match_log.append({"sample_id": sample_id, "matches": match_records})

        return self._apply_from_candidates(
            question, candidate_tokens, question_char_start, offset_mapping, sample_id
        )

    def _apply_from_candidates(
        self,
        question: str,
        candidate_tokens: list[tuple[int, str, float]],
        question_char_start: int,
        offset_mapping: list[tuple[int, int]],
        sample_id: str,
    ):
        """dataset.py:436-504 の適用ループと同一の処理."""
        from typo_cot.perturbation.dataset import PerturbedToken
        from typo_cot.perturbation.generator import (
            CharacterPerturbationGenerator,
        )

        perturbed_question = question
        perturbed_tokens = []
        used_token_indices: set[int] = set()
        offset_adjustment = 0

        for token_index, token_str, score in candidate_tokens:
            if len(perturbed_tokens) >= self.num_perturbations:
                break
            if token_index in used_token_indices:
                continue
            if token_index >= len(offset_mapping):
                continue

            char_start, char_end = offset_mapping[token_index]
            relative_start = char_start - question_char_start + offset_adjustment
            relative_end = char_end - question_char_start + offset_adjustment

            if relative_start < 0 or relative_end > len(perturbed_question):
                continue

            current_token = perturbed_question[relative_start:relative_end]

            token_seed = hash((self.seed, sample_id, token_str))
            token_generator = CharacterPerturbationGenerator(seed=token_seed)
            result = token_generator.perturb(current_token)

            if result is None:
                continue

            perturbed_question = (
                perturbed_question[:relative_start]
                + result.perturbed
                + perturbed_question[relative_end:]
            )
            offset_adjustment += len(result.perturbed) - len(current_token)

            perturbed_tokens.append(
                PerturbedToken(
                    token_index=token_index,
                    original_token=token_str,
                    perturbed_token=result.perturbed,
                    importance_score=score,
                    perturbation_type=result.perturbation_type.value,
                    char_position=result.position,
                )
            )
            used_token_indices.add(token_index)

        return perturbed_question, sorted(perturbed_tokens, key=lambda x: x.token_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched-Random 摂動データセット作成")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--num_perturbations", "-k", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="datasets/rebuttal")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    config_path = baseline_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    model_name = config.get("model", "unknown").split("/")[-1]
    benchmark = config.get("benchmark", "unknown")

    dataset_name = f"{model_name}_{benchmark}_k{args.num_perturbations}_matched_random"
    dataset_dir = Path(args.output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    creator = MatchedRandomDatasetCreator(
        baseline_dir=baseline_dir,
        num_perturbations=args.num_perturbations,
        seed=args.seed,
        include_choices=True,
    )
    dataset = creator.create()
    dataset.metadata["perturbation_mode"] = "matched_random"

    dataset_path = dataset_dir / "perturbed_dataset.json"
    dataset.save(dataset_path)
    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, ensure_ascii=False, indent=2)

    # マッチング品質統計
    all_matches = [m for entry in creator.match_log for m in entry["matches"]]
    n = len(all_matches)
    stats = {
        "n_samples": len(creator.match_log),
        "n_matches": n,
        "class_match_rate": sum(m["class_match"] for m in all_matches) / n if n else 0,
        "exact_len_match_rate": sum(m["len_diff"] == 0 for m in all_matches) / n if n else 0,
        "mean_len_diff": sum(m["len_diff"] for m in all_matches) / n if n else 0,
        "template_class_dist": {
            "content": sum(m["template_class"] == "content" for m in all_matches),
            "function": sum(m["template_class"] == "function" for m in all_matches),
        },
    }
    with open(dataset_dir / "matched_stats.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": stats, "per_sample": creator.match_log}, f,
                  ensure_ascii=False, indent=2)

    print(f"出力: {dataset_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
