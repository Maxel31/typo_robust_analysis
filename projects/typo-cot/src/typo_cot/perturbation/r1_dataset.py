"""実験10③: R1蒸留(Qwen2トークナイザ)向け摂動データセット作成.

共有 PerturbedDatasetCreator への追記型拡張 (共有コードは変更しない)。

Qwen2 トークナイザは選択肢マーカー "(A)" を '(A' + ')' に分割する。
共有クラスの除外パターンは '(A)' / 'A' / '(' 等の完全形のみを対象とするため、
'(A' 断片が R_Q 上位に来ると摂動標的に選ばれてしまう
(修正前の実測: mmlu LXT-4 で 5564/5700 サンプルが該当)。
アーカイブ (gemma/llama 系トークナイザ) では '(' と 'A' が別トークンとして
それぞれ除外されるため、マーカー標的はゼロ。本サブクラスは
「選択肢マーカーは摂動しない」というアーカイブの摂動ポリシーを
Qwen2 トークナイザでも保つ。
"""

import json
import logging
import re
from pathlib import Path

from typo_cot.perturbation.dataset import PerturbedDatasetCreator

logger = logging.getLogger(__name__)

# 選択肢マーカー断片: 開き括弧 + 選択肢文字1文字 (閉じ括弧なし)
_MARKER_FRAGMENT_RE = re.compile(r"^\([A-Ja-j]$")


class R1PerturbedDatasetCreator(PerturbedDatasetCreator):
    """R1蒸留 (Qwen2トークナイザ) 向けの摂動データセット作成クラス.

    共有クラスとの差分は _should_skip_token のみ:
    選択肢マーカー断片 '(A' 〜 '(J' (閉じ括弧なし) を摂動対象から除外する。
    """

    def _should_skip_token(self, token: str) -> bool:
        if super()._should_skip_token(token):
            return True
        return bool(_MARKER_FRAGMENT_RE.match(token.strip()))


def create_r1_perturbed_dataset(
    baseline_dir: str | Path,
    num_perturbations: int,
    output_dir: str | Path,
    seed: int = 42,
    random_perturbation: bool = False,
    include_choices: bool = True,
) -> Path:
    """R1蒸留向け摂動データセットを作成して保存.

    保存形式・ディレクトリ命名は共有 create_perturbed_dataset と同一
    (アーカイブ互換)。metadata に creator="R1PerturbedDatasetCreator" を追記。

    Args:
        baseline_dir: Phase 1 の結果ディレクトリ
        num_perturbations: 摂動回数 (k)
        output_dir: 出力ディレクトリ (datasets/perturbed)
        seed: ランダムシード
        random_perturbation: True なら Random-k (上位k除外の乱択)
        include_choices: True なら選択肢も摂動対象

    Returns:
        保存された perturbed_dataset.json のパス
    """
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)

    config_path = baseline_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    model_name = config.get("model", "unknown").split("/")[-1]
    benchmark = config.get("benchmark", "unknown")

    mode_suffix = "_random" if random_perturbation else ""
    choices_suffix = "_with_choices" if include_choices else "_question_only"
    dataset_name = (
        f"{model_name}_{benchmark}_k{num_perturbations}{mode_suffix}{choices_suffix}"
    )
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    creator = R1PerturbedDatasetCreator(
        baseline_dir=baseline_dir,
        num_perturbations=num_perturbations,
        seed=seed,
        random_perturbation=random_perturbation,
        include_choices=include_choices,
    )
    dataset = creator.create()
    dataset.metadata["creator"] = "R1PerturbedDatasetCreator"
    dataset.metadata["skip_policy_extension"] = "choice_marker_fragment_(X"

    dataset_path = dataset_dir / "perturbed_dataset.json"
    dataset.save(dataset_path)

    with open(dataset_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataset.metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"R1 摂動データセットを保存: {dataset_dir}")
    return dataset_path
