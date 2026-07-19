"""実験14: no-CoT 答えスパン生成コア (モデル依存部は generate_fn で注入).

生成本体 (ModelWrapper) はクロージャ generate_fn: list[prompt] -> list[継続]
として注入する。ユニットテストではモック、GPU 実行時は
ModelWrapper.generate_batch を包んだ関数を渡す (intervention.runner と同方式)。

各サンプルに no-CoT プロンプトを構築 → 答えスパンのみ生成 → 既存抽出器で
答えを抽出・採点する。CoT を通さない「質問→選択肢文字/数値の直接読み出し」を
測定するのが目的。
"""

from __future__ import annotations

from collections.abc import Callable

from typo_cot.evaluation.extractor import create_extractor
from typo_cot.models.nocot_prompts import create_nocot_prompt_template

GenerateFn = Callable[[list[str]], list[str]]

# 選択式ベンチマーク (choices を prompt テンプレートに渡す)
_MC_BENCHMARKS = {"mmlu", "mmlu_pro", "arc", "commonsense_qa"}


def build_nocot_prompt(sample: dict, benchmark: str) -> str:
    """1 サンプルの no-CoT 完全プロンプトを構築する.

    Args:
        sample: {"question", "choices", "subset"} を含む dict。
            摂動データは choices=None で question に選択肢が内包される
            (その場合テンプレートは question をそのまま使う)。
        benchmark: ベンチマーク名

    Returns:
        完全プロンプト文字列 (system + user)
    """
    template = create_nocot_prompt_template(benchmark)
    question = sample["question"]
    choices = sample.get("choices")
    subset = sample.get("subset")
    if benchmark in _MC_BENCHMARKS:
        result = template.generate(question=question, choices=choices, subject=subset)
    elif benchmark == "math":
        result = template.generate(question=question, subject=subset)
    else:  # gsm8k
        result = template.generate(question=question)
    return result.get_full_prompt()


def generate_nocot_records(
    samples: list[dict],
    benchmark: str,
    generate_fn: GenerateFn,
    batch_size: int = 16,
) -> dict[str, dict]:
    """全サンプルの no-CoT 答えスパンを生成・抽出・採点する.

    Args:
        samples: 各要素 {"sample_id","question","choices","correct_answer","subset"}
        benchmark: ベンチマーク名
        generate_fn: プロンプト列 -> 継続 (答えスパン) 列 (greedy 想定)
        batch_size: generate_fn 1 回に渡す最大プロンプト数

    Returns:
        sample_id -> {"answer","is_correct","generated","extraction_method"}
    """
    if not samples:
        return {}

    extractor = create_extractor(benchmark)
    prompts = [build_nocot_prompt(s, benchmark) for s in samples]

    generated: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        outputs = generate_fn(chunk)
        if len(outputs) != len(chunk):
            raise ValueError(
                f"generate_fn が {len(chunk)} 件に対し {len(outputs)} 件を返しました"
            )
        generated.extend(outputs)

    records: dict[str, dict] = {}
    for sample, gen in zip(samples, generated, strict=True):
        extraction = extractor.extract(gen)
        answer = extraction.extracted_answer
        is_correct = bool(answer) and extractor.is_correct(answer, sample["correct_answer"])
        records[sample["sample_id"]] = {
            "answer": answer,
            "is_correct": is_correct,
            "generated": gen,
            "extraction_method": extraction.extraction_method,
        }
    return records
