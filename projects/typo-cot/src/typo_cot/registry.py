"""Step 0 config レジストリ (configs/registry.yaml) の読み込みと整合検証.

prompt (few-shot テンプレートの sha256)・decoding (greedy)・seed を YAML に凍結し、
以後の全実験コードはレジストリ参照のみとする。`validate_registry` は現行
`models/prompts.py` からハッシュを再計算し、凍結値との drift を検出する。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from typo_cot.data.master_table import CONDITIONS, METRIC_SCOPE
from typo_cot.models.prompts import create_prompt_template

# projects/typo-cot/configs/registry.yaml
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "registry.yaml"
)

# プロンプトテンプレートのハッシュ計算に使う固定プローブ入力.
# 質問文自体はテンプレートに埋め込まれるため、固定文字列で埋めて
# テンプレート本文 (few-shot 例示・指示文・書式) の変化のみを検出する.
_PROBE_QUESTION = "PROBE QUESTION?"
_PROBE_SUBJECT = "probe subject"
_PROBE_CHOICES: dict[str, list[str] | None] = {
    "gsm8k": None,
    "mmlu": ["PROBE_A", "PROBE_B", "PROBE_C", "PROBE_D"],
    "mmlu_pro": [f"PROBE_{i}" for i in range(10)],
    "arc": ["PROBE_A", "PROBE_B", "PROBE_C", "PROBE_D"],
    "commonsense_qa": ["PROBE_A", "PROBE_B", "PROBE_C", "PROBE_D", "PROBE_E"],
}


def compute_prompt_hash(benchmark: str) -> str:
    """現行 prompts.py のテンプレートから決定的な sha256 を計算する."""
    template = create_prompt_template(benchmark)
    choices = _PROBE_CHOICES.get(benchmark)
    subject = _PROBE_SUBJECT if benchmark in ("mmlu", "mmlu_pro") else None
    result = template.generate(_PROBE_QUESTION, choices=choices, subject=subject)
    full_prompt = result.get_full_prompt()
    return hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()


def load_registry(path: Path | str | None = None) -> dict[str, Any]:
    """レジストリ YAML を読み込む."""
    path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_registry(registry: dict[str, Any]) -> None:
    """レジストリの整合性を検証する. 不整合は ValueError.

    - conditions が master_table.CONDITIONS と一致
    - metrics が master_table.METRIC_SCOPE と一致
    - prompts の sha256 が現行 prompts.py からの再計算値と一致 (drift 検出)
    - seed / decoding の必須キー
    """
    if int(registry.get("seed", -1)) != 42:
        raise ValueError(f"seed must be 42 (got {registry.get('seed')})")
    dec = registry.get("decoding") or {}
    for key in ("do_sample", "temperature", "max_new_tokens"):
        if key not in dec:
            raise ValueError(f"decoding.{key} is missing")

    conds = tuple((registry.get("conditions") or {}).keys())
    if conds != CONDITIONS:
        raise ValueError(f"condition set drift: {conds} != {CONDITIONS}")

    metrics = registry.get("metrics") or {}
    if metrics != METRIC_SCOPE:
        raise ValueError(f"metric scope drift: {metrics} != {METRIC_SCOPE}")

    prompts = registry.get("prompts") or {}
    for bench in registry.get("benchmarks") or []:
        entry = prompts.get(bench)
        if not entry:
            raise ValueError(f"prompt entry missing for benchmark: {bench}")
        expected = compute_prompt_hash(bench)
        if entry.get("sha256") != expected:
            raise ValueError(
                f"prompt hash drift for {bench}: registry={entry.get('sha256')} "
                f"recomputed={expected}"
            )


def prompt_id_for(registry: dict[str, Any], benchmark: str) -> str:
    """ベンチマークの凍結 prompt_id を返す."""
    return registry["prompts"][benchmark]["prompt_id"]


def build_registry_dict() -> dict[str, Any]:
    """現行コードベースからレジストリ dict を構築する (初回凍結用).

    凍結値の正典は configs/registry.yaml であり、本関数は生成・再検証にのみ使う。
    """
    return {
        "version": 1,
        "description": (
            "ARR August 2026 resubmission: frozen prompt/decoding/seed registry "
            "(Step 0). All experiment code must reference this registry."
        ),
        "seed": 42,
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
        "models": {
            "Llama-3.2-1B-Instruct": {"hf_id": "meta-llama/Llama-3.2-1B-Instruct"},
            "Llama-3.2-3B-Instruct": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct"},
            "Mistral-7B-Instruct-v0.3": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3"},
            "gemma-3-1b-it": {"hf_id": "google/gemma-3-1b-it"},
            "gemma-3-4b-it": {"hf_id": "google/gemma-3-4b-it"},
            # wave2 (2026-07-18): スコープ拡張モデル
            "Qwen2.5-7B-Instruct": {"hf_id": "Qwen/Qwen2.5-7B-Instruct"},
            "DeepSeek-R1-Distill-Qwen-7B": {
                "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "prompt_style": "zero_shot_chat_template",
            },
        },
        "benchmarks": ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math"],
        "conditions": {
            "clean": {"perturbation_mode": None, "num_perturbations": 0},
            "lxt1": {"perturbation_mode": "importance", "num_perturbations": 1},
            "lxt2": {"perturbation_mode": "importance", "num_perturbations": 2},
            "lxt4": {"perturbation_mode": "importance", "num_perturbations": 4},
            "lxt8": {"perturbation_mode": "importance", "num_perturbations": 8},
            "random4": {"perturbation_mode": "random", "num_perturbations": 4},
            "anti_lxt4": {"perturbation_mode": "bottom_k", "num_perturbations": 4},
        },
        "perturbation_types": ["proximity", "double_typing", "omission"],
        "prompts": {
            bench: {
                "prompt_id": f"{bench}_cot_v1",
                "sha256": compute_prompt_hash(bench),
            }
            for bench in ("gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa", "math")
        },
        "reasoning_prompts": {
            bench: {"prompt_id": f"{bench}_r1_think_v1"}
            for bench in ("gsm8k", "math", "mmlu")
        },
        "metrics": dict(METRIC_SCOPE),
    }


def write_registry_yaml(path: Path | str | None = None) -> Path:
    """`build_registry_dict()` を YAML に書き出す (初回凍結用)."""
    path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    registry = build_registry_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, allow_unicode=True, sort_keys=False)
    return path
