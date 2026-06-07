"""M1: AWQ/GPTQ/NF4/INT8/RTN で量子化バリアントを生成。

STATUS: stub — 実装は feature/quant_typo_neuron/quantization-gptq-awq。
README §5 の I/F に従うこと。
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to configs/*.yaml")
    args, overrides = p.parse_known_args()
    raise NotImplementedError(
        "feature/quant_typo_neuron/quantization-gptq-awq で実装予定 (config=%s, overrides=%s)"
        % (args.config, overrides)
    )


if __name__ == "__main__":
    main()
