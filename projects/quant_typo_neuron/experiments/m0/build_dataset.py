"""M0: WordNet単語同定データ 3版 (clean/typo/split) を生成。

STATUS: stub — 実装は feature/quant_typo_neuron/m0-wordnet-dataset。
README §5 の I/F に従うこと。
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="path to configs/*.yaml")
    args, overrides = p.parse_known_args()
    raise NotImplementedError(
        "feature/quant_typo_neuron/m0-wordnet-dataset で実装予定 (config=%s, overrides=%s)"
        % (args.config, overrides)
    )


if __name__ == "__main__":
    main()
