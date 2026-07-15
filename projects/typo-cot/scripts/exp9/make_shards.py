"""実験9 本番キューのシャード一覧 TSV を生成する.

スコープ (計画書 §4 実験9、昇格後): M5 x B5 x 摂動2条件 (lxt4 + random4)。
mmlu (2850 サンプル) のみ 2 分割し、その他は 1 設定 = 1 シャード
→ 5 モデル x 2 条件 x 6 シャード = 60 シャード。

TSV 列: name, model, benchmark, condition, start, n ("-" = 未指定)。
name は scripts/exp9/run_inner_repair.py の出力タグ (shard_tag) と一致し、
キューの冪等スキップは results/exp9/summary_{name}.json の存在で判定する。

先頭 5 行は arc/lxt4 x 各モデル (モデル固有の障害を早期に検出するため)。

Qwen2.5-7B-Instruct の追加 (基盤生成完了後):
    uv run python scripts/exp9/make_shards.py --models Qwen2.5-7B-Instruct --append
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from typo_cot.repair.pipeline import shard_tag

M5 = [
    "gemma-3-4b-it",  # スモーク済みモデルを先頭 = 最初の検証シャード
    "Llama-3.2-1B-Instruct",
    "gemma-3-1b-it",
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]
B5 = ["gsm8k", "mmlu", "mmlu_pro", "arc", "commonsense_qa"]
CONDITIONS = ["lxt4", "random4"]

# ベンチマーク -> 範囲シャード分割 [(start, n)]。未登録は 1 シャード (全量)。
SPLITS: dict[str, list[tuple[int, int | None]]] = {
    "mmlu": [(0, 1425), (1425, None)],  # 2850 サンプルを半分に
}

DEFAULT_OUT = Path(__file__).parent.parent.parent / "results" / "exp9" / "queue" / "shards_active.tsv"


def make_rows(models: list[str]) -> list[tuple[str, str, str, str, str, str]]:
    rows = []
    for model in models:
        for condition in CONDITIONS:
            for bench in B5:
                for start, n in SPLITS.get(bench, [(0, None)]):
                    name = shard_tag(model, bench, condition, start=start, n=n)
                    rows.append(
                        (
                            name,
                            model,
                            bench,
                            condition,
                            str(start) if (start > 0 or n is not None) else "-",
                            str(n) if n is not None else "-",
                        )
                    )
    # 早期検証: arc/lxt4 の各モデル分を先頭へ (残りは元の順序を保持)
    early = [r for r in rows if r[2] == "arc" and r[3] == "lxt4"]
    rest = [r for r in rows if r not in early]
    return early + rest


def main() -> None:
    p = argparse.ArgumentParser(description="実験9 シャード一覧 TSV 生成")
    p.add_argument("--models", nargs="+", default=M5)
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--append", action="store_true", help="既存 TSV に追記 (Qwen 拡張用)")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_rows(args.models)
    mode = "a" if args.append else "w"
    with open(out, mode) as f:
        if not args.append:
            f.write("# name\tmodel\tbenchmark\tcondition\tstart\tn\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    print(f"{len(rows)} シャードを {out} に{'追記' if args.append else '書き出し'}")


if __name__ == "__main__":
    main()
