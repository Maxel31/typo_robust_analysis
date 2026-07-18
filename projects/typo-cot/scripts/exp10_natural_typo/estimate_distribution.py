#!/usr/bin/env python3
"""実験10④: GitHub Typo Corpus から編集操作の経験分布を推定する CLI.

入力: data/external/github-typo-corpus.v1.0.0.jsonl.gz
      (Hagiwara & Mita 2020. 公式 S3 は消失済みのため Wayback Machine の
       スナップショット 20200906123950 から取得)
出力: configs/natural_typo_distribution.json (コミット対象)

フィルタ:
- is_typo == True の編集のみ (コーパス付属のtypoラベル)
- src/tgt とも lang == "eng"
- tgt(修正後)→src(修正前) が単一文字編集
  (置換/挿入/削除/隣接転置) で説明でき、編集文字が ASCII アルファベット

実行 (CPU のみ, 数分):
  uv run --no-sync python scripts/exp10_natural_typo/estimate_distribution.py
"""

import argparse
import gzip
import json
import logging
from collections import Counter, defaultdict

from typo_cot.perturbation.natural_typo import (
    NaturalTypoDistribution,
    extract_single_edit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 500  # 長大な行 (コード等) を除外


def normalize(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counter.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        default="data/external/github-typo-corpus.v1.0.0.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        default="configs/natural_typo_distribution.json",
    )
    parser.add_argument(
        "--min_cond_count",
        type=int,
        default=20,
        help="条件付き分布 (文字ごと) を保持する最小観測数",
    )
    args = parser.parse_args()

    op_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    sub_pairs: dict[str, Counter] = defaultdict(Counter)
    sub_marginal: Counter = Counter()
    ins_pairs: dict[str, Counter] = defaultdict(Counter)
    ins_marginal: Counter = Counter()
    word_len_counts: Counter = Counter()

    n_lines = 0
    n_edits = 0
    n_typo_eng = 0

    with gzip.open(args.corpus, "rt", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                commit = json.loads(line)
            except json.JSONDecodeError:
                continue
            for edit in commit.get("edits", []):
                n_edits += 1
                if not edit.get("is_typo"):
                    continue
                src = edit.get("src", {})
                tgt = edit.get("tgt", {})
                if src.get("lang") != "eng" or tgt.get("lang") != "eng":
                    continue
                src_text = src.get("text", "")
                tgt_text = tgt.get("text", "")
                if not src_text or not tgt_text:
                    continue
                if max(len(src_text), len(tgt_text)) > MAX_TEXT_LEN:
                    continue
                n_typo_eng += 1
                # intended = tgt (修正後), typo = src (修正前)
                single = extract_single_edit(tgt_text, src_text)
                if single is None:
                    continue
                op_counts[single.operation] += 1
                bucket_counts[single.bucket] += 1
                word_len_counts[min(len(single.word), 20)] += 1
                if single.operation == "substitution":
                    sub_pairs[single.intended_char][single.typed_char] += 1
                    sub_marginal[single.typed_char] += 1
                elif single.operation == "insertion":
                    ins_marginal[single.typed_char] += 1
                    if single.prev_char is not None:
                        ins_pairs[single.prev_char][single.typed_char] += 1

    n_used = sum(op_counts.values())
    logger.info(f"コーパス行数: {n_lines}, 総編集数: {n_edits}")
    logger.info(f"is_typo & eng & 長さ条件を満たす編集: {n_typo_eng}")
    logger.info(f"単一文字編集として採用: {n_used}")
    logger.info(f"操作分布 (件数): {dict(op_counts)}")
    logger.info(f"位置バケット (件数): {dict(bucket_counts)}")

    # 挿入のうち重複打鍵 (直前文字と同じ) の割合を記録
    doubling = sum(
        cnt
        for prev, counter in ins_pairs.items()
        for ch, cnt in counter.items()
        if ch == prev
    )
    n_ins_cond = sum(sum(c.values()) for c in ins_pairs.values())
    if n_ins_cond:
        logger.info(f"挿入のうち重複打鍵: {doubling}/{n_ins_cond} ({doubling / n_ins_cond:.1%})")

    dist = NaturalTypoDistribution(
        op_probs=normalize(op_counts),
        position_probs=normalize(bucket_counts),
        substitution_given_intended={
            ch: normalize(counter)
            for ch, counter in sorted(sub_pairs.items())
            if sum(counter.values()) >= args.min_cond_count
        },
        substitution_marginal=normalize(sub_marginal),
        insertion_given_prev={
            ch: normalize(counter)
            for ch, counter in sorted(ins_pairs.items())
            if sum(counter.values()) >= args.min_cond_count
        },
        insertion_marginal=normalize(ins_marginal),
        metadata={
            "source": "GitHub Typo Corpus v1.0.0 (Hagiwara & Mita 2020)",
            "corpus_file": str(args.corpus),
            "retrieval": (
                "Wayback Machine snapshot 20200906123950 of "
                "github-typo-corpus.s3.amazonaws.com (公式S3は消失)"
            ),
            "filters": {
                "is_typo": True,
                "lang": "eng",
                "max_text_len": MAX_TEXT_LEN,
                "edit": "single-char ASCII-alpha edit (sub/ins/del/adjacent transposition)",
            },
            "counts": {
                "corpus_lines": n_lines,
                "total_edits": n_edits,
                "typo_eng_edits": n_typo_eng,
                "single_char_edits_used": n_used,
                "op_counts": dict(op_counts),
                "bucket_counts": dict(bucket_counts),
                "insertion_doubling": doubling,
                "insertion_with_prev": n_ins_cond,
                "word_len_counts": {str(k): v for k, v in sorted(word_len_counts.items())},
            },
            "min_cond_count": args.min_cond_count,
        },
    )
    dist.save(args.output)
    logger.info(f"分布を保存: {args.output}")


if __name__ == "__main__":
    main()
