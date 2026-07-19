#!/usr/bin/env python3
"""実験13: 答え→CoT attention 集中度 (M3 の安価な代理) の算出.

各サンプルについて (prompt + CoT + 答えトリガー + 答え) を1回 forward し
(output_attentions=True, eager attention)、答えトークン位置の行から CoT
トークン列への attention 質量分布の Gini を層ごとに算出する。LOO Gini の
代理妥当性検証に用いる。GPU 1枚・生成なし・forward 1回/サンプル。

サンプル選定は run_loo_scoring.py と同一 (clean 正解から seed 固定で n 件)。

入力: アーカイブ baseline run_dir の results.json (読み取り専用)。使用フィールド:
      sample_id(str)/generated_text(str)/question/choices/subset/is_correct(bool)。
出力: {out_dir}/{model}_{benchmark}_{run_label}/
       ├── config.json    # timestamp は ISO-8601
       ├── results.json   # per-sample: attn_gini_mean / attn_gini_per_layer ...
       └── summary.json   # 設定レベル集計 (mean attn_gini, 層別プロファイル)
"""

import argparse
import importlib.util
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("attn_concentration")
logger.setLevel(logging.INFO)


def build_full_and_spans(prompt: str, split):
    """(full_text, cot_char_span, answer_char_span) を返す.

    full = prompt + cot_text + trigger_text + answer_text (答え以降は切る)。
    char span は full 上の [start, end)。
    """
    cot_start = len(prompt)
    cot_end = cot_start + len(split.cot_text)
    ctx = prompt + split.cot_text + split.trigger_text
    ans_start = len(ctx)
    full = ctx + split.answer_text
    ans_end = len(full)
    return full, (cot_start, cot_end), (ans_start, ans_end)


def positions_in_span(offsets, span):
    """offset_mapping から、トークン開始位置が span 内にあるトークン index のリスト.

    特殊トークン (BOS 等, offset=(0,0)) は除外し、各トークンを「開始位置が
    属する領域」に一意に割り当てる (境界トークンの二重計上を防ぐ)。
    """
    s, e = span
    return [i for i, (ts, te) in enumerate(offsets) if ts != te and s <= ts < e]


def main() -> None:
    ap = argparse.ArgumentParser(description="answer->CoT attention concentration (exp13)")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--clean_run_dir", type=str, default=None)
    ap.add_argument("--gpu_id", type=str, default="0")
    ap.add_argument("--output_dir", type=str, default="results/attention")
    ap.add_argument("--run_label", type=str, default="clean_attn")
    ap.add_argument("--max_seq_len", type=int, default=3072,
                    help="これを超えるトークン長のサンプルはスキップ (メモリ保護)")
    ap.add_argument("--reduce_answer", type=str, default="mean", choices=["mean", "sum"])
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from typo_cot.analysis.concentration import (
        answer_to_cot_distribution,
        attention_gini_per_layer,
        gini,
        top1_share,
    )
    from typo_cot.intervention.loo_scorer import split_generated_text
    from typo_cot.models.prompts import create_prompt_template
    from typo_cot.models.wrapper import setup_device

    # run_loo_scoring のヘルパ (load_run_entries/select_sample_ids/build_prompt) を流用
    loo_run = Path(__file__).parent / "run_loo_scoring.py"
    spec = importlib.util.spec_from_file_location("run_loo_scoring", loo_run)
    rls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rls)

    run_dir = Path(args.run_dir)
    entries = rls.load_run_entries(run_dir)
    n_selected = None
    if args.seed is not None:
        clean_dir = Path(args.clean_run_dir) if args.clean_run_dir else run_dir
        selected = rls.select_sample_ids(rls.load_run_entries(clean_dir), args.n, args.seed)
        n_selected = len(selected)
        sel = set(selected)
        entries = [e for e in entries if e["sample_id"] in sel]

    model_short = args.model.split("/")[-1]
    out_dir = Path(args.output_dir) / f"{model_short}_{args.benchmark}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device, _ = setup_device(args.gpu_id)
    logger.info(f"モデルロード (eager attention): {args.model} @ {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    template = create_prompt_template(args.benchmark)

    config = {
        "model": args.model, "benchmark": args.benchmark, "run_dir": str(run_dir),
        "n": args.n, "seed": args.seed, "clean_run_dir": args.clean_run_dir,
        "n_selected": n_selected, "reduce_answer": args.reduce_answer,
        "max_seq_len": args.max_seq_len, "method": "attention_concentration",
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    stats = {"total_seen": 0, "scored": 0, "skip_no_answer": 0,
             "skip_no_cot_or_answer_tok": 0, "skip_too_long": 0, "errors": 0}
    out_results: list[dict] = []
    n_layers = None
    t0 = time.time()

    for entry in entries:
        if args.n is not None and stats["scored"] >= args.n:
            break
        stats["total_seen"] += 1
        sid = entry["sample_id"]
        try:
            prompt = rls.build_prompt(template, args.benchmark, entry)
            split = split_generated_text(entry["generated_text"])
            if split is None:
                stats["skip_no_answer"] += 1
                continue
            full, cot_span, ans_span = build_full_and_spans(prompt, split)
            enc = tokenizer(full, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            seq_len = enc["input_ids"].shape[1]
            if seq_len > args.max_seq_len:
                stats["skip_too_long"] += 1
                continue
            cot_pos = positions_in_span(offsets, cot_span)
            ans_pos = positions_in_span(offsets, ans_span)
            if not cot_pos or not ans_pos:
                stats["skip_no_cot_or_answer_tok"] += 1
                continue

            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            output_attentions=True, use_cache=False)
            attns = out.attentions  # tuple[L] of [1, heads, seq, seq]
            n_layers = len(attns)
            per_layer = attention_gini_per_layer(
                attns, answer_positions=ans_pos, cot_positions=cot_pos,
                batch_index=0, reduce_answer=args.reduce_answer,
            )
            # 全層平均した分布の集中度 (層平均後に Gini/top1)
            dists = []
            for layer in attns:
                hm = layer[0].float().mean(axis=0).cpu().numpy()
                dists.append(answer_to_cot_distribution(
                    hm, ans_pos, cot_pos, reduce_answer=args.reduce_answer))
            mean_dist = np.mean(np.stack(dists, axis=0), axis=0)
            del out, attns
            if device.type == "cuda":
                torch.cuda.empty_cache()

            out_results.append({
                "sample_id": sid,
                "n_cot_tokens": len(cot_pos),
                "n_answer_tokens": len(ans_pos),
                "seq_len": int(seq_len),
                "attn_gini_mean": float(statistics.mean(per_layer)),
                "attn_gini_max": float(max(per_layer)),
                "attn_gini_per_layer": [float(x) for x in per_layer],
                "attn_gini_agg": float(gini(mean_dist, clip_negative=False)),
                "attn_top1_agg": float(top1_share(mean_dist, clip_negative=False)),
            })
            stats["scored"] += 1
            if stats["scored"] % 10 == 0:
                rate = (time.time() - t0) / stats["scored"]
                logger.info(f"{stats['scored']} 件 ({rate:.2f}s/sample)")
                tmp = out_dir / "results.json.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(out_results, f, ensure_ascii=False)
                tmp.replace(out_dir / "results.json")
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            logger.warning(f"{sid} 失敗: {exc}")
            continue

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(out_results, f, ensure_ascii=False, indent=2)

    def _m(key):
        xs = [r[key] for r in out_results]
        return statistics.mean(xs) if xs else None

    layer_profile = None
    if out_results and n_layers:
        layer_profile = [
            statistics.mean(r["attn_gini_per_layer"][li] for r in out_results)
            for li in range(n_layers)
        ]
    summary = {
        "experiment_info": {
            "model": args.model, "benchmark": args.benchmark,
            "method": "attention_concentration", "run_dir": str(run_dir),
            "n_layers": n_layers, "timestamp": datetime.now().isoformat(),
        },
        "stats": {**stats, "elapsed_sec": round(time.time() - t0, 1)},
        "metrics": {
            "mean_attn_gini_mean": _m("attn_gini_mean"),
            "mean_attn_gini_max": _m("attn_gini_max"),
            "mean_attn_gini_agg": _m("attn_gini_agg"),
            "mean_attn_top1_agg": _m("attn_top1_agg"),
            "mean_n_cot_tokens": _m("n_cot_tokens"),
            "layer_profile_mean_gini": layer_profile,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"完了: {out_dir}")
    logger.info(json.dumps({k: v for k, v in summary["metrics"].items()
                            if k != "layer_profile_mean_gini"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
