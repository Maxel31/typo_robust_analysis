#!/usr/bin/env bash
# 実験7 本番: pyspell 段の CPU 校正グリッド (5 モデル × 5 ベンチ = 25 データセット)。
#
# 粒度は「ベンチ×校正器×摂動元モデル」= LXT-4 摂動データセットごと。
# LXT-4 の標的語は摂動元モデルの帰属に依存するため、校正済みテキストも
# モデル固有になる (dev notes 参照)。GPU 不要・アーカイブは読み取りのみ。
#
# 冪等: 出力先に restoration_stats.json (最後に書かれるファイル) があれば
# スキップするので、中断後の再実行で続きから進む。
#
# 使い方:
#   nohup bash scripts/exp7/prod/run_pyspell_grid.sh > logs/exp7/pyspell_grid.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.."   # projects/typo-cot

ARCHIVE=/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/datasets/perturbed
OUT=data/exp7/corrected
PROGRESS=logs/exp7/pyspell_grid_progress.json
mkdir -p logs/exp7 "$OUT"

MODELS=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct gemma-3-1b-it gemma-3-4b-it Mistral-7B-Instruct-v0.3)
BENCHES=(gsm8k mmlu mmlu_pro arc commonsense_qa)

n_done=0; n_skip=0; n_fail=0
for m in "${MODELS[@]}"; do
  for b in "${BENCHES[@]}"; do
    src="$ARCHIVE/${m}_${b}_k4_with_choices/perturbed_dataset.json"
    dst="$OUT/${m}_${b}_k4_spellfix"
    if [ -f "$dst/restoration_stats.json" ]; then
      echo "[skip] $m x $b (完了済み)"
      n_skip=$((n_skip + 1))
      continue
    fi
    echo "[run ] $m x $b $(date '+%F %T')"
    if uv run python scripts/exp7/make_corrected_dataset.py \
        --input "$src" --corrector pyspell --output_dir "$OUT"; then
      n_done=$((n_done + 1))
    else
      echo "[FAIL] $m x $b" >&2
      n_fail=$((n_fail + 1))
    fi
    printf '{"done": %d, "skipped": %d, "failed": %d, "updated": "%s"}\n' \
      "$n_done" "$n_skip" "$n_fail" "$(date -Is)" > "$PROGRESS"
  done
done
printf '{"done": %d, "skipped": %d, "failed": %d, "finished": true, "updated": "%s"}\n' \
  "$n_done" "$n_skip" "$n_fail" "$(date -Is)" > "$PROGRESS"
echo "=== pyspell grid 完了: done=$n_done skip=$n_skip fail=$n_fail ==="
