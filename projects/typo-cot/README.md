# typo-cot

Typo（文字レベル摂動）× CoT の 2 軸分析 — **ARR 2026年8月再投稿** 実験プロジェクト。

AttnLRP による重要度帰属（R_Q: 質問トークン→最初の CoT トークン, R_C: CoT トークン→最終回答）を使い、
重要語への typo 注入が CoT 生成と正答率に与える影響を分析する。v1（25 設定の基礎分析）を土台に、
リバッタル対応と ERDC（Encode–Repair–Divert–Carry）連鎖の検証のため Step0 および実験1〜20 を
実施している。各実験の一覧・実行コマンドは [実験一覧](#実験一覧step0--実験20) を、
仮説判定・結果表などの正典は [結果・仮説の正本](#結果仮説の正本) を参照。

JSAI2026 リポジトリ (`/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026`,
パッケージ名 `attn_perturbation`) の後継。コードは `typo_cot` にリネームして移行済み、
実験データ（約 96GB）はアーカイブに置いたまま読み取り専用で参照する
（[docs/data_provenance.md](docs/data_provenance.md), [configs/paths.yaml](configs/paths.yaml)）。

## 初期実験マトリクス（v1, 25 設定）

Step0 以降の全実験は、この v1 マトリクスで生成した baseline / perturbed 推論結果と
AttnLRP 帰属（R_Q/R_C）を入力データとして再利用する。

| 軸 | 値 |
|----|----|
| モデル | Llama-3.2-1B/3B-Instruct, gemma-3-1b/4b-it, Mistral-7B-Instruct-v0.3 |
| ベンチマーク | gsm8k, mmlu, mmlu_pro, arc, commonsense_qa |
| 摂動条件 | importance k=1/2/4/8, random k=4, bottom_k k=4 |

正典は [docs/v1_run_manifest.md](docs/v1_run_manifest.md)（v1/v2 アナライザの差異も記載）。

## セットアップ

```bash
cd projects/typo-cot

# 軽量セット（metrics / figures / tables のみ。torch 不要）
uv sync

# 生成 + AttnLRP 帰属（Phase 1/3）および Phase 4 分析
# （analysis/analyzer.py と visualization/heatmap.py は torch を import する。GPU は必須ではない）
uv sync --extra lrp

# Appendix 分析（spacy / wordcloud） / W&B アップロード
uv sync --extra lrp --extra appendix --extra tracking
```

秘密情報はローカルの `.env`（gitignore 済み）に置く: `HF_TOKEN`, `WANDB_API_KEY`。

すべてのスクリプトは `uv run python scripts/...`（本ワークスペースの uv ワークスペース規約）で
実行し、GPU は各スクリプトの `--gpu_id`（旧リポジトリ由来の引数名。config には書かない）で指定する。

## GPUロックヘルパー（実験1〜20共通）

複数の実験を並行して GPU 上で実行するため、GPU 排他制御ヘルパー
`tmp/gpu-locks/run_with_gpu.sh` を経由して推論・パッチング系スクリプトを実行する
（`tmp/` はメインの作業ディレクトリにのみ存在する未追跡ファイルで、各実験 worktree には
コピーされないため、**絶対パスで参照する**）。

```bash
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
    uv run python scripts/run_inference.py --model ... --gpu_id "$CUDA_VISIBLE_DEVICES" ...
```

- 許可 GPU・タイムアウトはスクリプト冒頭のコメント（directive 日付付き）で管理されており、
  `GPU_CANDIDATES` / `GPU_LOCK_TIMEOUT` 環境変数で上書きできる。
- 空き GPU を `flock` + `nvidia-smi` で確認してから `CUDA_VISIBLE_DEVICES` を設定してコマンドを実行する。
- `SMOKE_PAUSED` ファイルが置かれている間はユーザー指示により GPU 作業を一時停止する
  （exit code 86、待機・リトライ禁止）。

以下の各実験の実行例における `run_with_gpu.sh` は、すべてこの絶対パスを指す。

## 実験一覧（Step0 〜 実験20）

各実験は個別の git worktree + ブランチ（`exp/<name>`）で開発されている。
以下のコマンドは各 worktree の `projects/typo-cot/` から実行したものとして記載する
（`uv run --no-sync` はモノレポ共有 `.venv` を再インストールせず使う場合の指定）。

| # | 実験名 | 概要 | Worktree / ブランチ | 状態 |
|---|--------|------|----------------------|------|
| Step0 | ベースライン生成・R_Q/R_C計算基盤 | 全実験の入力データ源（baseline 推論 + AttnLRP 帰属の master table 化・照合） | `exp-step0` / `exp/step0` | 完了 |
| 1 | 推論結果転写（transplant） | flip ~ Q_typo×CoT_typo の CoT 転写実験 | `exp-01-03-transplant` / `exp/01-03-transplant` | 完了 |
| 3 | KL-R_C 収束性 + R1蒸留拡張 | divergence（KL）と R_C の一致度、DeepSeek-R1蒸留モデルへの拡張 | 同上 | 完了 |
| 2 | ターゲット単語削除LOO | 重要語削除による Leave-One-Out 効果検証・recovery curve | `exp-02-target-deletion` / `exp/02-target-deletion` | 完了 |
| 4 | 固定ターゲット摂動 + MATH-500拡張 | 摂動対象語を固定した帰属比較、MATH-500 への拡張 | `exp-04-fixed-target` / `exp/04-fixed-target` | 完了 |
| 5 | マッチド統制条件 | 摂動位置をマッチさせた twin データセットによる統制群比較 | `exp-05-matched-control` / `exp/05-matched-control` | 完了 |
| 6 | AttnLRP帰属手法収束性 | gxi/ig/rollout 各帰属手法と LOO ランキングの一致度 | `exp-06-attribution` / `exp/06-attribution` | 完了 |
| 7 | 誤字修正器 | pyspell/neural/llm 修正器によるレストレーション効果 | `exp-07-correctors` / `exp/07-correctors` | 完了 |
| 11 | 連鎖媒介 | ERDC連鎖の媒介分析（KL_sum → repair_score） | 同上（`analysis/exp11_chain_mediation/`） | 完了 |
| 12 | R_C組成 | CoTトークンの構成分類（結論/数値/内容語など） | 同上（`analysis/exp12_rc_composition/`） | 完了 |
| 16 | 統一GLMM | モデレーター（M1/M2/M3）による setting 分散の吸収 | 同上（`analysis/exp16_unified/`） | 完了 |
| 17 | 行動修復 | 明示的自己修正マーカーと flip の関係 | 同上（`analysis/exp17_behavioral_repair/`） | 完了 |
| 8 | アクティベーションパッチング | 層単位 patching による効果の局在化（residual/attn/mlp） | `exp-08-patching` / `exp/08-patching` | 完了 |
| 8-fine | 単層注入局在 | 単層粒度（fine-grained）での patching | `exp-08-fine` / `exp/08-fine` | 完了 |
| 9 | 内部修復 | forward-only の repair_score 計算と flip 予測 | `exp-09-inner-repair` / `exp/09-inner-repair` | 完了 |
| 10 | スコープ拡張 | natural typo / R1蒸留摂動 / Qwen摂動 / MATH-500 の4サブスコープ | `exp-10-scope` / `exp/10-scope` | 完了 |
| 13 | 読み出し集中度（Gini） | attention concentration と LOO 結果の順位相関 | `exp-13-readout` / `exp/13-readout` | 完了 |
| 14 | no-CoTショートカット | CoT なし直接回答時の flip 率と DE の関係 | `exp-14-nocot` / `exp/14-nocot` | 完了 |
| 15 | パッチ→自由生成 | 早期窓 patch 後の自由生成によるレストレーション検証 | `exp-15-patch-freegen` / `exp/15-patch-freegen` | 完了 |
| 18 | 形式移植 | S3/DE の形式依存性検証（MC-GSM8K vs free-form-MMLU）。計画は [docs/experiments_11_18_plan.md](docs/experiments_11_18_plan.md) §3.8 | worktree 未作成 | 計画段階・未着手 |
| 19 | サイズラダー | モデルサイズに対する DE/repair のスケーリング検証。計画は [docs/size_ladder_plan.md](docs/size_ladder_plan.md) | `exp-19-size-ladder` / `exp/19-size-ladder` | **進行中**（バックグラウンドジョブ稼働中。詳細は当該 worktree・ブランチを参照） |
| 20 | 防御（D1） | 摂動に対する防御手法の効果検証 | `exp-20-defense` / `exp/20-defense` | **進行中**（バックグラウンドジョブ稼働中。詳細は当該 worktree・ブランチを参照） |

実験19・20 は本 README 作成時点でバックグラウンド実行中のため、実行コマンドの記載は省略する
（数値・状況は各ブランチのマージ後に更新予定）。以下、完了済みの各実験について、
実在するスクリプト・コマンドのみを掲載する。

### Step0 — ベースライン生成・R_Q/R_C計算基盤

主要スクリプト: `scripts/step0_build_master_table.py`, `scripts/step0_smoke_reproduce.py`

```bash
# スモーク（2設定のみで照合）
uv run python scripts/step0_smoke_reproduce.py --models gemma-3-4b-it --benchmarks gsm8k mmlu

# 本番（v1 25設定を全照合）
uv run python scripts/step0_smoke_reproduce.py

# master table 構築（全セル、GPU不要）+ 検証
uv run python scripts/step0_build_master_table.py
uv run python scripts/step0_build_master_table.py --verify
```

### 実験1 + 3 — 推論結果転写・KL-R_C収束性（+R1蒸留拡張）

主要スクリプト: `scripts/exp01_03/run_transplant.py`, `make_shards.py`, `verify_shard.py`,
`queue_worker.sh`, `queue_status.sh`

```bash
# スモーク
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
  uv run python scripts/exp01_03/run_transplant.py \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --baseline-dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --perturbed-dir $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --n 32 --dump-divergence --output-dir results/smoke/gemma3-4b_gsm8k_lxt4

# 本番キュー投入
python scripts/exp01_03/make_shards.py > scripts/exp01_03/shards_all.tsv
cd projects/typo-cot && WORKER_ID=w1 setsid nohup bash scripts/exp01_03/queue_worker.sh \
    < /dev/null >> logs/exp01_03/worker_w1.log 2>&1 &
bash scripts/exp01_03/queue_status.sh   # 監視
touch results/exp01_03/queue/STOP       # 停止
uv run python scripts/exp01_03/verify_shard.py results/exp01_03/<shard>   # 検証
```

（`$ARCHIVE` = `/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026`）

### 実験2 — ターゲット単語削除LOO

主要スクリプト: `scripts/exp2/run_target_deletion.py`, `run_recovery_curve.py`, `make_queue.py`,
`queue_worker.sh`, `queue_status.sh`

```bash
# スモーク
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp2/run_target_deletion.py \
  --baseline_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --model google/gemma-3-4b-it --benchmark gsm8k --arms smoke \
  --clean_correct_only --n 24 --output_dir results/exp2_smoke

# 本番キュー投入
uv run python scripts/exp2/make_queue.py   # results/prod/exp2/queue/shards_active.tsv 生成
cd projects/typo-cot && WORKER_ID=w1 setsid nohup bash scripts/exp2/queue_worker.sh \
    < /dev/null >> logs/exp2_queue/worker_w1.log 2>&1 &
bash scripts/exp2/queue_status.sh          # 監視
touch results/prod/exp2/queue/STOP         # 停止
```

### 実験4 — 固定ターゲット摂動 + MATH-500拡張

主要スクリプト: `scripts/run_fixed_target.py`, `analyze_fixed_target_delta.py`,
`results/smoke/run_smoke.sh`, `results/smoke_math/run_smoke_math.sh`,
`results/prod/run_queue.py`, `results/prod_math/run_queue_math.py`

```bash
# スモーク（GSM8K+MMLU）
bash tmp/gpu-locks/run_with_gpu.sh bash results/smoke/run_smoke.sh

# スモーク（MATH-500拡張）
bash results/smoke_math/run_smoke_math.sh

# 本番キュー投入（v1 25設定）
nohup uv run --no-sync python results/prod/run_queue.py >> results/prod/queue.log 2>&1 &
touch results/prod/STOP    # 停止
bash results/prod/run_delta_table.sh   # キュー後の delta table 集計

# 本番キュー投入（MATH-500, 6モデル）
setsid nohup uv run --no-sync python results/prod_math/run_queue_math.py \
    >> results/prod_math/queue_math.log 2>&1 < /dev/null &
touch results/prod_math/STOP_MATH   # 停止
```

### 実験5 — マッチド統制条件

主要スクリプト: `scripts/exp5/make_matched_twin_dataset.py`, `run_generation_queue.sh`,
`run_all_matched_datasets.sh`, `analyze_matched_control.py`, `queue_progress.py`, `summarize_prod.py`

```bash
# スモーク
bash tmp/gpu-locks/run_with_gpu.sh bash -c \
  'uv run python scripts/rebuttal/run_generation_only.py \
     --model google/gemma-3-4b-it --benchmark gsm8k \
     --perturbed_data data/exp5/matched_rnd/gemma-3-4b-it_gsm8k_k4_matched_rnd/perturbed_dataset.json \
     --gpu_id "$CUDA_VISIBLE_DEVICES" --batch_size 4 --limit 24 --output_dir results/exp5/perturbed'

# データセット構築（25設定, CPU）
bash scripts/exp5/run_all_matched_datasets.sh

# 本番生成キュー投入
nohup bash scripts/exp5/run_generation_queue.sh > logs/exp5/queue.log 2>&1 &
uv run --no-sync python scripts/exp5/queue_progress.py --print   # 監視
uv run --no-sync python scripts/exp5/summarize_prod.py \
  --archive $ARCHIVE --output_dir results/prod/exp5
```

### 実験6 — AttnLRP帰属手法収束性

主要スクリプト: `scripts/run_attribution_family.py`, `run_attribution_family_queue.sh`,
`run_loo_scoring.py`, `run_loo_production_queue.sh`, `analyze_attribution_family.py`,
`analyze_loo_rankings.py`

```bash
# LOOスモーク
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_loo_scoring.py \
  --run_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --model google/gemma-3-4b-it --benchmark gsm8k --n 16 --output_dir results/smoke

# LOO本番キュー投入
setsid nohup bash scripts/run_loo_production_queue.sh > results/loo/queue/logs/worker_$$.out 2>&1 &

# attribution-family本番キュー投入（gxi/ig/rollout）
setsid nohup bash scripts/run_attribution_family_queue.sh \
    > results/attribution_family/queue/logs/worker_$$.out 2>&1 &

# 集計
uv run python scripts/analyze_attribution_family.py \
  --results_root results/attribution_family --archive_analysis_root $ARCHIVE/outputs/analysis
uv run python scripts/analyze_loo_rankings.py \
  --loo_root results/loo --archive_analysis_root $ARCHIVE/outputs/analysis --mode occ
```

### 実験7 + 11 + 12 + 16 + 17 — 誤字修正器・連鎖媒介・R_C組成・統一GLMM・行動修復

主要スクリプト: `scripts/exp7/` 配下（`make_corrected_dataset.py`, `smoke_correct_and_eval.py`,
`analyze_correction.py`, `within_run_flip.py` ほか）+ `scripts/exp7/prod/` の4本のキュースクリプト。
実験11/12/16/17 は `analysis/exp11_chain_mediation/`, `analysis/exp12_rc_composition/`,
`analysis/exp16_unified/`, `analysis/exp17_behavioral_repair/` 配下（argparse なし、固定パスで実行）。

```bash
# 実験7: スモーク（修正→レストレーション→評価の1サイクル）
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp7/smoke_correct_and_eval.py \
  --corrector llm --n 16 --benchmark gsm8k \
  --input <perturbed_dataset.json> \
  --eval_model google/gemma-3-4b-it --output results/smoke/smoke_cycle_llm.json

# 実験7: 本番キュー投入（4本、scripts/exp7/prod/ 配下）
nohup bash scripts/exp7/prod/run_pyspell_grid.sh > logs/exp7/pyspell_grid.log 2>&1 &
nohup bash scripts/exp7/prod/run_correction_queue.sh neural > logs/exp7/correction_queue_neural.log 2>&1 &
nohup bash scripts/exp7/prod/run_eval_generation_queue.sh > logs/exp7/eval_queue.log 2>&1 &
setsid nohup bash scripts/exp7/prod/run_within_run_queue.sh phase1 2 0 \
    > logs/exp7/within_run_queue_w0.log 2>&1 &

# 実験7: 集計
uv run python scripts/exp7/aggregate_within_run.py \
  --output results/prod/exp7/within_run/within_run_summary.json \
  --output_md results/prod/exp7/within_run/within_run_summary.md
uv run python scripts/exp7/aggregate_final_results.py \
  --output results/prod/exp7/analysis/final_summary.json \
  --output_md results/prod/exp7/analysis/final_summary.md

# 実験11（analysis/exp11_chain_mediation/ から実行、CPU）
python dump_qwen_dedup_exclude.py
python run_exp11.py

# 実験12（analysis/exp12_rc_composition/ から実行、CUDA_VISIBLE_DEVICES="" でGPU不要）
python run_exp12.py

# 実験16（analysis/exp16_unified/ から実行。実験11・12の出力が前提）
python ../exp11_chain_mediation/run_exp11.py
python ../exp12_rc_composition/run_exp12.py
python build_features.py

# 実験17（analysis/exp17_behavioral_repair/scripts/ から実行）
python3 exp17_analysis.py
```

### 実験8 — アクティベーションパッチング

主要スクリプト: `scripts/exp8/run_patching.py`, `scripts/exp8/prod/run_prod_queue.sh`

```bash
# 本番キュー投入（3モデル×2ベンチマーク。CUDA_VISIBLE_DEVICES=0 固定、GPUロック不使用）
setsid nohup bash scripts/exp8/prod/run_prod_queue.sh > logs/exp8/prod/queue.log 2>&1 < /dev/null &
```

スモーク専用のシェルスクリプトは存在しない。`run_patching.py` のモジュール docstring に
`--n-pairs 16 --noop-check` を付けた小規模実行例が記載されている（`--model`, `--benchmark`,
`--baseline-dir`, `--perturbed-dir-lxt`, `--output-dir` が必須引数）。

### 実験8-fine — 単層注入局在

主要スクリプト: `scripts/exp8/run_patching_fine.py`, `scripts/exp8/prod_fine/smoke_fine.sh`,
`scripts/exp8/prod_fine/run_fine_queue.sh`, `scripts/exp8/analyze_fine.py`

```bash
# スモーク（typo/semantic 両モード、n-pairs=16）
setsid nohup bash scripts/exp8/prod_fine/smoke_fine.sh > logs/exp8_fine/smoke.log 2>&1 < /dev/null &

# 本番キュー投入（3モデル×2ベンチマーク、typo/semantic 両パス）
setsid nohup bash scripts/exp8/prod_fine/run_fine_queue.sh > logs/exp8_fine/queue.log 2>&1 < /dev/null &

# 集計
uv run --package typo-cot python scripts/exp8/analyze_fine.py \
    --results-dir results/prod/exp8_fine --out-dir analysis/exp8_fine
```

### 実験9 — 内部修復

主要スクリプト: `scripts/exp9/smoke.sh`, `run_inner_repair.py`, `make_shards.py`,
`queue_worker.sh`, `queue_status.sh`, `analyze_inner_repair.py`

```bash
# スモーク（forward-onlyでlxt4/random4を各n=32/16）
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
  bash scripts/exp9/smoke.sh

# 本番キュー投入
uv run python scripts/exp9/make_shards.py   # results/exp9/queue/shards_active.tsv 生成
cd projects/typo-cot && setsid nohup bash scripts/exp9/queue_worker.sh \
    < /dev/null >> logs/exp9/worker_w1.log 2>&1 &
bash scripts/exp9/queue_status.sh   # 監視
touch results/exp9/queue/STOP       # 停止

# 集計
uv run python scripts/exp9/analyze_inner_repair.py \
    --input-dir results/smoke/exp9 --output-dir results/smoke/exp9/analysis
```

### 実験10 — スコープ拡張（natural typo / R1蒸留 / Qwen摂動 / MATH-500）

4つのサブスコープが `scripts/exp10_math500/`, `scripts/exp10_natural_typo/`,
`scripts/exp10_qwen_perturbed/`, `scripts/exp10_r1_perturbed/` に分かれている。
いずれも `run_queue.sh <driver-id>` 形式（`<A|B>` や `<G|M>` は摂動条件・ベンチマークを表す
ドライバID）でキュー投入する。

```bash
# MATH-500拡張: 本番キュー投入（2ドライバ並行）
nohup bash scripts/exp10_math500/run_queue.sh A >> logs/exp10_math500/driverA.log 2>&1 &
nohup bash scripts/exp10_math500/run_queue.sh B >> logs/exp10_math500/driverB.log 2>&1 &
# 検証ゲート（1モデル分完了後に手動実行してから次に進む）
uv run --no-sync python scripts/exp10_math500/verify_vs_archive.py --model_short gemma-3-1b-it
touch scripts/exp10_math500/VERIFY_OK

# R1蒸留摂動: データセット作成 + 本番キュー投入
PYTHONHASHSEED=42 uv run --no-sync python scripts/exp10_r1_perturbed/create_datasets.py
setsid nohup bash scripts/exp10_r1_perturbed/run_queue.sh A \
    >> logs/exp10_r1_perturbed/driverA.log 2>&1 < /dev/null &

# Qwen摂動: データセット作成 + 本番キュー投入
PYTHONHASHSEED=42 uv run --no-sync python scripts/exp10_qwen_perturbed/create_datasets.py
setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh A \
    >> logs/exp10_qwen_perturbed/driverA.log 2>&1 < /dev/null &

# natural typo: 分布推定 + データセット作成 + 本番キュー投入 + A/B比較
uv run --no-sync python scripts/exp10_natural_typo/estimate_distribution.py
PYTHONHASHSEED=42 uv run --no-sync python scripts/exp10_natural_typo/create_datasets.py
setsid nohup bash scripts/exp10_natural_typo/run_queue.sh G \
    >> logs/exp10_natural_typo/driverG.log 2>&1 &
uv run --no-sync python scripts/exp10_natural_typo/compare_ab.py
```

### 実験13 — 読み出し集中度（Gini）

主要スクリプト: `scripts/run_attention_concentration.py`, `run_exp13_attention_queue.sh`,
`run_loo_scoring.py`, `run_exp13_loo_queue.sh`, `compute_loo_concentration.py`, `aggregate_exp13.py`

```bash
# LOOスモーク（実験6と共通のスクリプト）
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_loo_scoring.py \
  --run_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --model google/gemma-3-4b-it --benchmark gsm8k --n 16 --output_dir results/smoke

# 本番キュー投入（attention concentration / LOO 拡張）
setsid nohup bash scripts/run_exp13_attention_queue.sh > results/attention/queue/logs/worker_$$.out 2>&1 &
setsid nohup bash scripts/run_exp13_loo_queue.sh > results/loo/queue/logs/exp13_worker_$$.out 2>&1 &

# 集計
uv run python scripts/compute_loo_concentration.py
uv run python scripts/aggregate_exp13.py
```

### 実験14 — no-CoTショートカット

主要スクリプト: `scripts/exp14_nocot/run_nocot_shard.py`, `make_shards.py`, `queue_worker.sh`,
`queue_status.sh`, `aggregate.py`

```bash
# スモーク相当の小規模実行例（--n 16）
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp14_nocot/run_nocot_shard.py \
  --model google/gemma-3-4b-it --benchmark gsm8k --condition clean \
  --source-dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --output-dir results/exp14_nocot/gemma-3-4b-it_gsm8k_clean --n 16

# 本番キュー投入
python scripts/exp14_nocot/make_shards.py > scripts/exp14_nocot/shards_all.tsv
cd projects/typo-cot && setsid nohup bash scripts/exp14_nocot/queue_worker.sh \
    < /dev/null >> logs/exp14_nocot/worker_w1.log 2>&1 &
bash scripts/exp14_nocot/queue_status.sh   # 監視
touch results/exp14_nocot/queue/STOP       # 停止
```

### 実験15 — パッチ→自由生成

主要スクリプト: `scripts/exp15/run_free_generation.py`, `scripts/exp15/prod/run_queue.sh`,
`scripts/exp15/aggregate.py`

```bash
# スモーク
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/exp15/run_free_generation.py \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --baseline-dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --perturbed-dir-lxt $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --n-pairs 16 --levels early late --directions denoise --noop-check \
  --output-dir results/exp15/smoke_gemma_gsm8k

# 本番キュー投入（3モデル×2ベンチマーク）
setsid nohup bash scripts/exp15/prod/run_queue.sh > logs/exp15/prod/queue.log 2>&1 < /dev/null &

# 集計
python scripts/exp15/aggregate.py --results-root results/exp15
```

この worktree は `.venv` とヘルパースクリプトをメインリポジトリから絶対パスで参照するため、
`run_queue.sh` は `MAIN_REPO`（既定値: メインリポジトリの絶対パス）環境変数を内部で使用する。

## 結果・仮説の正本

実験結果の数値・仮説判定は README では要約せず、以下の正典ドキュメントを直接参照する。

- [docs/hypothesis_registry.md](docs/hypothesis_registry.md) — 全仮説（H1〜H18、実験1〜18に対応）の事前登録予測と Phase A/B 判定の正典
- [docs/all_results_by_setting.md](docs/all_results_by_setting.md) — 全設定（モデル×ベンチマーク×摂動条件）の結果表
- [docs/experiment_details.md](docs/experiment_details.md) — 各実験の詳細な実行記録・パラメータ
- [docs/discussion_family_effects.md](docs/discussion_family_effects.md) — モデルファミリー間の効果差に関する考察
- [docs/experiments_11_18_plan.md](docs/experiments_11_18_plan.md) — 実験11〜18（ERDC拡張）の計画・スケジュール
- [docs/size_ladder_plan.md](docs/size_ladder_plan.md) — 実験19（サイズラダー）の計画
- [docs/improvement_plan.md](docs/improvement_plan.md) — 論文改善計画
- [docs/paper_outline.md](docs/paper_outline.md) — 論文アウトライン
- [docs/followup_plan_20260719.md](docs/followup_plan_20260719.md) — 2026-07-19時点のフォローアップ計画
- [docs/experiment_plan.md](docs/experiment_plan.md) — ARR 2026年8月再投稿の実験計画 v2（実験1〜10・実行計画・実装マッピング、初版）
- [docs/work_items.md](docs/work_items.md) — 実験計画 v2 §5/§7 に基づくチェックボックス式の作業分解
- [docs/v1_run_manifest.md](docs/v1_run_manifest.md) — v1 25 設定の run manifest（正典）
- [docs/data_provenance.md](docs/data_provenance.md) — アーカイブデータの出自と参照方法
- [docs/rebuttal_draft.md](docs/rebuttal_draft.md) — リバッタル草稿
- [docs/implementation_reference.md](docs/implementation_reference.md) — 実装リファレンス

## レイアウト

```
typo-cot/
├── configs/paths.yaml        # アーカイブデータへの読み取り専用参照（正典）
├── src/typo_cot/             # 旧 attn_perturbation（ロジック不変で移行）
│   ├── config.py             # pydantic ExperimentConfig
│   ├── data/loader.py        # ベンチマークローダ（HF datasets）
│   ├── models/{wrapper,prompts}.py  # lxt ラップ付きモデルラッパ / CoT プロンプト
│   ├── lrp/analyzer.py       # AttnLRP 帰属 (R_Q / R_C)
│   ├── perturbation/         # typo 生成 + 摂動データセット作成
│   ├── evaluation/extractor.py
│   ├── analysis/             # metrics / PerturbationAnalyzer / appendix
│   └── visualization/        # figures / tables / heatmap / aggregators
├── scripts/                  # Phase 1–5 CLI + rebuttal/ + パイプライン .sh + 実験別サブディレクトリ
│                             # （exp01_03/, exp2/, exp5/, exp7/, exp8/, exp9/, exp10_*/,
│                             #   exp14_nocot/, exp15/ など。各実験の詳細は上記「実験一覧」参照）
├── analysis/                  # 実験11/12/16/17 など、argparse を持たない固定パス実行の分析スクリプト
├── tests/                     # 移行済み unit tests（モック使用、GPU 不要）
├── docs/                      # 実装リファレンス・run manifest・data_provenance.md・仮説registry・結果表
├── data/                      # gitignore。test_perturbed/ スモークフィクスチャ・実験別データセット
└── results/                   # gitignore。新規実行の出力先
```

## テスト

```bash
cd projects/typo-cot
uv sync --extra lrp   # tests は torch / transformers をモック込みで import する
uv run pytest
```

## 依存とバージョンの注意

- 旧リポジトリの解決バージョン（アーカイブの `uv.lock` が一次資料）:
  **Python 3.11, torch 2.9.1, transformers 4.57.6, lxt 2.1, numpy 2.4.1, datasets 4.5.0**。
- 本ワークスペースは **Python ≥3.12, torch 2.10.x, transformers 4.57.6**（vLLM が torch を強く拘束）。
  transformers は完全一致（lxt が transformers 内部を monkey-patch するため最重要）。
- **要スモークテスト**: torch 2.10 + Python 3.12 での AttnLRP 数値が旧結果と一致するか、
  同一シード・同一プロンプトで R_Q/R_C を照合してから再生成に使うこと。

## Troubleshooting

- **`uv sync` が他プロジェクトのパッケージを .venv から消す**: 本モノレポは共有ルート
  `.venv` の uv workspace のため、typo-cot で `uv sync --extra ...` を実行すると
  もう一方のワークスペースパッケージ（例: `quant_typo_neuron`）が prune されることがある
  （セットアップ検証で確認済み）。影響を受けたプロジェクトのディレクトリで普通に
  `uv run <cmd>` を実行すれば再インストールされる。
- **`tmp/gpu-locks/run_with_gpu.sh` が見つからない**: このヘルパーはメインの作業ディレクトリ
  （`/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis`）にのみ存在する未追跡ファイル。
  各実験 worktree からは絶対パスで参照する（上記「GPUロックヘルパー」節を参照）。

## 移行時の注意（TODO）

- [ ] AttnLRP 数値回帰テスト（旧 outputs の 1 設定、例: Llama-3.2-1B_gsm8k k4 importance を
      再分析して `outputs/analysis/` と diff）
- [ ] `perturbation/generator.py`（proximity/double_typing/omission）は
      `typo_utils.data.typo` と**意味論も RNG 呼び出し順も異なる**。再現性のため旧実装を
      そのまま保持。統一するなら typo_utils 側に新 TypoType として意図的に移植する。
- [ ] `config.py` は pydantic のまま（モノレポ規約は OmegaConf YAML）。デフォルト値を
      変えないよう、変換は別 PR で慎重に行う。
- [ ] GPU 引数は旧来の `--gpu_id`（モノレポ規約は `--gpu-ids`）。挙動を変えないため据え置き。
- [ ] スクリプトの読み込みデフォルトは CWD 相対の `outputs/`・`datasets/` のまま。
      アーカイブ読み込み時は `configs/paths.yaml` のパスを明示的に渡す（ハードコード禁止）。
- [ ] `scripts/upload_to_wandb.py` は `typo_utils.tracking.ExperimentLogger` への置き換え候補。
- [ ] rouge_l / jaccard@k などのメトリクスは安定後に `typo_utils.eval.metrics` へ昇格候補。
