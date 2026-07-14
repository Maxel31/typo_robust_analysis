# typo-cot

Typo（文字レベル摂動）× CoT の 2 軸分析 — **ARR 2026年8月再投稿** 実験プロジェクト。

AttnLRP による重要度帰属（R_Q: 質問トークン→最初の CoT トークン, R_C: CoT トークン→最終回答）を使い、
重要語への typo 注入が CoT 生成と正答率に与える影響を分析する。

JSAI2026 リポジトリ (`/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026`,
パッケージ名 `attn_perturbation`) の後継。コードは `typo_cot` にリネームして移行済み、
実験データ（約 96GB）はアーカイブに置いたまま読み取り専用で参照する
（[docs/data_provenance.md](docs/data_provenance.md), [configs/paths.yaml](configs/paths.yaml)）。

## 実験マトリクス（v1, 25 設定）

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

## パイプライン（Phase 1–5）

すべて `projects/typo-cot/` から実行する。GPU は `--gpu_id` CLI 引数で指定
（旧リポジトリ由来の引数名。config には書かない）。

```bash
# Phase 1: ベースライン推論 + AttnLRP 帰属
uv run python scripts/run_inference.py \
    --model meta-llama/Llama-3.2-1B-Instruct --benchmark gsm8k \
    --gpu_id 0 --output_dir ./results/baseline

# Phase 2: 摂動データセット作成（重要度スコアから importance/random/bottom_k 選択）
uv run python scripts/run_perturbation.py \
    --baseline_dir ./results/baseline/... --output_dir ./results/datasets/perturbed

# Phase 3: 摂動後推論（run_inference.py に --perturbed_dataset を渡す）

# Phase 4: before/after 一括分析（アーカイブ出力を読む場合は paths.yaml のパスを渡す）
uv run python scripts/run_analysis.py \
    --outputs_dir /home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs \
    --output_dir ./results/analysis

# Phase 5: 論文 Figure 2a/2b/3, Table 3/5/6
uv run python scripts/build_figures_tables.py \
    --analysis_dir ./results/analysis --output_dir ./results/figures
```

一括実行: `scripts/run_full_pipeline.sh`（環境変数 `GPU_ID`, `OUTPUTS_ROOT`,
`PERTURBATION_MODES`, `K_LIST`, `SKIP_PHASES`）、`scripts/run_v2_extended_pipeline.sh`,
`scripts/run_v2_parallel.sh`, `scripts/_run_in_tmux.sh`。
ハードコードされた MODELS×BENCHMARKS 配列が run record を兼ねる。

リバッタル実験（spellfix・matched-random 統制、fixed-target 帰属、noise floor、
統計補強）は `scripts/rebuttal/` の 9 スクリプト。

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
├── scripts/                  # Phase 1–5 CLI + rebuttal/ + パイプライン .sh
├── tests/                    # 移行済み unit tests（モック使用、GPU 不要）
├── docs/                     # 実装リファレンス・run manifest・data_provenance.md
│                             # experiment_plan.md / work_items.md（ARR 再投稿計画）
├── data/                     # gitignore。test_perturbed/ スモークフィクスチャ
└── results/                  # gitignore。新規実行の出力先
```

## ドキュメント

- [docs/experiment_plan.md](docs/experiment_plan.md) — ARR 2026年8月再投稿の実験計画 v2（実験1〜10・実行計画・実装マッピング）
- [docs/work_items.md](docs/work_items.md) — 実験計画 v2 §5/§7 に基づくチェックボックス式の作業分解（Step 0 → 実験4→5→7→1→3→2→8→9→6→10 + 並行トラック）
- [docs/v1_run_manifest.md](docs/v1_run_manifest.md) — v1 25 設定の run manifest（正典）
- [docs/data_provenance.md](docs/data_provenance.md) — アーカイブデータの出自と参照方法

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
