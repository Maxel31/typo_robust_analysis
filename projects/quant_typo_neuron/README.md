# quant_typo_neuron — 量子化 × typoニューロン

> **README駆動開発（README-Driven Development）**
> 本READMEは実装に先行して書かれた「実行方法の仕様（spec）」です。ここに書かれたコマンド・入出力・契約（contracts）が各機能実装の受け入れ基準になります。
> 実装が進むまでは記載のコマンドはまだ動作しません（これは意図的です）。各機能PRはこのREADMEに記された I/F に合わせて実装してください。

---

## 1. 概要

LLM の **重み量子化** が **typo（誤字）への頑健性** に与える影響を、機構（FFN ニューロン）レベルで分析する研究プロジェクト。
本プロジェクトのスコープは **M0–M2**（同定 → 量子化 → 評価ハーネス）。M3（効果推定）以降は別途。

- **RQ1（現象）**: 量子化は typo 入力で精度・較正をどの程度悪化させるか
- **RQ2（特異性）**: その悪化は typo に **特有** か（clean 一律劣化と区別できるか）
- **機構**: typo 修正に寄与する FFN ニューロン（typoニューロン）を同定し（M0）、量子化前後の挙動と結びつける

対象モデル（Tsuji et al. EMNLP2025 と整合）: Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / Gemma-2-9B（コア）、スケール用に Llama-3.2-1B/3B・Qwen2.5-14B。環境は B100 96GiB×8 を前提。

---

## 2. セットアップ

モノレポ（uv workspace）の一員。リポジトリルートで依存を解決する。

```bash
# リポジトリルートで（推奨）
uv sync --extra llm --extra quant          # transformers/torch/datasets + gptqmodel/bitsandbytes
# 機構解析（M4以降）を先取りする場合
uv sync --extra llm --extra quant --extra mech   # + nnsight

# このプロジェクト配下で実行（results/ がここに出力される）
cd projects/quant_typo_neuron
```

`.env`（任意, W&B 連携）:

```
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=quant_typo_neuron
```

未設定でも `typo_utils.tracking.ExperimentLogger` がローカル `results/` に必ず保存する（W&B は no-op フォールバック）。

---

## 3. ディレクトリ構成

```
projects/quant_typo_neuron/
├── README.md                  # ← 本ファイル（実行方法の仕様）
├── pyproject.toml             # typo-utils[llm,quant] に依存
├── configs/
│   ├── base.yaml              # 対象モデル群・typo種別・eps・seed の共通定義
│   ├── neuron_identification.yaml                # 同定設定（dataset版・top-k率・responsibility定義）
│   ├── quantization.yaml                # 量子化設定（method×bit・group_size・較正データ）
│   └── robustness_evaluation.yaml                # 評価設定（タスク・条件・seed×5）
├── experiments/
│   ├── neuron_identification/                    # データ構築・同定・再現ゲート
│   ├── quantization/                    # 量子化・重み差分抽出
│   └── robustness_evaluation/                    # 評価ドライバ
├── src/quant_typo_neuron/
│   ├── contracts.py           # ★ 全Mが依拠する共有契約（後述）
│   ├── data/                  # wordnet_id 構築・タスクローダ
│   ├── neuron_identification/                    # responsibility 集計・スコアリング
│   ├── quantization/                    # gptq/awq・bnb・weight_diff
│   └── robustness_evaluation/                    # 結果スキーマ実装・long形式変換
├── data/                      # 中間データ（gitignore、.gitkeep のみ追跡）
└── results/                   # 実験出力（gitignore、.gitkeep のみ追跡）
```

汎用プリミティブは共有パッケージ `typo_utils` 側に置く（再利用・index一貫性のため）:

- `typo_utils.neurons.hooks` — FFN中間活性化 forward hook（M0/M4 共有）
- `typo_utils.quant.{rtn, loader}` — 自前RTN（M1/M5共有）・統一量子化ローダ
- `typo_utils.eval.calibration` — ECE / reliability diagram
- `typo_utils.data.typo`（拡張）— 実typo生成（sub_keyboard/insert/delete/transpose）

---

## 4. 契約（contracts）

全機能が同じ規約を共有する。`src/quant_typo_neuron/contracts.py` に定義（実装は scaffold PR で確定）。

### 4.1 ニューロン index 規約
- **「ニューロン n」= FFN中間次元の1次元**。SwiGLU系で `gate_proj`/`up_proj` の出力次元 = `down_proj` の入力次元。
- `NeuronIndex = tuple[layer:int, dim:int]`
- mask 形式: `dict[int, list[int]]`（layer→dim列）または bool tensor `[num_layers, d_ff]`
- **この規約を M0（同定）・M4（活性化抽出）・M5（重み保護）で必ず共有**する。

### 4.2 量子化 variant registry
```python
@dataclass
class QuantVariant:
    name: str          # 例 "gptq_w4", "nf4", "rtn_w4", "fp16"
    method: str        # "fp16" | "gptq" | "awq" | "nf4" | "int8" | "rtn"
    bits: int | None   # 4 | 8 | None(fp16)
    group_size: int | None
    lib_version: str   # 再現性のため記録
    extra: dict        # zero_point 等

def load_variant(name: str) -> tuple[Model, QuantVariant]: ...
```
- **KVキャッシュは全条件 FP16 固定**（重み量子化だけを操作変数にし交絡を排除）。

### 4.3 結果スキーマ（M3 GLMM 用に**項目単位 0/1** を保持）
1行 = 1項目 × 1条件。`results/<exp>/<run>/items.jsonl` に保存。
```jsonc
{
  "model": "llama-3.1-8b-instruct",
  "method": "gptq", "bit": 4,
  "typo_type": "sub_keyboard", "eps": 1,
  "dataset": "gsm8k", "seed": 0, "item_id": "gsm8k-0123",
  "correct_clean": 1,        // 0/1（平均ではなく生の正誤）
  "correct_typo": 0,         // 0/1
  "conf": 0.81               // 較正用の確信度
}
```
**平均値ではなく項目単位 0/1 を必ず保存する**（M3 の GLMM が item を変量効果に使うため）。

---

## 5. 実行方法

### M0 — typoニューロン同定（最優先・関門）

```bash
# 1) WordNet単語同定データ 3版（clean / typo(t=1) / split=tokenization長を揃える分割）を生成
uv run python experiments/neuron_identification/build_dataset.py --config configs/neuron_identification.yaml
#   → data/wordnet_id/{clean,typo,split}.jsonl

# 2) responsibility 集計 → Δ_n → typoニューロン mask M_n（上位0.5%）/ typoヘッド M_h（上位1.5%）
uv run python experiments/neuron_identification/identify.py --config configs/neuron_identification.yaml
#   → results/neuron_identification/<run>/{delta.npz, neuron_mask.json, head_mask.json}

# 3) 再現ゲート①: ablation 検証（M_n を 0/mean 置換 → typo精度↓ & clean保持、random同数では↓しない）
uv run python experiments/neuron_identification/ablation_gate.py --config configs/neuron_identification.yaml --mask results/neuron_identification/<run>/neuron_mask.json

# 4) 再現ゲート②: seed/定義間の安定性（top-0.5% の Jaccard・層分布の順位相関）
uv run python experiments/neuron_identification/stability_gate.py --config configs/neuron_identification.yaml
```

**ゲート判定**: ① ② を通過 → M1/M2 へ。不通過 → M4/M5 を撤回し診断＋tokenization分析に縮小（仕様書どおり）。

### M1 — 量子化

```bash
# AWQ/GPTQ (GPTQModel, W4/W8, group_size=128, C4 128×2048較正) / NF4・INT8 (bitsandbytes) / 自前RTN
uv run python experiments/quantization/quantize.py --config configs/quantization.yaml
#   → 量子化モデル群（variant registry に登録）

# 重み差分 ΔW = W_fp16 − dequant(W_q) を layer/row/col 単位で抽出（M4 再構成誤差用）
uv run python experiments/quantization/weight_diff.py --config configs/quantization.yaml
#   → results/quantization_weight_diff/<run>/delta_w/...
```

- 第一候補: GPTQModel + bitsandbytes。AutoAWQ/AutoGPTQ は非推奨のため不使用。
- group_size・zero_point・ライブラリ版を全条件で固定し記録（`QuantVariant.lib_version`）。

### M2 — 評価ハーネス（clean/typo × FP16/量子化）

```bash
# 4条件（clean-FP16 / typo-FP16 / clean-Q / typo-Q）で精度・ECE を測定、項目単位0/1を保存
uv run python experiments/robustness_evaluation/evaluate.py --config configs/robustness_evaluation.yaml
#   → results/robustness_evaluation/<run>/items.jsonl  （4.3 のスキーマ）
```

- **typo は本物のみ**: sub_keyboard / insert / delete / transpose。大文字化・emoji・leetspeak・slang は除外。
- eps = `1`（1文字）または比率 `0.05 / 0.10 / 0.20`。**typo生成 seed × 5**。
- ECE は reliability diagram から自前計算（`typo_utils.eval.calibration`）。
- データ: gsm8k / bbh / mmlu / longgen / wordnet_id。reasoning系の既製 typo は R2ATA を利用可。

---

## 6. 結果レイアウト

```
results/
├── neuron_identification/<run>/{delta.npz, neuron_mask.json, head_mask.json, config.json}
├── quantization_weight_diff/<run>/delta_w/...
└── robustness_evaluation/<run>/{items.jsonl, metrics.json, config.json}
```

`analysis/` は **ローカル results/ を正** として読む（W&B 非依存で図が描ける）。

---

## 7. 開発ワークフロー（stacked PR）

**1ブランチ=1機能**（M より細かい単位）で並行実装。**各PRは実依存ブランチを base とする DAG スタック**：独立機能（#2–#9）は `scaffold` を base に並列実装でき、依存機能はその実依存ブランチを base にする。マージはボトムアップ。

| # | ブランチ `feature/quant_typo_neuron/<name>` | base | 主な内容 |
|---|---|---|---|
| 0 | `readme` | `main` | 本README（実行方法の仕様） |
| 1 | `scaffold` | readme | プロジェクト雛形 + 契約(contracts) + utilsスタブ + 依存/lock |
| 2 | `neuron_identification-ffn-hooks` | scaffold | FFN中間活性化 hook（utils） |
| 3 | `neuron_identification-wordnet-dataset` | scaffold | WordNet 3版データ生成 |
| 4 | `quantization-interface` | scaffold | 統一量子化ローダ（utils） |
| 5 | `quantization-rtn` | scaffold | 自前RTN（utils, M5共有） |
| 6 | `robustness_evaluation-typo-generators` | scaffold | 実typo生成（utils typo.py 拡張） |
| 7 | `robustness_evaluation-dataset-loaders` | scaffold | タスクデータ読込 |
| 8 | `robustness_evaluation-ece-calibration` | scaffold | ECE 自前実装（utils） |
| 9 | `robustness_evaluation-result-schema` | scaffold | ItemResult 実装 + long形式変換 |
| 10 | `neuron_identification-responsibility-scoring` | neuron_identification-ffn-hooks | Δ_n・mask 算出（+ wordnet-dataset 取り込み） |
| 11 | `quantization-gptq-awq` | quantization-interface | GPTQ/AWQ 量子化 |
| 12 | `quantization-bnb-nf4-int8` | quantization-interface | NF4/INT8 量子化 |
| 13 | `quantization-weight-diff` | quantization-interface | ΔW 抽出（+ rtn 取り込み） |
| 14 | `neuron_identification-ablation-gate` | neuron_identification-responsibility-scoring | 再現ゲート① |
| 15 | `neuron_identification-stability-gate` | neuron_identification-responsibility-scoring | 再現ゲート② |
| 16 | `robustness_evaluation-runner` | robustness_evaluation-result-schema | 4条件評価ドライバ（統合点） |

> 独立8機能（#2–#9）は `scaffold` を base に並列実装可能。依存機能は実依存ブランチを base にし、複数依存があるものは主依存を base にして残りを PR 本文へ明記する。`pyproject.toml`/`uv.lock` は `#1 scaffold` で確定し、以後の feature では触らない（lock競合回避）。実装は各エージェントが各 `git worktree` で並行。

---

## 8. 検証（受け入れ基準）

- `uv sync --extra llm --extra quant` 成功、`uv run python -c "import typo_utils.neurons.hooks, typo_utils.quant.rtn, typo_utils.quant.loader, typo_utils.eval.calibration"` が通る
- 最小E2E（小型モデル Llama-3.2-1B）: M1 量子化1種 → M2 評価1セル → `items.jsonl` に **項目単位 0/1** が出力される
- M0: `ablation_gate` で M_n の ablation が random を有意に上回って typo精度を落とす（再現ゲート通過）
