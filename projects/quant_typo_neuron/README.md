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

> **GPU 指定**: モデルを動かすスクリプト（`build_dataset` / `identify` / `ablation_gate` / `quantize` / `weight_diff` / `evaluate`）は `--gpu-ids` 引数で使用GPUを選べます（例 `--gpu-ids 2,3`、torch import 前に `CUDA_VISIBLE_DEVICES` を設定）。`stability_gate` は mask のみ扱う CPU 処理。実行時は extras 有効化（`uv run --extra llm --extra quant python ...`）。
> **モデル選定**: M0 は「定義文→単語を生成」できる必要があるため **capable モデル推奨**（Tsuji コア: `google/gemma-2-9b` / `Qwen/Qwen2.5-7B` / `meta-llama/Llama-3.1-8B`）。0.5B〜1B 級は正答数が少なく検証に不向き。`<model_slug>` = モデル名末尾（例 `Qwen2.5-7B`）。

### M0 — typoニューロン同定（最優先・関門, Tsuji 完全準拠）

```bash
# 1) 単語同定データ構築（GPU・モデル使用）: original_data.json(62643) を走査し、
#    モデルが generate で正答した項目のみ収集（+ 勾配 importance で最重要トークンを特定）
uv run --extra llm python experiments/neuron_identification/build_dataset.py \
  --config configs/neuron_identification.yaml model=Qwen/Qwen2.5-7B dataset.n_samples=20000 --gpu-ids 2,3
#   → data/<model_slug>/meaning_dataset.json （gitignore）

# 2) typoニューロン/ヘッド同定（GPU）: act_fn 活性の Δ=typo−max(clean,split)、ヘッドは attention entropy の Δ
uv run --extra llm python experiments/neuron_identification/identify.py \
  --config configs/neuron_identification.yaml model=Qwen/Qwen2.5-7B dataset.data_size=2000 --gpu-ids 2,3
#   → results/<model_slug>/{sorted_neurons.pkl, sorted_heads.pkl, neuron_mask.json, head_mask.json, delta.npz}

# 3) 再現ゲート①: invert test（GPU）: top-k typoニューロンを deactivate→単語生成 acc/prob を
#    original/typo/split × top/random で測定。typo に特異的な低下 & clean 保持を判定
uv run --extra llm python experiments/neuron_identification/ablation_gate.py \
  --config configs/neuron_identification.yaml model=Qwen/Qwen2.5-7B dataset.data_size=2000 \
  --neurons results/<model_slug>/sorted_neurons.pkl --heads results/<model_slug>/sorted_heads.pkl --gpu-ids 2,3
#   → results/<model_slug>/ablation_gate.json （orig/typo/split × top/random の絶対 acc/prob + 判定）

# 4) 再現ゲート②: seed/定義間の安定性（CPU）: 複数 seed の neuron_mask の Jaccard・層分布 Spearman
uv run --extra llm python experiments/neuron_identification/stability_gate.py \
  --config configs/neuron_identification.yaml \
  --mask results/<model_slugA>/neuron_mask.json --mask results/<model_slugB>/neuron_mask.json \
  --min-jaccard 0.5 --min-rank-corr 0.7
```

**ゲート判定**: ① ② を通過 → M1/M2 へ。不通過 → M4/M5 を撤回し診断＋tokenization分析に縮小（仕様書どおり）。

### M1 — 量子化

```bash
# AWQ/GPTQ (GPTQModel, W4/W8, group_size=128, C4 128×2048較正) / NF4・INT8 (bitsandbytes) / 自前RTN
uv run python experiments/quantization/quantize.py --config configs/quantization.yaml --gpu-ids 2,3
#   → 量子化モデル群（variant registry に登録）

# 重み差分 ΔW = W_fp16 − dequant(W_q) を layer/row/col 単位で抽出（M4 再構成誤差用）
uv run python experiments/quantization/weight_diff.py --config configs/quantization.yaml --gpu-ids 2,3
#   → results/quantization_weight_diff/<run>/delta_w/...
```

- 第一候補: GPTQModel + bitsandbytes。AutoAWQ/AutoGPTQ は非推奨のため不使用。
- group_size・zero_point・ライブラリ版を全条件で固定し記録（`QuantVariant.lib_version`）。

### M2 — 評価ハーネス（clean/typo × FP16/量子化）

```bash
# 4条件（clean-FP16 / typo-FP16 / clean-Q / typo-Q）で精度・ECE を測定、項目単位0/1を保存
uv run python experiments/robustness_evaluation/evaluate.py --config configs/robustness_evaluation.yaml --gpu-ids 2,3
#   → results/robustness_evaluation/<run>/items.jsonl  （4.3 のスキーマ）
```

- **typo は本物のみ**: sub_keyboard / insert / delete / transpose。大文字化・emoji・leetspeak・slang は除外。
- eps = `1`（1文字）または比率 `0.05 / 0.10 / 0.20`。**typo生成 seed × 5**。
- ECE は reliability diagram から自前計算（`typo_utils.eval.calibration`）。
- データ: gsm8k / bbh / mmlu / longgen / wordnet_id。reasoning系の既製 typo は R2ATA を利用可。

### フル実行（全機能を通しで）

> **前提**: 全機能を一度に動かすには、全 feature PR をマージした **統合状態**（`main`、または全 feature を統合したブランチ）で実行する。個々の feature ブランチには自分の依存しか無いため、M0→M1→M2 を通すには統合済みコードが必要。

```bash
# 0) 統合コードを取得（全PRを main にマージ済みの場合）
cd /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis
git switch main && git pull
uv sync --extra llm --extra quant
cd projects/quant_typo_neuron

# モデル: gated(Llama/Gemma) は `huggingface-cli login`。
#   gated を避けるなら open 小型へ差し替え:
#   for f in neuron_identification quantization robustness_evaluation; do
#     sed -i 's#^model:.*#model: Qwen/Qwen3-0.6B#' configs/$f.yaml
#   done

# ============ M0: typoニューロン同定（Tsuji 完全準拠。capable モデル推奨）============
M=Qwen/Qwen2.5-7B    # 例。Gemma-2-9B / Llama-3.1-8B など。出力先 results/<model_slug>/
# (1) 単語同定データ構築（GPU・モデル使用）: 正答項目のみ収集
uv run --extra llm python experiments/neuron_identification/build_dataset.py \
  --config configs/neuron_identification.yaml model=$M dataset.n_samples=20000 --gpu-ids 2,3
# (2) typoニューロン/ヘッド同定（GPU）→ sorted_neurons.pkl / sorted_heads.pkl / masks / delta.npz
uv run --extra llm python experiments/neuron_identification/identify.py \
  --config configs/neuron_identification.yaml model=$M dataset.data_size=2000 --gpu-ids 2,3
# (3) 再現ゲート①: invert test（GPU）。top-k typoニューロンを deactivate→生成acc/prob を top/random比較
uv run --extra llm python experiments/neuron_identification/ablation_gate.py \
  --config configs/neuron_identification.yaml model=$M dataset.data_size=2000 \
  --neurons results/Qwen2.5-7B/sorted_neurons.pkl --heads results/Qwen2.5-7B/sorted_heads.pkl --gpu-ids 2,3
# (4) 再現ゲート②: 安定性（CPU）。複数モデル/seed の neuron_mask を --mask で並べる
uv run --extra llm python experiments/neuron_identification/stability_gate.py \
  --config configs/neuron_identification.yaml \
  --mask results/Qwen2.5-7B/neuron_mask.json --mask results/<別model_slug>/neuron_mask.json \
  --min-jaccard 0.5 --min-rank-corr 0.7

# ============ M1: 量子化 ============
# (5) 量子化バリアント生成（GPU）。methods/bits はカンマ区切り
uv run --extra llm --extra quant python experiments/quantization/quantize.py \
  --config configs/quantization.yaml --methods gptq,nf4,int8 --bits 4,8 --gpu-ids 2,3
# (6) 重み差分 ΔW 抽出（GPU）。--model-id で対象モデルを直接指定も可
uv run --extra llm --extra quant python experiments/quantization/weight_diff.py \
  --config configs/quantization.yaml --variant rtn_w4 --gpu-ids 2,3

# ============ M2: 評価ハーネス ============
# (7) 4条件評価 → 項目単位0/1（GPU）。データ(datasets/<task>/<split>.jsonl or HF)が必要
uv run --extra llm --extra quant python experiments/robustness_evaluation/evaluate.py \
  --config configs/robustness_evaluation.yaml --gpu-ids 2,3
#   → results/robustness_evaluation/<run>/items.jsonl
```

補足:
- **GPU 指定は引数 `--gpu-ids 2,3`**（実スクリプト内で torch import 前に `CUDA_VISIBLE_DEVICES` を設定）。M0 の `build_dataset`/`identify`/`ablation_gate` はモデルを使うため GPU 必須。`stability_gate` は mask 同士の集合比較のみで CPU（`--gpu-ids` を持たない）。
- M0 の出力先は `<model_slug>`（= `model=` で渡したモデル名の末尾。例 `Qwen/Qwen2.5-7B` → `Qwen2.5-7B`）で決まる。前段 `identify` の `results/<model_slug>/sorted_neurons.pkl`・`sorted_heads.pkl` を後段 `ablation_gate` の `--neurons`/`--heads` に、`neuron_mask.json` を `stability_gate` の `--mask` に渡す。
- **`quantize.py`（実GPTQ）** は `gptqmodel 7.0.0` × `transformers 5.10.2` の非互換に注意（§Caveats）。NF4/INT8/RTN は影響なし。
- 各スクリプトの全フラグは `--help` で確認可（例 `... quantize.py --help`）。

---

## 6. 結果レイアウト

```
results/
├── <model_slug>/                       # M0（Tsuji 準拠）: モデル名末尾でディレクトリが決まる
│   ├── meaning_dataset.json            #   build_dataset の正答データ
│   ├── sorted_neurons.pkl              #   identify: Δ降順の typoニューロン
│   ├── sorted_heads.pkl                #   identify: Δ(entropy)降順の typoヘッド
│   ├── neuron_mask.json / head_mask.json
│   ├── delta.npz                       #   identify: 各ニューロン/ヘッドの Δ
│   └── ablation_gate.json              #   ablation_gate: invert test の acc/prob
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
- M0（Tsuji 準拠・再現ゲート①）: `ablation_gate` の invert test で、top typoニューロンを deactivate すると **typo/split 入力での生成精度・確率が落ち、かつ同数の random ニューロン deactivate では落ちない**（top ≫ random）。`results/<model_slug>/ablation_gate.json` で確認。※効果検証には capable モデル（Qwen2.5-7B / Gemma-2-9B / Llama-3.1-8B 等）が必要。0.5B 級では clean 自体が脆く判定不能。
- M0（再現ゲート②）: `stability_gate` で複数モデル/seed の `neuron_mask` 間の Jaccard ≥ 0.5・層分布順位相関 ≥ 0.7
