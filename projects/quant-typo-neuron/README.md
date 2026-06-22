# quant-typo-neuron

LLMの量子化がtypoに対する頑健性に与える影響を定量的に測定する実験フレームワーク。

## 概要

9種のBaseモデル x 4種の量子化手法(各2ビット幅) x 8ベンチマーク x 6 typo条件の大規模実験を管理する。

### モデル

- Llama-3.2-1B / 3B, Llama-3.1-8B
- Gemma-2-2B / 9B
- Qwen2.5-1.5B / 3B / 7B
- Mistral-7B-v0.3

### 量子化手法

| 手法 | ビット幅 | ライブラリ |
|------|---------|-----------|
| GPTQ | W4, W8 | llm-compressor |
| AWQ | W4, W8 | llm-compressor |
| SmoothQuant | W8A8, W4A16 | llm-compressor |
| QEP | W4, W8 | OneCompression |

### ベンチマーク

| ベンチマーク | 評価方式 | few-shot |
|-------------|---------|----------|
| ARC-Easy | log-likelihood | 0-shot |
| ARC-Challenge | log-likelihood | 25-shot |
| HellaSwag | log-likelihood | 10-shot |
| MMLU | log-likelihood | 5-shot |
| PIQA | log-likelihood | 0-shot |
| GSM8K | generation | 5-shot |
| Wikitext-2 | perplexity | - |
| C4 | perplexity | - |

### Typo条件

- clean (0 typos)
- swap: 1, 2, 4 typos
- random: 4 typos
- replace: 4 typos

## セットアップ

```bash
# 基本依存 + vLLM + 量子化ライブラリ
uv sync --extra vllm --extra quant
```

## 再現手順

### 1. 較正データの準備

```bash
cd projects/quant-typo-neuron
uv run python scripts/prepare_calibration_data.py \
    --dataset wikitext \
    --num-samples 512 \
    --output data/calibration/wikitext.jsonl
```

### 2. few-shot例のキャッシュ

```bash
uv run python scripts/prepare_few_shot.py --output-dir data/few_shot --seed 42
```

### 3. モデルの量子化

```bash
# 全モデル一括
bash scripts/quantize_all.sh --gpu-ids 2,3

# 個別実行
uv run python scripts/quantize_model.py \
    --model meta-llama/Llama-3.2-1B \
    --method gptq --bits 4 \
    --output-dir data/quantized/Llama-3.2-1B/gptq_w4 \
    --gpu-ids 2,3
```

### 4. 実験の実行

```bash
# 全sweep実行
bash scripts/run_all.sh --gpu-ids 2,3

# 個別実行
uv run python experiments/run_eval.py --config configs/base_eval.yaml \
    --gpu-ids 2,3 \
    model.name=meta-llama/Llama-3.2-1B \
    benchmark.name=arc_easy \
    typo.type=swap typo.num_typos=4

# 量子化モデル
uv run python experiments/run_eval.py --config configs/quant_eval.yaml \
    --gpu-ids 2,3 \
    model.name=data/quantized/Llama-3.2-1B/gptq_w4 \
    model.quant_method=gptq model.bits=4
```

### 5. 結果の集約・可視化

```bash
# Jupyter notebook
uv run jupyter lab analysis/analyze.ipynb
```

## 構成

```
configs/               実験設定YAML
  base_eval.yaml       基本設定
  typo_eval.yaml       typo実験設定
  quant_eval.yaml      量子化モデル設定
  quantization.yaml    量子化パラメータ設定
experiments/
  run_eval.py          評価CLIエントリーポイント
scripts/
  prepare_calibration_data.py  較正データ準備
  prepare_few_shot.py          few-shot例キャッシュ
  quantize_model.py            単一モデル量子化
  quantize_all.sh              一括量子化
  run_all.sh                   全sweep実行
analysis/
  analyze.ipynb                結果分析ノートブック
src/quant_typo_neuron/
  benchmarks/                  ベンチマークローダー群
  analysis/                    集約・可視化
  few_shot.py                  few-shotキャッシュ
  runner.py                    実験パイプライン
  output.py                    結果保存
```

## テスト

```bash
uv run pytest projects/quant-typo-neuron/tests/ -v
```
