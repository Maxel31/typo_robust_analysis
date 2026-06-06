# typo_robust_analysis

LLM の typo（誤字）頑健性を分析・実験するためのモノレポ。
`uv` workspace で共有パッケージ `typo_utils` を全プロジェクトから利用する。

## ディレクトリ構成

```
typo_robust_analysis/
├── pyproject.toml        # uv workspace ルート
├── uv.lock               # 依存の一元管理
├── utils/                # 共有パッケージ typo_utils（typo注入/データ/評価/可視化/記録）
├── _sample_project/      # 新規プロジェクトのコピー元テンプレート
├── projects/             # 実プロジェクト群（_sample_project をコピーして作成）
├── datasets/             # 共有データ（大容量は gitignore / 外部参照）
└── scripts/new_project.sh
```

各プロジェクトの中身:

```
projects/<name>/
├── configs/              # 実験設定 YAML（model/dataset/typo 条件）
├── experiments/
│   ├── reproduction/     # 再現実験
│   └── proposed/         # 提案手法の実験
├── analysis/             # results を読み込み可視化（notebook + 図）
├── src/<name>/           # プロジェクト固有ロジック（成熟したら utils へ昇格）
├── results/              # 実験出力（gitignore）
└── data/                 # 中間・ローカルデータ（gitignore）
```

## セットアップ

```bash
uv sync                  # workspace 全体を解決し typo_utils を editable で導入
cp .env.example .env     # W&B を使う場合は値を設定
```

## 新規プロジェクトの開始

```bash
scripts/new_project.sh repro_attention_typo
cd projects/repro_attention_typo
uv sync                       # この member を共有 .venv に editable で導入
uv run python experiments/reproduction/run.py --config configs/repro_baseline.yaml
```

> 実験はプロジェクトディレクトリ内で実行する（`uv run` がその member を自動で同期し、
> `results/` がそのプロジェクト配下に作られる）。

## 設計方針

- **共有は uv workspace**: `typo_utils` を editable リンク。更新は全プロジェクトに即反映、lock は一元管理。
- **再現/提案/分析の分離**: `experiments/{reproduction,proposed}` で実験、`analysis/` で可視化。
- **設定は YAML**: `configs/*.yaml` を `typo_utils.config.load_config()`（OmegaConf）で読む。
- **記録はローカル + W&B 併用**: 常に `results/<exp>/<run_id>/` に保存し、加えて W&B へlog（未設定時は offline）。
