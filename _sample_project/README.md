# sample-project

LLM の typo 頑健性に関する 1 テーマを扱うプロジェクトテンプレート。
`scripts/new_project.sh <name>` でこのディレクトリをコピーして使う。

## 構成

- `configs/` … 実験設定 YAML。`repro_*.yaml`（再現）/ `proposed_*.yaml`（提案手法）。
- `experiments/reproduction/` … 既存研究の再現実験。
- `experiments/proposed/` … 提案手法の実験。
- `analysis/` … `results/` を読み込み図を生成（notebook + 図）。
- `src/sample_project/` … このプロジェクト固有のロジック（共有化できそうなら `utils` へ昇格）。
- `results/`, `data/` … 実験出力・中間データ（gitignore）。

## 実行

このプロジェクトディレクトリ内で実行する（`results/` がここに作られる）。

```bash
uv sync
uv run python experiments/reproduction/run.py --config configs/repro_baseline.yaml
uv run python experiments/proposed/run.py     --config configs/proposed_method.yaml
```

設定はコマンドラインで上書きできる:

```bash
uv run python experiments/reproduction/run.py --config configs/repro_baseline.yaml typo.rate=0.2
```
