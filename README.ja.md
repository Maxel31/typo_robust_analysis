# Edited-Word Activation Patching

[English](README.md) | [日本語](README.ja.md)

論文 **“Edited-Word Activation Patching Reverses Selected Typo-Induced Answer
Changes after Tokenization.”** の再現実験コードです。

公開用の再現パッケージは
[日本語README](projects/typo-cot/README.ja.md) にあります。実験名には
`layerwise-kl-patching`、`cot-swap`、`clean-prefix-scan` のように、実際に
行う操作が分かる名前を使っています。RQ1などの論文中のラベルは相互参照にだけ
使用します。

## 論文と再現手順

まず、以下のセットアップと実験別コマンドを使用してください。最終版19ページの
論文に記載された実験設計、コホート、分母、報告値が、各コマンドで再現する対象です。
必要な場合は、カタログが出力するSHA-256を使って手元の論文PDFを確認できます。

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

手元の論文PDFを確認する場合は、次の2コマンドの出力を比較します。

```bash
sha256sum "/path/to/Edited-Word Activation Patching Reverses Selected Typo-Induced Answer Changes after Tokenization.pdf"
uv run --project projects/typo-cot typo-cot experiments source
```

このハッシュは整合性確認用であり、実験への追加入力ではありません。古いスクリプト、
ブランチ、結果メモ、READMEの記載が最終論文と異なる場合は、論文の仕様を使用します。
転記した実験契約は
[`paper-experiments.md`](projects/typo-cot/docs/paper-experiments.md) にあります。

## クイックスタート

```bash
git clone https://github.com/Maxel31/typo_robust_analysis.git
cd typo_robust_analysis
git switch develop

uv sync --project projects/typo-cot
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments show layerwise-kl-patching
```

カタログに記載された論文の全操作を実装済みです。`experiments list` で、各操作の
compute class、必須引数、出力、実装状態を確認できます。

GPU実験には論文に固定された `lrp` 環境を使用します。CPUだけで行うカタログ確認と
契約テストには、モデルのダウンロードは不要です。

## ARR追加実験とtypo頑健化学習

ARR rebuttalの追加実験と、今後のtypo頑健化学習については、実装より先に
[`rebuttal_analysis_plan_v1.md`](projects/typo-cot/docs/rebuttal_analysis_plan_v1.md)
と
[`robustness_training_plan_v1.md`](projects/typo-cot/docs/robustness_training_plan_v1.md)
でコマンドと解析契約を固定しています。パッケージREADMEでは実験ごとに操作内容が
分かるコマンドを示し、レビュー済み実装がmergeされるまでは
`interface-frozen` と表示します。学習機能は、held-out評価でclean性能を維持しながら
頑健性が改善したことを確認してから公開します。

## リポジトリ構成

```text
.
├── .github/workflows/              # PRの自動レビュー
├── README.md
├── README.ja.md
├── pyproject.toml                  # uvワークスペース
├── uv.lock                         # 固定された再現環境
└── projects/typo-cot/
    ├── README.md                   # 英語版のセットアップと実行コマンド
    ├── README.ja.md                # 日本語版のセットアップと実行コマンド
    ├── docs/                       # 論文の実験契約とprovenance
    ├── results/                    # git対象外のローカル出力（.gitkeepのみ追跡）
    ├── src/typo_cot/               # import可能な実装
    └── tests/                      # CPU単体テスト／契約テスト
```

## リポジトリの範囲

rootの `uv` workspaceには、公開する再現パッケージだけを含めます。model／datasetの
cache、生成した実験結果、提供された論文PDF、過去の図表ソースを監査するローカル
archiveは、意図的にgitで追跡しません。PRは
`.github/workflows/claude-code-review.yml` で自動レビューします。

変更は1操作につき1ブランチ・1PRとし、`develop` をbaseにします。CIと対応すべき
レビュー指摘を解消してから、次の操作をマージします。
