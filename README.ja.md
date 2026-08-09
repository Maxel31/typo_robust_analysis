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

リファクタは操作単位で公開しています。`catalogued` は論文上の実験契約と将来の
コマンドが確定しているものの、公開runnerが未実装であることを表します。
`implemented` は実行可能です。カタログが状態を明示するため、未実装コマンドを
README上で実行可能とは扱いません。

GPU実験には論文に固定された `lrp` 環境を使用します。CPUだけで行うカタログ確認と
契約テストには、モデルのダウンロードは不要です。

## リポジトリ構成

```text
.
├── .github/workflows/              # PRの自動レビュー
├── _sample_project/                # 当面残す旧プロジェクトテンプレート
├── datasets/                       # 共有データセット置き場とローカルキャッシュ
├── README.md
├── README.ja.md
├── pyproject.toml                 # uvワークスペース
├── uv.lock                        # 共有環境ロック
├── projects/typo-cot/
│   ├── README.md                  # 英語版のセットアップと実行コマンド
│   ├── README.ja.md               # 日本語版のセットアップと実行コマンド
│   ├── docs/                      # 論文の実験契約とprovenance
│   ├── results/                   # git対象外のローカル出力（.gitkeepのみ追跡）
│   ├── src/typo_cot/              # import可能な実装
│   └── tests/                     # CPU単体テスト／契約テスト
├── scripts/new_project.sh         # 当面残す汎用scaffolding helper
└── utils/                         # typo-cotから独立した共有utility
```

## 当面残すワークスペース支援機能

リポジトリ単位の開発環境には、引き続き `utils/` の `typo-utils` ワークスペースmemberが
ありますが、公開パッケージ `projects/typo-cot` は依存していません。リファクタ中は
リポジトリを `uv` ワークスペースとして維持します。`_sample_project/` と
`scripts/new_project.sh` は既存の汎用scaffoldingで、論文再現の導線には含みません。
PRは `.github/workflows/claude-code-review.yml` で自動レビューします。これら支援機能の
削除や追加cleanupは、独立したPRで扱います。

変更は1操作につき1ブランチ・1PRとし、常に `develop` をbaseにします。現在のPRの
CIと対応すべきレビュー指摘を解消してから、次の操作へ進みます。
