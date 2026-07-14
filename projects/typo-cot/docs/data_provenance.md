# データ来歴 (Data Provenance)

このプロジェクトは JSAI2026 リポジトリ
(`/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026`, パッケージ名 `attn_perturbation`)
からコードのみを移行した後継プロジェクトである。実験データ本体（約 96GB）は
アーカイブに置いたまま **読み取り専用** で参照する。パスの正典は
[`configs/paths.yaml`](../configs/paths.yaml)。

## アーカイブに残したもの（コピーしない）

| アーカイブ内パス | サイズ | 内容 | 参照方法 |
| --- | --- | --- | --- |
| `outputs/baseline/` | 21GB | 90 dirs (`{model}_{benchmark}/`)。Phase 1 のベースライン生成ログ + AttnLRP relevance | `paths.yaml: archive_baseline` を `--outputs_dir` 等に渡す |
| `outputs/perturbed/` | ~70GB | 322 dirs。Phase 3 の摂動後生成ログ + relevance | `paths.yaml: archive_perturbed_outputs` |
| `outputs/analysis/` | 621MB | Phase 4 のベンチマーク別 `analysis_results.json` | `paths.yaml: archive_analysis` |
| `outputs/rebuttal/` | 1.3GB | ARR リバッタル実験の出力（spellfix / matched_random / noise floor 等） | `paths.yaml: archive_rebuttal_outputs` |
| `outputs/figures/` | ~2MB | 論文最終 figure2a/2b/3, table3/5/6 (csv/tex) | `paths.yaml: archive_figures` |
| `datasets/perturbed/` | 1.1GB | 360 dirs (`config.json` + `perturbed_dataset.json`)。Phase 2 の摂動データセット | `paths.yaml: archive_perturbed_datasets` |
| `datasets/rebuttal/` | 23MB | spellfix / matched_random 統制データセット（`scripts/rebuttal/make_*_dataset.py` で再生成可能） | `paths.yaml: archive_rebuttal_datasets` |
| `uv.lock` (旧) | 828KB | 当時の解決バージョンの記録: Python 3.11, torch 2.9.1, transformers 4.57.6, lxt 2.1, numpy 2.4.1, datasets 4.5.0 | バージョン照合の一次資料として参照 |
| `docs/my_paper.pdf` | ~4MB | 投稿論文 PDF | 必要ならパス参照 |

wandb/ キャッシュ・.venv・旧 .env（秘密情報）はコピーせず参照もしない。

## このリポジトリにコピーしたもの

- `src/attn_perturbation/` → `src/typo_cot/`（パッケージ名変更のみ、ロジック不変）
- `scripts/` → `scripts/`（`sys.path` ハック除去のみ）
- `tests/` → `tests/`（import の rename のみ）
- `docs/`: implementation_reference.md, v1_run_manifest.md（25 設定の run manifest 正典）,
  rebuttal_draft.md, drawio 図 3 点
- `datasets/test_perturbed/` (268KB) → `data/test_perturbed/`（小さなスモーク用フィクスチャ。
  注: `data/` は gitignore 対象なのでローカル便宜品。正典はアーカイブ側）

## 実験設定の正典

25 設定 = {Llama-3.2-1B/3B-Instruct, gemma-3-1b/4b-it, Mistral-7B-Instruct-v0.3}
× {gsm8k, mmlu, mmlu_pro, arc, commonsense_qa}、摂動条件は
k=1/2/4/8 importance + k4 random + k4 bottom_k。
詳細は [`v1_run_manifest.md`](v1_run_manifest.md)（v1/v2 アナライザの差異
＝answer-span 除外フィルタの有無もここに記載）。

## 読み書きの規約

- アーカイブ (`/home/...`) は**読み取り専用**。スクリプトの `--output_dir` 系の引数に
  アーカイブパスを渡してはならない。
- 新規実行の出力は `projects/typo-cot/results/`（または旧デフォルトの
  プロジェクト内 `outputs/`・`datasets/`、いずれも gitignore 済み）へ。
- アーカイブはモノレポと別ファイルシステム (/home vs /diskthalys, NFS) にあるため、
  パスは必ず `configs/paths.yaml` を経由し、コードにハードコードしない。
