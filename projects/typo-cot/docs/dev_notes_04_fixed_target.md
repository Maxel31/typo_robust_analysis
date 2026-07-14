# 実験4 (fixed-target 化) 開発メモ

ブランチ: `exp/04-fixed-target` / 最終更新: 2026-07-14

## 実装物

| ファイル | 役割 |
|---|---|
| `src/typo_cot/attribution/fixed_target.py` | コアロジック: 回答パターン検出 (`ANSWER_PATTERNS` / `find_answer_match`)、splice 計画 (`plan_splice` / `plan_run`)、`compare_cot_payloads` (検証用)、GPU 依存部 (`build_prompt` / `analyze_cot_fixed`) |
| `src/typo_cot/analysis/fixed_stats.py` | 統計層: `partial_corr_flip` (ρ(J\|R))、`bootstrap_partial_corr_ci`、`holm_adjust`、`paired_bootstrap_delta_rho` (Δρ)、`join_fixed_default_records`、`cot_jaccard_from_scores` |
| `src/typo_cot/data/run_io.py` | アーカイブ run ディレクトリへの薄い読み取り層 (Step 0 master table に一行で差し替える前提) |
| `scripts/run_fixed_target.py` | 全設定一般化ランナー (`--flip_only` / `--sample_ids(_file)` / `--compare_dir` / `--compare_default`) |
| `scripts/analyze_fixed_target_delta.py` | Δρ 全設定表 (付録用) の生成 (JSON+CSV, bootstrap CI + Holm) |
| `tests/test_fixed_target.py` / `tests/test_fixed_stats.py` | GPU 不要のユニットテスト |

## 規約の凍結 (rebuttal 実装との同一性)

- `ANSWER_PATTERNS` は `scripts/rebuttal/run_fixed_target_attribution.py` と同一
  (テスト `test_patterns_match_rebuttal_reference` で同一性を機械検証)。
- flip 判定は生成テキスト中の生スパン文字列比較 (大文字化しない)。
- splice: 摂動側テキストの回答スパンだけを baseline 回答文字列に置換。
  非 flip は spliced_text == 元テキスト → R_C^fixed = R_C^default (定義上同値)。
- AttnLRP: `analyze_cot_fixed` は既存 `analyze_combined` の CoT→Answer パスを
  同一手順で実行 (回答トークン位置の logit backward、自由記述は位置平均 =
  `compute_relevance(target_position=...)` の既存規約)。
- ρ(J|R): analyzer `_compute_partial_correlation` と同じ残差 Pearson (一次偏相関)。
  r は pingouin.partial_corr と厳密一致 (テストで検証)。p は dof=n-3 の t 分布
  (pingouin 準拠; analyzer 旧実装は scipy.pearsonr の dof=n-2 で p がわずかに異なる)。

## CPU 検証結果 (2026-07-14, GPU 不要パス)

1. **plan_run の完全再現** (`results/smoke/plan_validation_vs_rebuttal.json`):
   rebuttal 4設定 (Gemma-3-4B/Llama-3.2-3B × GSM8K/MMLU) 全 7,771 サンプルで
   統計カウント・skipped_ids・splice メタデータ (baseline/perturbed answer, spliced)
   が rebuttal 出力と全一致 (mismatch 0)。
2. **ρ の完全再現**: アーカイブ `analysis_exp1` の full_results.json から
   `join_fixed_default_records` + `partial_corr_flip` で計算した ρ(J@10|R) が
   rebuttal の報告値と小数4桁一致:
   - GSM8K: Gemma −0.5089→−0.5455 / Llama −0.5292→−0.5396
   - MMLU: Gemma −0.4932→−0.1750 / Llama −0.6181→−0.1088

## 全設定展開の手順 (基盤生成完了後)

1. Step 0 master table (または既存アーカイブ dirs) から各設定の
   baseline/perturbed ペアを列挙。
2. `scripts/run_fixed_target.py --flip_only` で flip 事例のみ AttnLRP 再計算
   (非 flip は default の `_cot.pt` を再利用 — 定義上同値)。
3. `scripts/rebuttal/run_rebuttal_analysis.py` (union 除外を再現する既存ドライバ)
   で fixed_target ディレクトリを分析 → full_results.json。
4. `scripts/analyze_fixed_target_delta.py` で Δρ 全設定表 (B=10,000, Holm)。

## 未実装 / 別途判断が必要

- GLMM 再推定 (R lme4 / glmmTMB): R 環境が必要。未着手。
- 形式間メタ比較 (自由記述 vs 多肢選択の Δρ): 全設定の Δρ 表が揃ってから。
  `delta_rho_table.json` を入力にする小スクリプトで足りる。
- Figure 3 再生成 (fixed 版差し替え) と散布図代表例: 既存
  `scripts/build_figures_tables.py` の Fig.3 経路を fixed_target の
  analysis 出力へ向ければよい (全設定の再計算完了後)。
- MMLU-Pro/ARC/CSQA/MATH-500 と他モデルは基盤生成 (Qwen/MATH は実験10) 待ち。
