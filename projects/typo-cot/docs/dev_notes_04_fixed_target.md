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

## GPU スモーク: pending (2026-07-14)

ユーザー指示 (tmp/gpu-locks/SMOKE_PAUSED, GPU 3/4 他ユーザー占有中) により未実施。
入力 (`results/smoke/sample_ids_{gsm8k,mmlu}.json`: flip 24 + 非flip 8 each) と
ドライバ (`results/smoke/run_smoke.sh`) は準備済み。再開手順と完了判定は
`results/smoke/SMOKE_PENDING.md` 参照。

## GPU スモーク: 実施済み (2026-07-14, GPU 3 via run_with_gpu.sh)

実行前に 2 つの環境バグを修正 (各 RED→GREEN コミット済み):

1. `transformers<5` を lrp extra に固定 — uv.lock が transformers 5.10.2 に
   解決されており lxt 2.1 が import 不能 (find_pruneable_heads_and_indices
   削除) だった。archive/JSAI2026 の lock と同じ 4.57.6 に解決。
2. `setup_device` が run_with_gpu.sh の設定した CUDA_VISIBLE_DEVICES を
   `--gpu_id` で無条件上書き → 占有中の物理 GPU 0 で OOM。環境変数が
   既にある場合は優先するよう修正。
3. `torch==2.9.1` に固定 (archive lock と同一)。torch 2.12 では bf16
   カーネル差で vs_reference の top10 Jaccard が 4/32 サンプルで <1.0 だった。

結果 (`results/smoke/fixed_target/.../comparison.json`, 各 n=32, 実行 ~30 秒/設定):

- GSM8K: vs_reference min top10_jaccard = **1.0 (32/32) PASS**。
  vs_default は 7/8 が 1.0、`gsm8k_00010` (非flip) のみ 0.818 — ただし
  archive の rebuttal 参照 .pt と default .pt 自体がこのサンプルで不一致
  (top10_j=0.818, max_abs_diff=0.11)。我々の出力は rebuttal 参照と完全一致
  (max_abs_diff=0.002) なので、archive 側 2 参照間の非整合が原因。
- MMLU: vs_reference 31/32 = 1.0、`mmlu_abstract_algebra_0069` (非flip) のみ
  0.818。dedup 後 top10 境界のタイ ('The' vs ')"'、スコア差 ~0.004) が数値
  ノイズで反転したもの。rebuttal 本番は gpu_id='5,6' (マルチGPU) で実行されて
  おり、単一 GPU 3 では完全なビット一致は再現不能。vs_default も同サンプル
  のみ 0.818 (同因)。
- all_cot_range_match = true (両設定)、n_tokens_match 全 true、errors 0。

## 本番ラン (2026-07-14 開始, 25設定)

- 対象: v1 manifest の 5モデル×5ベンチ (LXT-4)。Qwen/MATH は実験10の基盤生成待ち。
- 計画パス (CPU, `results/prod/plan_all_settings.py` → `settings_plan.json`):
  全25設定で flip 合計 13,953 / processed 合計 39,073 (再計算は flip のみ ≈ 36%)。
  最大シャード = gemma-3-1b-it×mmlu (flip 1,244)。
- `--flip_only` の出力を analyzer 直結の完全な run にするため、非flip の
  materialize (default `_cot.pt`/`.pt` の冪等 symlink + results.json エントリ合流)
  を追加 (run_io.link_reused_scores / fixed_target.fixed_target_entry,
  RED→GREEN コミット済み)。default _cot.pt 欠損の非flip は skipped_ids に
  `nonflip_default_cot_pt_missing` で記録され解析から除外される。
- キュー: `results/prod/run_queue.py` (nohup)。1シャード=1設定で
  (1) GPU: run_with_gpu.sh 経由 `run_fixed_target.py --flip_only`
  (2) CPU: `run_rebuttal_analysis.py` (union 除外 + fixed skipped_ids 追加除外) で
  default/fixed 両条件 → `results/prod/analysis/{bench}/{model}/k4_*/full_results.json`。
  シャード間でロック解放 (実験10 と交互)。進捗 `results/prod/progress.json`、
  再開 = 同コマンド再実行 (完了ステップは成果物存在でスキップ)、停止 =
  `results/prod/STOP` 作成。
- 検証設計: rebuttal 済み4設定を再計算してキュー先頭に配置し、`--compare_dir` で
  全 flip の rebuttal 参照 `_cot.pt` と比較 (comparison.json)。Δρ 表には
  provenance 統一のため 25設定とも自前の再計算・再分析を用いる
  (アーカイブ analysis_exp1 は突合参照として使用)。
- Δρ 表: キュー完了後 `bash results/prod/run_delta_table.sh`
  → `results/prod/delta_rho/delta_rho_table.{json,csv}` (B=10,000, Holm)。

## MATH-500 拡張 (2026-07-18, 実験4-MATH)

事前登録分岐の発動 (MC 型で有意消失 → 自由記述主役化) を受け、第2の自由記述
ベンチマーク MATH-500 で fixed-target 再現を行う。データは exp-10-scope
worktree の再生成済み run (読み取りのみ):
`$WT10/outputs/baseline/<model>_math` / `$WT10/outputs/perturbed/<model>_math_k4_importance`。

### 規約上の重要な発見と設計判断

- **既存 MATH .pt はフォールバック規約**: lrp/analyzer の `_find_answer_pattern`
  に boxed パターンが無いため、exp-10 再生成 (=アーカイブ互換) の MATH
  importance_scores は全サンプルで回答スパン未検出 → target=-1 (最終トークン)、
  回答トークンが CoT 範囲に混入した状態で計算されている
  (例: baseline gemma-3-4b math_00000 は cot_token_end = n_tokens-2)。
- **default 側も再計算する**: 上記のまま fixed (boxed 回答トークンへ帰属) と
  対比すると「ターゲット規約の差」と「帰属先回答内容の差」が交絡する。
  そこで摂動側 R_C を default (生成された答えへ帰属)・fixed (元の答えへ帰属)
  の両方とも同一の boxed 回答トークン規約で再計算し、差を帰属先の内容のみに
  純化する (GSM8K の実験4手続きの MATH 対応版)。
- **clean 側はアーカイブ互換 .pt を利用** (指示どおり)。J@10 の clean 側だけ
  フォールバック規約なのは両条件対称なので Δρ の内的妥当性には影響しないが、
  ρ の絶対値は GSM8K と規約が揃わない可能性がある → スモークの
  comparison.json (default再計算 vs フォールバック .pt の top10 Jaccard) で
  規約変更の影響を定量化し、必要なら clean 再計算を感度分析として追加
  (open question)。
- **boxed 検出規約** (`fixed_target.find_boxed_answer`, TDD):
  「最後の閉じた \boxed{...}」を採用・末尾の未閉じ boxed は無視
  (`MATHAnswerExtractor` と同一の採用規約)。中身スパンは文字単位の括弧追跡
  (ネスト対応)。回答スパン開始はトリガー句 `The (final )?answer is[:\s]*\$?`
  があればその先頭。flip 判定は boxed 中身の生文字列比較 (rebuttal 規約踏襲)。
  片側にしか boxed が無いサンプルはスキップ (strict=boxed の union 除外と同じ
  母集団)。\boxed を含まないテキストは従来経路 (GSM8K/MMLU 回帰なし;
  既存テスト 215 passed)。
- **splice**: 摂動テキストの boxed 中身のみを baseline の boxed 中身に置換。
  ターゲット = boxed 中身の先頭トークン (`map_answer_char_spans_to_tokens` は
  analyzer のトークン写像ループと同一規約)。

### 実装 (コミット)

- `test_fixed_target_math.py` (RED) → `fixed_target.py` boxed 対応 (GREEN)
- `run_fixed_target.py`: `--mode default` (splice なし・回答トークン規約で
  default 再計算、出力 suffix `default_recomputed`) / `--reuse_from`
  (非flip 再利用元を default 再計算 dir へ) / `--skip_existing` (サンプル単位の
  冪等再開)。既存 CLI の挙動は不変。

### 計画数 (settings_plan_math.json, 500サンプル中)

| model | processed | flip | union除外後母集団 (flip) |
|---|---|---|---|
| gemma-3-1b-it | 194 | 124 | 162 (103) |
| gemma-3-4b-it | 239 | 42 | 216 (36) |
| Llama-3.2-1B | 328 | 251 | 288 (215) |
| Llama-3.2-3B | 346 | 240 | 313 (212) |
| Mistral-7B | 323 | 271 | 295 (245) |
| Qwen2.5-7B | 292 | 58 | 274 (55) |

GPU 見積: Σ(processed + flip) ≈ 2,708 backward + モデルロード 12 回。

### 運用 (results/ は gitignore のため非追跡の作業ファイル)

- 計画: `uv run --no-sync python results/prod_math/plan_math_settings.py`
- スモーク (gemma-3-4b n=16): `bash results/smoke_math/run_smoke_math.sh`
  → `uv run --no-sync python results/smoke_math/validate_smoke_math.py`
- 本番キュー: `setsid nohup uv run --no-sync python
  results/prod_math/run_queue_math.py >> results/prod_math/queue_math.log
  2>&1 < /dev/null &`
  (1モデル = default GPU パス → fixed GPU パス → CPU 分析 → Δρ 表随時更新。
  再開は同コマンド、停止は `results/prod_math/STOP_MATH` 作成。
  進捗 `results/prod_math/progress_math.json`)
- 分析出力: `results/prod_math/analysis/math/<model>/k4_{default_recomputed,fixed_target}/full_results.json`
- Δρ 表: `results/prod_math/delta_rho/delta_rho_table.{json,csv}`

## 未実装 / 別途判断が必要

- GLMM 再推定 (R lme4 / glmmTMB): R 環境が必要。未着手。
- 形式間メタ比較 (自由記述 vs 多肢選択の Δρ): 全設定の Δρ 表が揃ってから。
  `delta_rho_table.json` を入力にする小スクリプトで足りる。
- Figure 3 再生成 (fixed 版差し替え) と散布図代表例: 既存
  `scripts/build_figures_tables.py` の Fig.3 経路を fixed_target の
  analysis 出力へ向ければよい (全設定の再計算完了後)。
- MMLU-Pro/ARC/CSQA/MATH-500 と他モデルは基盤生成 (Qwen/MATH は実験10) 待ち。
