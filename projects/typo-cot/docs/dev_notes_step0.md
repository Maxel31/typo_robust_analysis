# Step 0 開発メモ (資産棚卸しと凍結)

担当: exp/step0 ブランチ。共有ドキュメント (experiment_plan.md 等) は編集しない方針のため、
Step 0 の設計判断はここに記録する。

## 成果物

| 成果物 | パス | 備考 |
|---|---|---|
| 統合テーブル io 層 | `src/typo_cot/data/master_table.py` | スキーマ・条件・main/appendix ラベルの凍結 |
| 行構築ロジック | `src/typo_cot/data/master_builder.py` | 純粋関数 (io なし) |
| アーカイブ読み取り層 | `src/typo_cot/data/archive_reader.py` | 読み取り専用・sha256 |
| 再現集計 | `src/typo_cot/analysis/reproduce.py` | 条件別精度・偏相関 |
| config レジストリ | `configs/registry.yaml` + `src/typo_cot/registry.py` | prompt hash / seed / decoding 凍結 |
| 構築スクリプト | `scripts/step0_build_master_table.py` | `--verify` でハッシュ再検証 |
| スモーク検証 | `scripts/step0_smoke_reproduce.py` | 実装完了条件の照合 |
| parquet 実体 | `data/{model}/{benchmark}/{condition}.parquet` | gitignore (再生成可能) |
| 移行 manifest | `data/master_manifest.json` | 移行元 sha256・行数・span失敗数 |

## スキーマ (1 行 = 1 サンプル × 1 モデル × 1 ベンチ × 1 条件)

仕様の列に加え、再現検証に必要な最小限の列を追加した:

- 仕様列: `sample_id / model / benchmark / condition / question_text / cot_text /
  answer_span / answer_pred / answer_gold / flip / cot_rouge_l_f1 /
  cot_jaccard_top{3,5,10,15,20} / r_q / r_c / span_extract_ok / seed / prompt_id`
- 追加列: `is_correct` (条件別精度の再現に必須), `pattern` (correct→incorrect 等),
  `subset`, `original_question`, `perturbed_tokens` (摂動メタ), `source_path` (プロベナンス)
- `r_q` / `r_c` は アーカイブ results.json の `question_top_k_words` / `cot_top_k_words`
  (単語×スコア) の JSON 文字列。トークン別の生 relevance (.pt) は容量の都合で
  parquet に入れず、`source_path` の隣の `importance_scores/{sample_id}.pt` を参照する。
- `answer_span` = 現行 extractor の `extract_strict()` が返す canonical スパン文字列
  (strict 未検出は NA)。`answer_pred` = アーカイブに記録された `extracted_answer`
  (fallback パターン込み)。`span_extract_ok` = strict 検出の成否。

## 条件名の対応 (凍結)

| master table | アーカイブ suffix |
|---|---|
| clean | (baseline) |
| lxt1/2/4/8 | k{1,2,4,8}_importance |
| random4 | k4_random |

bottom_k (k4_bottom_k) は当初仕様の条件列挙に含まれないため未収録だったが、
**2026-07-18 の wave2 取込で `anti_lxt4` として収録済み** (下記 wave2 節)。
**注意**: 旧 analyzer の union 除外は bottom_k を含まない 5 摂動条件で計算されて
おり、本テーブルの `span_extract_ok` から除外集合を正確に再導出できる
(gemma-3-4b-it × gsm8k で 178 件一致を確認)。

## flip / CoT 指標の由来

`flip`・`cot_rouge_l_f1`・`cot_jaccard_top{k}`・`pattern` は再計算せず、アーカイブ
`outputs/analysis/{bench}/{model}/{suffix}/full_results.json` の `sample_results`
から移行した (Step 0 は棚卸しであり再計算しない)。分析から union 除外された
サンプルは該当列が NA になる。`flip.notna()` が旧 analyzer の集計対象と一致する。

## スモーク検証の照合先と結果

実装完了条件 = 統合テーブルからの再計算がアーカイブと一致すること。

1. **条件別精度** (論文 Table 3 相当): `is_correct` 平均 ==
   `outputs/{baseline,perturbed}/*/summary.json` の accuracy ==
   `outputs/figures/table5.csv` の各セル (atol 1e-9)。
2. **偏相関** (論文 Fig.3 系): flip を目的変数、`ROUGE-L` を統制した
   `Jaccard@10` の偏相関 ρ(J|R) と、その逆 ρ(R|J)。旧 analyzer と同一の
   「統制変数への線形回帰残差同士の Pearson 相関」(scipy) で再計算し、
   `outputs/analysis/**/full_results.json` の `partial_correlations`
   (target=answer_changed, n 含む) と一致 (atol 1e-9)。
3. **span 失敗**: 条件別の strict 未検出数を集計し、union 除外の再導出値が
   analysis の集計対象数 (n) と一致。旧 analyzer は before/after の共通
   sample_id のみを対象とするため、期待値は
   |clean ∩ lxt4| − |union除外 ∩ (clean ∩ lxt4)| で計算する
   (摂動データセット生成時に skip されたサンプルが条件により 1〜2 件ある:
   例 gemma-3-1b-it × gsm8k の k4_importance は 1318 行)。

### 全 25 設定の結果 (2026-07-14)

- 精度: summary.json 150/150 セル一致、figures/table5.csv 90/90 セル一致
- 偏相関 (k=10, lxt4): 25/25 設定で ρ(J|R)・ρ(R|J)・n が analysis と一致 (atol 1e-9)
- span 除外整合: 25/25 設定一致。union 除外率 全体 13.19%、
  最大 38.59% (Mistral-7B × GSM8K) — これはアーカイブ analysis の
  metadata (excluded_no_answer_count) から直接計算した値とも一致する。
  experiment_plan.md §4 の「全体 7.28%・最大 31.16%」とは定義が異なる
  (おそらく rebuttal 期の lenient/per-pair 変種)。→ open question として報告。
- ハッシュ検証: `--verify` で移行元 400 ファイルの sha256・150 parquet の行数 OK
- 合計 238,855 行 (25 設定 × 6 条件、うち clean 39,810 行 = 各設定のサンプル数合計)

### 注意: `outputs/figures/table3.csv` は旧版

アーカイブ `outputs/figures/table3.csv` (偏相関表) は span 除外フィルタ適用**前**の
旧 analyzer 出力 (n=1319 系) から生成されたもので、`outputs/analysis` 配下の
v2 値 (n=1141 系) と一致しない (例: gsm8k × gemma-3-4b の ρ(J|R):
旧 -0.5710 / v2 -0.5089)。v1_run_manifest.md の注意書きどおり正典は
`outputs/analysis` であり、スモークの照合先も analysis 配下とした。
table5.csv (精度) は summary.json と完全一致しており問題ない。

## レジストリ (configs/registry.yaml)

- seed=42、greedy decoding (do_sample=false, temperature=0.0, max_new_tokens=512)
  はアーカイブ全 config.json / models/wrapper.py の既定と一致することを確認して凍結。
- prompt は `models/prompts.py` のテンプレートに固定プローブ入力を与えた
  full prompt の sha256 で凍結 (`typo_cot.registry.compute_prompt_hash`)。
  テンプレート本文が 1 文字でも変わるとテスト
  (`tests/test_registry.py::TestValidation::test_prompt_hashes_match_prompts_py`)
  が落ちる。
- 指標の本文/付録ラベル (修正C): main = ROUGE-L, Jaccard@10, flip。他は appendix。
  `master_table.METRIC_SCOPE` と registry.yaml の二重定義は validate_registry で同期検証。

## 再構築手順

```bash
# 全 25 設定 (アーカイブ読み取りのみ、GPU 不要、~3 分)
uv run python scripts/step0_build_master_table.py
# 移行ハッシュ・行数の再検証
uv run python scripts/step0_build_master_table.py --verify
# スモーク (25 設定全照合)
uv run python scripts/step0_smoke_reproduce.py
```

## wave2 取込 (2026-07-18): 新規生成 + Anti-LXT-4

150 parquet (v1 25 設定 × 6 条件) に 67 セルを追加し **217 parquet / 342,653 行**
とした。取込元と行数:

| グループ | セル | 行数 | 取込元 |
|---|---|---|---|
| v1 (既存) | 150 | 238,855 | アーカイブ (変更なし、**再ビルドで 150/150 byte 一致**) |
| anti_lxt4 (k4_bottom_k) | 25 | 39,805 | アーカイブ perturbed (analysis 無 → flip/CoT 指標 NA) |
| MATH-500 再生成 (M5 × 3 条件) | 15 | 7,500 | exp-10-scope outputs |
| Qwen2.5-7B (B5 × 3 + math × 3) | 18 | 33,936 | clean(B5) はアーカイブ、他は exp-10-scope |
| R1 蒸留 (gsm8k/math/mmlu × 3 条件) | 9 | 22,557 | exp-10-scope outputs (<think> 形式) |

実装 (TDD、RED→GREEN):

- `master_table.py`: `CONDITIONS` に `anti_lxt4` (suffix `k4_bottom_k`) を追加。
  union 除外の意味論は **`V1_UNION_CONDITIONS` (5 摂動条件) に凍結**
  (旧 analyzer は bottom_k を含まないため。スモークの除外導出もこちらを使用)。
- `master_builder.py`: R1 <think> 形式 (`cot_text` キーを持つレコード) は
  `cot_text` 列 = think 部、strict span 抽出 = `answer_text` (</think> 後の
  読み出し部) のみから行う。切断 (`has_think_close=false`) は answer_text 空 →
  span 失敗として数える。標準レコードの経路は不変。
- `archive_reader.py`: `build_cell_plan(paths, registry)` で 217 セルの取込元
  (baseline/perturbed のルート・analysis 有無・prompt_id) を宣言的に列挙。
  Qwen の B5 clean はアーカイブ、摂動は exp-10-scope という混成もここで解決。
- `registry.yaml`: models += Qwen2.5-7B-Instruct / DeepSeek-R1-Distill-Qwen-7B、
  benchmarks += math (prompt hash 凍結 `math_cot_v1`)、conditions += anti_lxt4、
  `reasoning_prompts` ({bench}_r1_think_v1; chat template 依存のため sha256 なし)。
  R1 は zero-shot chat template / max_new_tokens 4096(gsm8k,mmlu)・8192(math) の
  例外を models 注記に記録 (生成規約は exp-10-scope dev notes 参照)。
- `paths.yaml`: `exp10_outputs` (exp-10-scope worktree の outputs、読み取り専用)。

検証 (2026-07-18):

1. **既存 150 parquet の再現性**: 全 217 セルをフル再ビルド後、
   事前 snapshot と sha256 照合 → **150/150 byte 一致** (思考形式分岐・条件追加は
   既存行に影響しない)。
2. **manifest 検証**: `--verify` で 217 エントリ・移行元 520 ファイルの
   sha256 と parquet 行数 OK。
3. **スモーク**: v1 25 設定で accuracy **175/175** (anti_lxt4 25 セルも
   summary.json と一致 = span失敗率 150 セル表の分母データ完備)、
   table5 90/90、偏相関 25/25、span 除外 25/25 (全体 13.19%、従来どおり)。
   スモークの既定対象は照合先がアーカイブ成果物のため v1 25 設定に固定
   (wave2 は照合先なし)。
4. テスト 201 passed / 24 skipped (GPU系)。

R1 の span 失敗 (strict canonical 未検出) は clean 11.3〜16.2% /
摂動 12.6〜28.0% と v1 モデルより高め (truncation + reasoning 系の書式ゆれ)。
flip/CoT 指標列は wave2 全セルで NA (analysis 未実施のため。実験1 の
再判定パイプラインが本テーブルから計算する)。

## 再構築手順 (wave2 以降)

```bash
# 全 217 セル (アーカイブ + exp-10-scope 読み取りのみ、GPU 不要、~3 分)
uv run python scripts/step0_build_master_table.py
# 移行ハッシュ・行数の再検証 (520 ソースファイル)
uv run python scripts/step0_build_master_table.py --verify
# スモーク (v1 25 設定の全照合; wave2 は対象外)
uv run python scripts/step0_smoke_reproduce.py
```
