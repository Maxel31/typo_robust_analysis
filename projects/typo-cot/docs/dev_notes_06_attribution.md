# 実験6 開発メモ（exp/06-attribution ブランチ専用）

このファイルは実験6（attribution 手法の横断比較 + leave-one-out）の実装メモ。
共有ドキュメント（experiment_plan.md / work_items.md / README.md）は編集しない
方針のため、設計判断はここに記録する。

## 今回のスコープ（wave 1）

実験6-(iv) LOO（leave-one-out）のみ。(i)〜(iii)（G×I / IG / occlusion / rollout）
は次 wave で同ブランチに実装する。

## LOO の定義（experiment_plan.md §4 実験6-(iv) に準拠）

- clean CoT の各「語タイプ」について全出現を削除した変種 CoT を作る
- (質問プロンプト + 変種 CoT + 答えトリガー) を teacher-forcing し、
  「元の答えトークン列の log-prob 合計」の低下量をその語の重要度とする
- 生成不要のスコアリングのみ（1変種 = 1 forward）
- 元の答えで測るため構成上 fixed-target（実験4のプロトコルと自動的に整合）

## 実装: `src/typo_cot/intervention/loo_scorer.py`

### 生成テキストの3分割

`split_generated_text(generated_text)` が
`scripts/rebuttal/run_fixed_target_attribution.py` / `lrp/analyzer.py:_find_answer_pattern`
と同一の ANSWER_PATTERNS（同一正規表現・最後のマッチ採用）で

- `cot_text`     = 回答パターン開始位置より前の全テキスト
- `trigger_text` = パターン開始〜答え文字列直前（例: "The answer is "、"The answer is ("）
- `answer_text`  = パターンの group(1)（例: "18"、"B"）

に分割する。回答パターンが無いサンプルはスキップ（stats に記録）。
answer_text より後のテキスト（"." 以降など）はスコアリング対象外。

### 語タイプの定義

CoT:Jaccard の語彙（`tokens_to_words` の空白区切りマージ → word_scores）と揃えるため、
CoT テキストを空白区切りチャンクに分割し、チャンク両端の句読点を剥がしたものを
語タイプのキーとする（case-sensitive）。

- 例: "eggs." と "eggs" は同一タイプ "eggs"（削除時は "eggs" 部分のみ削除し句読点は残す）
- 剥がすと空になる純句読点チャンク（"=", "-", "*" など）はチャンク全体をキーとする
  （GSM8K の演算語を語タイプとして保持するため）
- 部分文字列は削除しない（"cat" タイプの削除で "catalog" は壊れない。出現スパンは
  チャンク単位で記録するため構成上安全）
- 削除後は多重スペースを1つに詰める

注意: CoT:Jaccard@k 本体（analysis/analyzer.py の `_compute_jaccard_metrics_by_token`）
はサブワードトークン文字列の dedup で計算しており、厳密には「トークンタイプ」。
テキスト編集ベースの LOO ではサブワード削除が定義できないため、語（空白区切り）
タイプを採用し、R_C 側との比較には `_cot.pt` の `word_scores`（同じ空白区切りマージ）
を使う。この対応関係は open question として報告済み。

### log-prob スコアリング

- `sequence_logprob(model, tokenizer, context, target)`:
  full = context + target をトークナイズし、context のトークン列との最長共通接頭辞
  以降のトークン（= 答えトークン列。境界マージが起きた場合は保守的に共通接頭辞から）
  の log-prob 合計を 1 forward で計算（no_grad / float32 log_softmax）
- `batched_answer_logprobs(...)`: 変種ごとに context が異なるため右パディング +
  attention_mask でバッチ化。逐次版と数値一致することをユニットテストで保証
- `score_sample_loo(...)`: base（元 CoT）と全変種をスコアし
  importance = base_logprob - variant_logprob（正 = 削除で答えの確率が下がる = 重要）

### 出力スキーマ（R_C ランキング互換）

サンプルごとに `loo_word_scores`: `[{"word": str, "score": float}, ...]`（スコア降順、
results.json の `cot_top_k_words` / `_cot.pt` の `word_scores` と同じ word/score 形式）。
付帯情報（出現回数・変種 log-prob）は別キー `word_types` に分離し、ランキング本体の
スキーマを汚さない。これにより既存の `top_k_jaccard_by_token` がそのまま使え、
実験2（top-LOO削除腕）・実験3（precision@k）・実験7（LOO再集計）が同一形式で読める。

- R_C 側の CoT 語ランキングは `rc_word_ranking_from_cot_pt(data)` で
  `_cot.pt` の `word_scores` を cot_token_start/end 範囲で絞って抽出
- LOO vs R_C の Jaccard@k は `loo_jaccard_topk`（両側を `normalize_word` で
  端句読点正規化してから `top_k_jaccard_by_token`）

## スクリプト: `scripts/run_loo_scoring.py`

アーカイブの任意の run ディレクトリ（baseline / perturbed、results.json 必須）を
読み取り専用で入力し、`{output_dir}/{model_short}_{benchmark}_loo/` に
config.json / results.json / summary.json を出力する。clean 条件と摂動条件で
それぞれ実行して LOO ランキングを作れば、LOO 版 Jaccard@10 の再計算
（clean vs perturbed）は `loo_jaccard_topk` の再利用で可能（ρ(J|R) 再計算は
analysis/analyzer.py の既存経路に loo_word_scores を渡す。次 wave）。

- プロンプト再構築は rebuttal スクリプトと同一（`create_prompt_template(benchmark)`）
- モデルロードは `create_model_wrapper(wrap_for_lxt=False)`（forward のみで backward 不要）
- GPU は必ず `tmp/gpu-locks/run_with_gpu.sh` 経由

## Step 0 (master table) への接続

データアクセスはスクリプト内の `load_run_entries(run_dir)` に隔離。master table が
入手可能になったら、この1関数を「condition で master table を引く」実装に差し替える。
出力の `loo_word_scores` は master table 側スキーマ案の `loo_ranking`(json) 列に対応。

## 既知の注意点

- R_C 側 `word_scores` は `tokens_to_words` の仕様で改行をまたいで語が結合される
  ことがある（例: "dollars.\nThe"）。`normalize_word` は端句読点のみ剥がすため、
  この種の結合語は LOO 側と一致しない → LOO vs R_C Jaccard@10 はやや下方バイアス。
  LOO 版 Jaccard（clean-LOO vs perturbed-LOO）は両側とも同じ語タイプ定義なので影響なし。
- Gemma-3-4B-it × GSM8K clean の実測: 回答パターン検出 48/50、
  語タイプ数 mean≈37 (min 21 / max 64)、context 長 ≈ 900〜1200 トークン。
  数値の答えはトークナイザ上で桁ごとに分かれる（"18" → ["1","8"]）。
  `_target_start` の境界マージは実データでは未発生（フォールバックは保険）。

## 本番実行レシピ（次 wave / GPU 確保後）

M3×B2（Gemma-3-4B-it / Llama-3.2-3B-Instruct / Mistral-7B-Instruct-v0.3 ×
GSM8K / MMLU）、n≈200〜300/設定。clean と摂動（LXT-4 / Random-4）の両 run に適用:

```bash
# clean 側（例: Gemma × GSM8K）
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_loo_scoring.py \
  --run_dir $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --model google/gemma-3-4b-it --benchmark gsm8k --n 300 \
  --output_dir results/loo --run_label loo_clean

# 摂動側（LXT-4）
bash tmp/gpu-locks/run_with_gpu.sh uv run python scripts/run_loo_scoring.py \
  --run_dir $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --model google/gemma-3-4b-it --benchmark gsm8k --n 300 \
  --output_dir results/loo --run_label loo_lxt4
```

LOO 版 Jaccard@10 は `compute_loo_jaccard_pairs(clean_results, perturbed_results)`。
ρ(J|R) の再計算は analysis/analyzer.py の相関経路に loo_jaccard を変数として渡す
（次 wave。flip / ROUGE-L は Step 0 master table か既存 analysis_results.json から取得）。

## スモーク（2026-07-14, wave 1 完了条件）

Gemma-3-4B-it × GSM8K clean（アーカイブ baseline）n=16。
結果は `results/smoke/gemma-3-4b-it_gsm8k_loo_clean_smoke/` 参照
（results/ は gitignore 対象）。主要数値は最終報告と summary.json を参照。

## LOO 定義の変更（2026-07-14, wave 2: 案B を主定義に）

### 経緯とユーザー決定

wave 1 の LOO は「語タイプの全出現を一括削除して1変種」（案A）だった。
先行研究調査（調査メモ: scratchpad/loo_definition_survey.md、セッション限りの
一時ファイル。要旨は下記）の結果、1テキスト内の全出現一括削除には直接の先例が
確認できず、位置（出現）単位の削除が標準（Li et al. 2016; Jain & Wallace 2019;
ERASER; Atanasova et al. 2020）で、語タイプの重要度は「出現ごとの消去効果の平均」
として定義する先例が Li, Monroe & Jurafsky (2016, arXiv:1612.08220) にある。

**ユーザー決定（2026-07-14）**: 案B（出現ごと削除 → タイプへ集約）を主定義、
現行の案A（全出現一括削除）は type-level erasure（冗長な再言及を遮断する
反実仮想）として感度分析に併記する。

### 実装（`deletion_mode` 引数）

- `score_sample_loo(..., deletion_mode="occurrence")`（デフォルト）:
  1出現 = 1変種。タイプ重要度 = 出現スコアの**平均**（`aggregation: "mean"`）。
  max は `word_types[].score_max` に副次保存（集約関数の感度分析用）。
- `deletion_mode="type"`: 従来の全出現一括削除（`aggregation: "whole_type"`）。
- ランキング本体 `loo_word_scores` は R_C 互換の `{word, score}` のまま。
  `n_occurrences` / `n_variants` / `deletion_mode` / `aggregation` /
  `occurrence_scores` / `variant_logprobs` はメタデータ側に記録。
- CLI: `run_loo_scoring.py --deletion-mode {occurrence,type}`（デフォルト occurrence）。

### R_C 側改行またぎ結合語の修正（Jaccard 計算パス）

R_C の `word_scores` は改行をまたいで語が結合される（例 "dollars.\nThe"）。
`expand_multiword_entries` を `loo_jaccard_topk` に組み込み、空白を内包する
エントリを構成語に分解（各構成語は親スコアを引き継ぐ）してから比較する。

修正前後（type モード smoke n=16、Gemma-3-4B-it × GSM8K clean）:
- mean LOO vs R_C Jaccard@10: **0.3483 → 0.4551（+0.107）**、16サンプル中12で変化。
- R_C 上位ランキング中の結合語エントリは計113件。偽不一致の除去として有意な差。

### 検証（2026-07-14, GPU 3/4, n=16, 両モード）

`results/smoke/gemma-3-4b-it_gsm8k_loo_clean_{occ,type}_smoke/`:
- (a) 案B の変種数 = 出現数合計: 16/16 サンプルで `n_variants == Σ n_occurrences` を確認。
- (b) 案B vs 案A の Top-10 Jaccard: mean 0.755 / median 0.818（min 0.538, max 1.000）
  → 定義変更でランキングは概ね保たれるが同一ではない（感度分析の価値あり）。
- (c) 案B vs R_C Jaccard@10（改行修正後）: mean **0.4599** / median 0.4286。
  参考: 案A vs R_C（改行修正後）は mean 0.4492 / median 0.4286（案B がわずかに高い）。
- 案B の mean_top1_loo_score は 0.17（案A は 1.14）: 平均集約は反復言及される
  数値のスコアを希釈する（調査メモの「冗長性バイアス」がそのまま観測される）。
  論文では案A 併記でこの差自体を分析対象にできる。

## 本番実行 (2026-07-18, 実験6-(iv) LOO ランキング本番)

スコープ: M3 (Gemma-3-4B-it / Llama-3.2-3B-Instruct / Mistral-7B-Instruct-v0.3)
× B2 (GSM8K / MMLU) × {clean, LXT-4 (k4_importance)}。
主定義 = occurrence (案B)、感度分析 = type (案A、Gemma-3-4B × B2 のみ)。

- サンプル選定: `run_loo_scoring.py --seed 42 --n 300 --clean_run_dir <baseline>`
  で clean 正解サンプルから決定論的に 300 件 (`select_sample_ids`)。
  摂動 run も同じ選定 (baseline を `--clean_run_dir` に指定) で同一 id 集合。
- Mistral R_C: アーカイブ `_cot.pt` の word_scores は全文1語に結合する既知バグ
  (4条件×8サンプルの実査で 32/32 degenerate)。exp/02-target-deletion コミット
  fef3958 の token_scores 貪欲整列ローダーを移植し、`load_rc_ranking` が
  `full_text = prompt + generated_text` を配線して自動再構築。
  再構築発動数は summary.json の `stats.rc_degenerate` に記録。
- キュー: `scripts/run_loo_production_queue.sh` (16シャード、summary.json で冪等、
  mkdir claim、進捗 JSON は results/loo/queue/progress/)。GPU は
  tmp/gpu-locks/run_with_gpu.sh 経由 (GPU 3-6、他系統とロック共有)。
- 集計: `scripts/analyze_loo_rankings.py` — LOO 版 CoT:Jaccard@10
  (clean-LOO vs LXT4-LOO、expand_multiword_entries 適用パス) と
  ρ(J_LOO@10|R) (R = アーカイブ full_results.json の cot_rouge_l.f1、
  参照値 ρ(J_RC@10|R) を同一結合行 + アーカイブ全数で併記)。
  出力: results/loo/aggregate_loo_{occ,type}.json。

### コスト再見積り（本番 M3×B2, n≈200〜300）

事前想定「出現ごと削除で forward 数が最大2〜3割増」は**実測で否定**:
- 変種数: mean 36.3（タイプ数）→ mean 87.9（出現数合計）= **x2.42**。
  GSM8K CoT は機能語・数値の反復が多く、想定よりはるかに多出現。
- 実行時間（n=16, Gemma-3-4B-it, batch_size=8）: type 31.8s → occurrence 75.9s
  = x2.39（変種数比とほぼ一致。約 4.7s/sample）。
- 本番見積り（occurrence モード）: Gemma-4B 級で n=300 ≈ 24分/run。
  M3×B2 × 3条件（clean / LXT-4 / Random-4）= 18 run。Mistral-7B は ~2x として
  合計 **約 8〜10 GPU 時間**（2 GPU 並行で実時間 4〜5 時間）。
  案A 感度分析を全条件で併走すると +40%（案A は x1 なので安価）。
  シャード分割（--n/--sample_offset で 100 サンプル単位）で実行する。
