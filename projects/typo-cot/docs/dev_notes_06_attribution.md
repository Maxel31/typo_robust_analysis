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

## スモーク（2026-07-14, wave 1 完了条件）

Gemma-3-4B-it × GSM8K clean（アーカイブ baseline）n=16。
結果は `results/smoke/gemma-3-4b-it_gsm8k_loo/` 参照（results/ は gitignore 対象）。
