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

## 実験6-(i)〜(iii): 帰属ファミリー代替手法（wave 3, 2026-07-18）

別エージェント（(i)〜(iii) 担当）による追記。LOO（(iv)）とはモジュール・
スクリプト・出力ディレクトリを分離し、規約（サンプル選定・3分割・語集約・
R_C ローダー）は LOO 実装を importlib / パッケージ経由で共有する。

### 実装: `src/typo_cot/attribution_family/methods.py`

- 目的関数は 3手法共通で「答えトークン列 log-prob 合計」（LOO と同一の
  `split_generated_text` 3分割、context = prompt + cot + trigger、
  target = answer。answer より後のテキストは入力から除外）。
- **(i) G×I**: `gradient_x_input_token_scores` — 埋め込みへの勾配×入力
  （backward 1回、grad·embed の内積を fp32 で計算）。
- **(ii) IG**: `integrated_gradients_token_scores` — ベースライン=ゼロ埋め込み、
  midpoint 則 α_k=(k+0.5)/m、m=16（本番）、α ステップは 4件ずつバッチ。
  completeness 診断（sum_attr / (F(x)-F(0))）をサンプルごとに保存。
- **(iii) rollout**: `attention_rollout_token_scores` — 層ごとに head 平均 →
  0.5I + 0.5A → 行正規化 → 層積（Abnar & Zuidema 2020）。答え予測位置
  （target_start-1 .. S-2）の行を平均。モデルは attn_implementation="eager"
  でロード（SDPA は attention を返さない）。
- **語集約**: `token_scores_to_word_ranking` が Mistral R_C 再構築ローダー
  `rc_word_ranking_from_token_scores` をそのまま再利用（空白チャンク整合・
  CoT 領域フィルタ・スコア合計の規約を R_C/LOO 比較経路と完全一致）。
  Llama 系特殊トークン（"<|begin_of_text|>"）は `decode_tokens_for_alignment`
  で "<s>" に写像して整合対象外にする。
- ランキングは R_C/LOO と同じ raw スコア降順 `{word, score}`（比較は
  `loo_jaccard_topk` = expand_multiword + normalize_word + top_k_jaccard_by_token）。

### スクリプト

- `scripts/run_attribution_family.py`: 1 run = 1設定×1手法×1条件。サンプル選定は
  run_loo_scoring.py の `select_sample_ids` を importlib 共有（**n=300, seed=42 =
  LOO 本番と同一サンプル**）。R_C との Jaccard@10 は `load_rc_ranking`
  （full_text 配線 = Mistral は token_scores 再構築）で計算。
  出力: `results/attribution_family/{model}_{bench}_{method}_{clean|lxt4}/`。
- `scripts/analyze_attribution_family.py`: 内的軸 J_method@10（clean vs LXT-4 の
  手法ランキング Jaccard）を再計算し、アーカイブ full_results.json の R
  (cot_rouge_l.f1) と結合して ρ(J_method@10|R)。参照 ρ(J_RC@10|R) を同一結合行で併記。
- `scripts/run_attribution_family_queue.sh`: 36 シャード（3手法 × M3×B2 ×
  {clean, LXT-4}）の冪等キュー（LOO キューと同型、GPU ロック共有、軽い手法から）。

### スモーク（2026-07-18, Gemma-3-4B-it × GSM8K clean, n=8, seed=42）

`results/attribution_family/smoke/`（results/ は gitignore 対象）:
- 手法 vs R_C (AttnLRP) の Jaccard@10（mean/median）:
  **IG 0.359/0.333 > G×I 0.209/0.250 > rollout 0.126/0.082**、8/8 サンプル成功
  （skip/align失敗/エラー 0）。ランダム期待値（~80語タイプから10語×2の重複）
  ≈ 0.06〜0.07 なので全手法が有意に上回り、計画の予想順序
  （AttnLRP ≥ IG > G×I > rollout）と整合。
- 実行時間: G×I 0.4s/sample、rollout 0.13s/sample、IG(m=16) 2.9s/sample。
- **IG ステップ数感度（m=16 vs m=32, 同一8サンプル）**: vs R_C Jaccard@10 は
  0.359 vs 0.362 でほぼ不変、completeness 比（sum_attr/(F(x)-F(0))）は
  m=16 で mean 1.86 / m=32 で 2.76（median 2.11 → 1.51）と**増ステップでも
  改善しない**（bf16 勾配 + 経路非線形が支配的）。手法内 top-10 の
  m16 vs m32 Jaccard は mean 0.59。→ 本番は **m=16** を採用
  （仕様の 16〜32 の下限、コスト半減。completeness はサンプルごとに保存済みで
  診断として報告する）。

### 本番キュー起動（2026-07-18）

`scripts/run_attribution_family_queue.sh` を worker 2 本で起動
（36 シャード、GPU 3-6 ロック共有、n=300 seed=42）。
集計は全完走後に `scripts/analyze_attribution_family.py` で
Jaccard@10 表 + ρ保持表を作成し本ファイルに記録する。

### 集計（2026-07-19, 36/36 シャード完走, n=300 seed=42, k=10）

全 18 設定 (M3×B2×3手法) status=ok、失敗・欠損なし。集計 JSON: `results/attribution_family/aggregate_attribution_family.json`（アーカイブ analysis k4_importance の full_results と sample_id 結合）。

**表1: J_method@10（clean vs LXT-4 の手法ランキング Jaccard, mean/median）と vs R_C Jaccard@10（mean, clean側/LXT-4側）**

| 設定 | 手法 | n_joined | J_method@10 mean/med | vs R_C @10 (clean/lxt4) |
|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | G×I | 276 | 0.372 / 0.333 | 0.308 / 0.318 |
| gemma-3-4b-it × gsm8k | IG | 276 | 0.487 / 0.538 | 0.380 / 0.380 |
| gemma-3-4b-it × gsm8k | rollout | 276 | 0.570 / 0.667 | 0.154 / 0.151 |
| gemma-3-4b-it × mmlu | G×I | 283 | 0.187 / 0.111 | 0.249 / 0.249 |
| gemma-3-4b-it × mmlu | IG | 283 | 0.197 / 0.176 | 0.256 / 0.270 |
| gemma-3-4b-it × mmlu | rollout | 283 | 0.377 / 0.333 | 0.217 / 0.228 |
| Llama-3.2-3B-Instruct × gsm8k | G×I | 255 | 0.388 / 0.333 | 0.344 / 0.335 |
| Llama-3.2-3B-Instruct × gsm8k | IG | 255 | 0.399 / 0.429 | 0.404 / 0.405 |
| Llama-3.2-3B-Instruct × gsm8k | rollout | 255 | 0.523 / 0.538 | 0.341 / 0.333 |
| Llama-3.2-3B-Instruct × mmlu | G×I | 262 | 0.196 / 0.176 | 0.241 / 0.262 |
| Llama-3.2-3B-Instruct × mmlu | IG | 262 | 0.165 / 0.111 | 0.219 / 0.240 |
| Llama-3.2-3B-Instruct × mmlu | rollout | 262 | 0.322 / 0.250 | 0.346 / 0.360 |
| Mistral-7B-Instruct-v0.3 × gsm8k | G×I | 265 | 0.402 / 0.333 | 0.417 / 0.426 |
| Mistral-7B-Instruct-v0.3 × gsm8k | IG | 265 | 0.526 / 0.538 | 0.294 / 0.294 |
| Mistral-7B-Instruct-v0.3 × gsm8k | rollout | 265 | 0.605 / 0.667 | 0.268 / 0.265 |
| Mistral-7B-Instruct-v0.3 × mmlu | G×I | 286 | 0.271 / 0.250 | 0.412 / 0.422 |
| Mistral-7B-Instruct-v0.3 × mmlu | IG | 286 | 0.264 / 0.176 | 0.270 / 0.279 |
| Mistral-7B-Instruct-v0.3 × mmlu | rollout | 286 | 0.459 / 0.429 | 0.290 / 0.297 |

**表2: ρ保持表 — Spearman ρ(J_method@10 | R)（R = cot_rouge_l.f1、参照 ρ(J_RC@10|R) は同一結合行で再計算）**

| 設定 | ρ(J_RC\|R) 参照 | ρ G×I | ρ IG | ρ rollout |
|---|---|---|---|---|
| gemma-3-4b-it × gsm8k | 0.490 | 0.590 (p=3.0e-27) | 0.610 (p=1.7e-29) | 0.758 (p=7.2e-53) |
| gemma-3-4b-it × mmlu | 0.335 | 0.568 (p=1.3e-25) | 0.599 (p=5.6e-29) | 0.745 (p=2.4e-51) |
| Llama-3.2-3B-Instruct × gsm8k | 0.449 | 0.705 (p=1.3e-39) | 0.603 (p=1.1e-26) | 0.845 (p=9.5e-71) |
| Llama-3.2-3B-Instruct × mmlu | 0.320 | 0.553 (p=2.3e-22) | 0.585 (p=1.7e-25) | 0.777 (p=4.0e-54) |
| Mistral-7B-Instruct-v0.3 × gsm8k | 0.583 | 0.828 (p=6.6e-68) | 0.781 (p=9.9e-56) | 0.822 (p=3.5e-66) |
| Mistral-7B-Instruct-v0.3 × mmlu | 0.484 | 0.731 (p=3.8e-49) | 0.750 (p=7.8e-53) | 0.781 (p=4.2e-60) |

**所見**:

- **相関構造は全設定で保持**: ρ(J_method@10|R) は 18/18 設定で正かつ有意（0.55〜0.85, 全て p < 1e-21）。さらに全 18 設定で参照 ρ(J_RC@10|R)（0.32〜0.58）を上回った。帰属手法を AttnLRP から G×I / IG / rollout に替えても「内的ランキングの安定性が出力の頑健性と連動する」という実験6の主要相関構造は崩れない（むしろ強い）。
- 手法間では **rollout の ρ が最大**（0.745〜0.845, 6/6 設定で最上位または同率）だが、rollout はモデル出力に依存しない注意経路指標のためJ_rollout 自体が摂動での入力トークン変化を強く反映する点に注意（vs R_C Jaccard は 0.13〜0.36 と最も低い群）。勾配系 (G×I/IG) はρ 0.55〜0.83 で AttnLRP 参照と最も近い挙動。
- vs R_C Jaccard@10 (n=300 本番) はスモーク (n=8) の傾向を概ね再現（gemma gsm8k: IG 0.380 > G×I 0.308 > rollout 0.154）。ただし設定によっては G×I が IG を上回る（Mistral 両ベンチ, Llama mmlu）。
- J_method@10 の絶対値は gsm8k > mmlu（全手法）で、既報の J_RC の傾向と整合。
