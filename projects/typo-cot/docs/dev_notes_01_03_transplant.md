# 実験1 (CoT transplant 2×2) + 実験3 (forced-decoding divergence) 開発メモ

ブランチ: `exp/01-03-transplant` / 担当: 実験1・実験3 専任エージェント

## モジュール構成 (src/typo_cot/intervention/)

| モジュール | 役割 |
|---|---|
| `records.py` | `PairRecord` — clean×typo のサンプル対。Step 0 master table に差し替えやすいフィールド構成 |
| `archive_loader.py` | アーカイブ baseline/perturbed `results.json` を sample_id で結合 → `PairRecord`。**アーカイブ依存はここだけ** |
| `cell_builder.py` | 答え句 (`The answer is`) 直前での CoT 切断 + 4セル (A/B/C/D) の teacher-forcing 入力構築 |
| `runner.py` | `run_cells(pairs, generate_fn, ...)` — バッチ生成・抽出・TE 照合。モデルは generate_fn として注入 (ユニットテストはモック) |
| `analysis.py` | flip 表 (TE/DE/IE)・見出し復帰率・IE\|CoT変化・bootstrap CI・GLMM (BinomialBayesMixedGLM, VB) |
| `divergence.py` | 位置別 KL/log-prob/rank のチャンク計算・オフセット補正付き位置対応・発散オンセット・precision@k + シャッフル帰無 |

CLI: `scripts/exp01_03/run_transplant.py` (実験3 は `--dump-divergence` フラグ)。

## セル定義

- A = (clean 質問, clean CoT) … 基準
- B = (typo 質問, typo CoT) … 総効果 TE (アーカイブ再現の検証にも使用)
- C = (typo 質問, clean CoT) … 直接効果 DE
- D = (clean 質問, typo CoT) … 間接効果 IE

## 設計判断 (技術的なもの)

1. **teacher-forcing は単純連結**: アーカイブの生成 (`scripts/run_inference.py`) は
   chat template を使わないプレーンテキスト few-shot completion
   (`system_prompt + "\n\n" + user_prompt`) なので、切断済み CoT を
   プロンプト末尾に連結するのが忠実な prefill 実装。将来 DeepSeek-R1-Distill 等の
   新規生成でチャットテンプレートを使う場合は `continue_final_message` 相当が必要。
2. **除外フラグ** (`cell_builder`): `no_trigger_*` / `multi_trigger_*` /
   `early_trigger_*` (先頭25%以内) / `residual_fragment_*`
   (prefix に `Answer:` 等の変種が残留)。主分析は全フラグなしに限定し、
   `flip_rate_sensitivity` に除外込みの値を併記。
3. **CUDA_VISIBLE_DEVICES を触らない**: `models/wrapper.setup_device` は
   CUDA_VISIBLE_DEVICES を上書きするため使用しない。`ModelWrapper` を直接
   構築し、run_with_gpu.sh が設定したデバイスをそのまま使う。
4. **divergence の位置対応**: prompt 単体と full input を別々にトークナイズし、
   suffix のトークン ID 列が clean/typo run で完全一致した場合のみ計算
   (`token_alignment_mismatch` でフラグして除外、件数は summary に記録)。
5. **GLMM**: A セルは構造的に flip=0 で切片が縮退するが、
   BinomialBayesMixedGLM のベイズ事前分布が正則化するため有限に推定される。
   全条件同時推定 (交互作用込み) という仕様通りの形。
6. **cot_changed**: 空白正規化した切断後 prefix の不一致で判定
   (ROUGE-L<1 の代理。master table 到来後に CoT:ROUGE-L 列へ差し替え可)。

## アーカイブ CPU 検証 (2026-07-14)

gemma-3-4b-it × gsm8k × k4_importance: 1319 ペア結合、除外 132 件
(no_trigger_typo 69 / no_trigger_clean 35 / multi_trigger 46 / residual 12 / early 1)。
mmlu 側も 2850 ペア結合、選択肢インライン再構成が clean 側テンプレート整形と一致。

## 本番実行の想定コマンド

```bash
bash /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/tmp/gpu-locks/run_with_gpu.sh \
  uv run python scripts/exp01_03/run_transplant.py \
  --model google/gemma-3-4b-it --benchmark gsm8k \
  --baseline-dir  $ARCHIVE/outputs/baseline/gemma-3-4b-it_gsm8k \
  --perturbed-dir $ARCHIVE/outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance \
  --dump-divergence --output-dir results/exp01_03/gemma-3-4b-it_gsm8k_k4_importance
```

Random-4 は `--perturbed-dir ..._k4_random`。全 25 設定は docs/v1_run_manifest.md 参照。

## 事前登録: 分岐別の結論文 (実行前に固定, 2026-07-14)

- **パターンX (CoT 媒介優位)**: IE が TE の大部分 (目安 7 割以上) を占め、
  DE 条件で clean CoT を強制すると大半の flip が元の答えに復帰する場合 —
  「Transplanting the clean CoT under the perturbed question restores the
  original answer in the majority of flipped cases (headline restore rate),
  while transplanting the perturbed CoT under the clean question reproduces
  most flips. Typo-induced errors are therefore causally mediated by the
  CoT text itself, not merely correlated with it: the pathway
  typo → CoT change → answer change carries the effect.」
- **パターンY (直接経路優位)**: DE が TE の大きな割合 (目安 4 割以上) を占め、
  clean CoT を強制しても flip が残る場合 —
  「Even when the clean CoT is forced verbatim, a substantial fraction of
  flips persists (DE), revealing a direct pathway from the perturbed input
  to the answer that bypasses the generated reasoning text. This motivates
  the internal-state analysis of Experiment 8, and implies that
  CoT-level defenses alone cannot fully recover accuracy.」

どちらの結果でも執筆可能。GLMM の交互作用項が大きい場合は
「質問 typo と CoT typo の効果は加法的でない」ことを明記する。

## GPU スモーク結果 (2026-07-14, GPU 3, n=32)

前回 nohup キューが GPU 3 解放後に自走し 18:17–18:18 に完了
(logs/smoke_gsm8k_lxt4.log, results/smoke/gemma-3-4b-it_gsm8k_k4_importance/)。
実行時間: モデルロード込み約40秒 (4セル生成 ~14s + divergence 32件 ~5s)。

- **(a) TE 再現一致率**: 非除外 29/29 = **100%** (PASS)。全32件では 31/32=96.875% —
  唯一の不一致 gsm8k_00007 は multi_trigger_typo/no_trigger_clean で除外対象
  (最初の trigger で切断→再生成 "20.13671" vs アーカイブは丸め後 "20.14" を再掲する
  2重 answer 文。除外フラグの設計意図どおりの検出)。
- **(b) 4セル**: 全32件で A/B/C/D 同一骨格 (プレーンテキスト連結)・greedy
  (do_sample=False, temperature=0.0)。flip 表出力: n_included=22,
  TE=2, DE=0, IE=2, headline_restore_rate=1.0 (PASS)。
- **(c) divergence**: 29/29 計算成功 (alignment 失敗 0)。位置別 KL/log-prob/rank
  出力あり。KL は上位10%位置に平均 93.1% (中央値 94.6%, 最小 69.0%) 集中 (PASS)。
- 除外: 3件 (no_trigger_clean 2, multi_trigger_typo 1, multi_trigger_clean 1;
  gsm8k_00007 は重複フラグ)。A誤答除外 7件 → 分析対象 22件。

注意: summary.json の `te_match_rate` は除外込みの全件比。本実行 (n≥1000) では
除外サンプル分だけ 99% を割り得るため、非除外限定の一致率も併記するか
基準の母集団を明文化すること。

## 本番キュー (2026-07-15)

- **スコープ**: M5 (gemma-3-1b/4b, Llama-3.2-1B/3B, Mistral-7B-v0.3) × B5
  (gsm8k/mmlu/mmlu_pro/arc/commonsense_qa) × {k4_importance, k4_random} = 50 設定。
  アーカイブに全 50 の baseline/perturbed ログを確認済み (計 79,618 結合ペア)。
  mmlu (2850 ペア) のみ `--start`/`--n` で 2 分割 → **総シャード数 60**。
  n は全量 (`--clean-correct-only` なし。clean 正解条件付けは分析側 flip_table が適用)。
  batch_size=8 / max_new_tokens=16 (スモーク実測準拠の既定値)。
- **TE 再現一致率の判定母集団は非除外サンプル限定** (multi_trigger 等の構造的除外は
  アーカイブと一致し得ないため)。全件比は参考値として verify_shard.py が併記。
- **キュー構成** (`scripts/exp01_03/`):
  - `make_shards.py` → `shards_all.tsv` (60 行)。Qwen2.5-7B / R1蒸留 / MATH-500 は
    基盤生成完了後に **shards_active.tsv へ行を追記するだけ**でキューが拾う
    (worker はループ毎に一覧を再読込)。
  - `queue_worker.sh`: 冪等スキップ (summary.json 存在)・mkdir 原子 claim・
    stale claim 自動回収 (pid 生存確認)・進捗 JSON
    (`results/exp01_03/queue/progress_<id>.json`)。GPU は run_with_gpu.sh 経由のみ。
    rc=86 (PAUSED) はワーカー終了、rc=124 (ロック待ちタイムアウト) は failed に
    せず再試行。失敗は `queue/failed/<name>` (削除で再試行)。
  - 起動: `cd <proj> && WORKER_ID=wN setsid nohup bash scripts/exp01_03/queue_worker.sh
    < /dev/null >> logs/exp01_03/worker_wN.log 2>&1 &`
  - 監視: `bash scripts/exp01_03/queue_status.sh` / 停止: `touch results/exp01_03/queue/STOP`
  - 検証: `uv run python scripts/exp01_03/verify_shard.py results/exp01_03/<shard>`
- **段階投入**: 検証シャード 2 本 (gemma-3-4b gsm8k LXT-4/Random-4 全量) を先行実行し、
  TE 再現率 (非除外)・flip 表・KL 集中がスモーク水準であることを確認してから
  残り 58 シャードを shards_active.tsv に追記する。

## 検証シャード結果 (2026-07-15, 本実行 gate PASS)

gemma-3-4b-it × gsm8k 全量 (n=1319) × 両摂動条件。GPU 6 で各 ~11〜14 分
(≈0.6 s/sample、スモーク実測どおり)。

| 指標 | LXT-4 | Random-4 | スモーク水準 |
|---|---|---|---|
| TE 再現一致率 (非除外) | 99.66% (1183/1187) | 99.83% (1199/1201) | 100% (29/29) |
| TE 再現一致率 (全件, 参考) | 96.66% | 97.35% | 96.88% |
| flip TE / DE / IE | 9.79% / 1.34% / 9.60% | 5.71% / 1.33% / 5.81% | (n=22: TE2/DE0/IE2) |
| headline restore rate | 93.1% | 90.0% | 1.0 |
| KL 上位10%集中 (平均/中央値) | 93.6% / 96.0% | 88.5% / 90.3% | 93.1% / 94.6% |
| divergence alignment | 1187/1187 | 1201/1201 | 29/29 |

- **両条件で IE≈TE・DE 小 (パターンX方向)、効果量は LXT > Random** — 修正Aの
  報告論理 (分解構造の両条件一致) と整合する第一データ点。
- 非除外 TE 不一致の内訳 (LXT側 4件): max_new_tokens=16 到達による答え句切断 2件
  (gsm8k_00240, 01218 — 答え句の前に一文を再掲するケース)、再トークン化境界での
  greedy 分岐 2件 (gsm8k_00844, 01265 — "Alternatively," 継続)。~0.3% の
  teacher-forcing 固有アーティファクトで、全セル対称 (max長は計画の ≦16 凍結値を維持)。

## 本番実行完了 (2026-07-15 16:51, M5×B5×2条件 全60シャード)

09:26 開始 → 16:51 全完了 (ワーカー2〜3並列、失敗 0・再実行 0)。
`results/exp01_03/<shard>/{summary.json,outcomes.json,divergence/}`。

- 生成ペア 79,618 (4セル×答えスパン)、主分析対象 (A正解×非除外) 39,275
- **TE 再現一致率 (非除外, pooled): 98.23%** (67,251/68,462)。モデル別では
  gemma-3-4b ≈99.7% / Llama-3B ≈99% / Mistral-7B ≈98.5% / Llama-1B ≈97.5% /
  gemma-3-1b ≈95% (最低 mmlu_pro 92.2%)。小型モデルほど再トークン化境界の
  greedy 分岐と max16 切断の影響が大きい (全セル対称なので分解には内的整合)
- **divergence 68,462 プロファイル、alignment 失敗 0**
- pooled flip: LXT-4 TE=23.9% / DE=7.4% / IE=19.7% (IE/TE=0.83, DE/TE=0.31)、
  Random-4 TE=17.0% / DE=6.1% / IE=13.6% (IE/TE=0.80, DE/TE=0.36)
  → **IE優位の分解構造が両摂動条件で一致、効果量は LXT > Random** (修正Aの見出し論理成立)
- ベンチ形式差: gsm8k (自由記述) は DE ≈1〜3% で restore 90〜99% (パターンX鮮明)、
  多肢選択は DE がやや大 (計画の予想どおり)
- 集計テーブルは `scripts/exp01_03/verify_shard.py` を全シャードに回して再現可能

## R1蒸留系 (DeepSeek-R1-Distill-Qwen-7B) 実験1+3 — 最終実験セル (2026-07-19)

実験10-③「R1蒸留系は実験1・3のみ参加」。生成ログは exp-10-scope worktree
(`outputs/{baseline,perturbed}/DeepSeek-R1-Distill-Qwen-7B_{gsm8k,math,mmlu}_*`)
に完備。R1 の生成規約 (チャットテンプレート・<think>分離・答え抽出チェーン) は
`src/typo_cot/models/reasoning.py` を exp/10-scope から移植して参照実装とした。

### 設計判断 (R1 差分の <think> 構造への自然拡張)

1. **移植点 (CoT 切断点)**: 「<think> 開始 〜 </think> 直後の答え文の答え宣言直前」。
   forced CoT = <think>本文 + </think> + (答え文の最初の答え宣言直前まで)。基底
   モデルの「"The answer is" 直前で切断→短スパン再生成」を <think> 構造へ写像した
   もの。宣言直前で切るので teacher-forcing 再生成は宣言+答えの短スパンのみ
   (max_new_tokens: gsm8k/mmlu=64, math=128)。実装: `intervention/reasoning_cells.py`
   の `truncate_reasoning_cot` (cell_builder に `truncator=` 注入)。
2. **答えトリガー**: 基底の "The answer is" のみでは MMLU 包含率が 47% まで落ちる
   (R1 は Answer:/ANSWER:/The correct option is/\boxed{} も使う)。`</think>` 後の
   答え文に対し `REASONING_ANSWER_TRIGGER` (これらを網羅) を適用。<think> 本文内の
   推論句 ("so the answer is X") はトリガーにしない (本文は常に丸ごと移植)。
3. **除外規則の自然拡張** (凍結規則の意図=「答えに到達し、forced prefix に答えが
   漏れない」を <think> 構造で満たす):
   - `no_trigger`: </think> 未生成 (CoT 途中切断) or 答え文に宣言なし。
   - `multi_trigger`: R1 は同一答えを反復宣言する癖が強い (Qwen 先例と同様)。
     最初の宣言直前で切れば再生成は最初の宣言を復元するので反復は曖昧でない。
     `dedup_same_answer_triggers=True` (R1 は既定 ON) で **全宣言が同一答えなら
     除外しない**。答えが途中で変わる真の曖昧さのみ除外。除外込みは
     `flip_rate_sensitivity` に併記。
   - `early_trigger`: 常に False (<think> 本文は常に丸ごと移植)。
   - `residual`: 答え文の宣言直前プロセにのみ適用 (<think> 本文は対象外)。
   実測 dedup 込み包含率: gsm8k≈87-92% / mmlu≈73-75% / math≈66-71%
   (multi_diff=真の曖昧さのみ除外)。dedup 無効なら mmlu は 23% まで落ちる。
4. **プロンプト**: チャットテンプレート (`build_full_prompt`) を `prompt_builder=`
   注入。tokenize は `add_special_tokens=False` (テンプレートが `<｜begin▁of▁sentence｜>`
   を内包)。生成は専用 `build_reasoning_generate_fn` (新規トークン ID のみ
   skip_special_tokens=True でデコード。ModelWrapper.generate_batch の文字列
   スライスは特殊トークン skip で位置ずれするため不使用)。
5. **抽出**: 再生成スパンは `reasoning.extract_reasoning_answer` チェーン
   (`extract_fn=` 注入。$ 記号・boxed フォールバック込み) で抽出。
6. **divergence (実験3)**: <think> 本文全体を強制 CoT として位置別 KL/rank を計算。
   tokenize は add_special_tokens=False。precision@k は R1 では R_C 未計算
   (`rc_computed=False`, cot_top_k_words=[]) のため N/A (KL/onset プロファイルのみ)。
   モデル依存を注入で隔離 (build_cell_inputs/run_cells に prompt_builder/truncator/
   extract_fn パラメータ追加。既定 None で基底5モデルの挙動は完全不変=既存テスト
   44件 GREEN 維持)。
7. **reasoning モード判定**: `is_reasoning_model` がモデル名 (r1-distill/deepseek-r1)
   から自動判定。`--reasoning` フラグでも明示可。キュー行は基底と同形式で追加のみ。

### スモーク結果 (2026-07-19, gate PASS)

n=16 × 3ベンチ (LXT-4)、GPU 3/4/5:

| ベンチ | TE 再現 (非除外) | divergence alignment | 除外 |
|---|---|---|---|
| gsm8k | 10/10 = 100% | 10/10 (失敗0) | 6 (multi_diff 2, no_trigger 4) |
| mmlu  | 12/12 = 100% | 12/12 (失敗0) | 4 (multi_diff 1, no_trigger 3) |
| math  | 8/9 = 89%     | 9/9 (失敗0)   | 7 (multi_diff 5, no_trigger 5) |

- 4セル整合: 例 gsm8k_00002 A=195000/B=70000/C=195000(DE復帰)/D=70000(IE flip)
  — CoT 媒介 (パターンX) の signature。boxed/選択肢/数値の抽出すべて正常。
- reasoning=True・dedup=True・max_new_tokens 64/128 が自動解決されることを確認。

### 本番キュー (2026-07-19 投入)

6設定 = {gsm8k, math, mmlu} × {LXT-4, Random-4}。全量 (gsm8k 1319 / math 500 /
mmlu 5700)。mmlu は 1425×4 シャード → 総 12 シャード。`shards_active.tsv` に追記
(基底と同形式・追加のみ)。GPU 3/4/5/6, worker 4本, run_with_gpu.sh flock 経由。

## 残タスク / 注意 (基底5モデル分)

- precision@10 の語タイプ対応はトークン近似 (KL 側) vs 単語 (R_C 側)。
  本実行前に offset_mapping ベースの単語集約に精緻化する余地あり。
- LOO ランキング比較 (修正B) は未実装 (実験8 側と共有予定のため保留)。
