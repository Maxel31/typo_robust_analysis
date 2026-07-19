# 実験15 (本命) 開発メモ — 早期層 patch → CoT 自由生成による S1→S2 因果閉鎖

ブランチ: `exp/15-patch-freegen` / 担当: 実験15 専任エージェント
上流依存: `exp/08-patching` の `intervention/` (PatchInjector 等 hook 実装) と
`exp/04-fixed-target` の `setup_device` 修正済み `models/wrapper.py`。
`git checkout exp/08-patching -- <path>` で取り込み (develop 上で既存介入テスト
70 件が緑であることを確認済み)。

## 仮説 H15

実験8 は「早期層を直せば**答え**が戻る」(clean CoT 強制下) を示した。本実験は
「早期層を直せば **CoT の逸脱そのもの**が消える」を示す。これが通れば
損傷 (S1) → 分岐 (S2) → 搬送 (S3) の 3 リンクすべてが介入で支持され、統一主張
(ERDC 連鎖) の背骨が完成する。実験8 の制約 — GSM8K では clean CoT を
teacher-forcing するため「答え分岐の主経路である CoT 自体の逸脱」が遮断され、
flip がほぼ観測不能 (分岐 0〜3%) — を、本実験の **CoT 自由生成** が解放する。

## モジュール構成

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/free_generation.py` | 生成保持型 hook (`generate_ids` / `generate_ids_patched`)・発散オンセット `divergence_index`・摂動語スパン整列 `align_span_positions` / `locate_word_char_spans`・CoT ROUGE-L ラッパ `cot_rouge_l` (既存 `analysis.metrics.rouge_l_score` = 文字単位 LCS、論文 Table 6 と同一定義を再利用)。GPU 非依存 |
| `scripts/exp15/run_free_generation.py` | CLI。flip ペア選定 (LXT-4/Random-4 半々)・{窓×方向} セル・sham(no-op)検証・シャード・冪等 (ペア単位 JSON) |
| `scripts/exp15/aggregate.py` | 設定別集計・事前登録 H15 判定 |
| `scripts/exp15/prod/run_queue.sh` | 本番キュー (M3×B2、`run_with_gpu.sh` 経由、冪等、`SMOKE_PAUSED` 尊重) |

## 手法 (生成保持 hook と KV キャッシュ整合)

exp8 の `PatchInjector` は「答えトークンでの測定」用に、recipient run の
**prefill 時**に患部 (摂動語スパン) 位置へ donor 活性を注入し、generate の
decode ステップ (系列長 1) では自動的に無効化する設計だった。

本実験はこの hook を **`model.generate` 全体を囲うコンテキストマネージャ**として
そのまま用いる。患部 (摂動語スパン) は **prompt 内**にあるため、prefill での
1 回注入で足り、以降の decode は追加注入不要で、患部の効果は KV キャッシュ経由で
CoT 生成全体にわたり保持される。

- **recipient = typo 質問プロンプト** (質問のみ。CoT は与えず自由生成)
- **早期窓 residual を摂動語スパン位置で clean 値に patch** → CoT 全体を greedy 生成
- **位置整列**: exp8 セルC の `question_span` 整列を流用。アーカイブ
  `perturbed_tokens` の各語について clean 質問中 original_token / typo 質問中
  perturbed_token の文字スパンを特定し (質問領域の最終出現)、offset_mapping で
  スパン末尾トークン 1 点に対応。語 i ↔ 語 i で整列 (どちらかで見つからない語は
  両側落とし donor/recipient の行数を一致させる)。
- **恒等性 (sham) の担保**: 捕捉→再注入は bit 同一なので、recipient 自身の活性を
  同位置に注入した恒等パッチは greedy 生成を一切変えないはず。これをユニット
  (tiny Llama, 24 decode ステップ) と実モデルスモークの両方で検証する。

窓 (各モデル層数準拠、幅 6): 早期 `[0,6)` (exp8 で確定した最良窓 residual [0,6) 相当) /
中期 (中央) / 後期 `[n-6,n)`。方向: denoise (clean→pert, recipient=typo) /
noise (pert→clean, recipient=clean)。

## 指標 (flip ペア = clean 正解 ∧ typo 誤答)

- **ROUGE-L(patched, clean 生成 CoT)** と増分 `= ROUGE(patched,clean) − ROUGE(typo,clean)`
- **発散オンセット**: patched が clean 生成と分岐する最初のトークン位置 (None=完全一致=消失)
- **flip**: patched の抽出答えが正解か (restoration_rate = flip 解消率)
- 統制: **後期窓** (効かないはず)・**sham** (恒等=no-op、bit 不変)・
  **noise 方向** (clean 実行に typo 状態を注入 → 分岐・flip が誘発されるか = 十分性)

## スモーク検証 (2026-07-19, Gemma-3-4B-it × GSM8K, n=16 LXT-4)

実装検証: ユニット 30 件 (free_generation 21 + aggregate 9) + 既存介入 77 件すべて緑、
ruff クリーン。実データ 20 flip ペアで span 整列 4/4 成功 (0 drop)。

スモーク: 16/16 done, 0 除外, 0 失敗。fresh flip = 11/16 (自由生成では GSM8K でも
flip が再現。exp8 の teacher-forcing 下 ≈0% と対照的)。

| level | dir | n | restoration / induced_flip | ROUGE 増分 | ROUGE vs clean | onset 消失/誘発 |
|---|---|---|---|---|---|---|
| **early** | **denoise** | 11 | **0.909** | **+0.222** | 0.825 | **0.727** |
| mid | denoise | 11 | 0.636 | +0.152 | 0.754 | 0.636 |
| late | denoise | 11 | 0.091 | −0.009 | 0.594 | 0.273 |
| **early** | **noise** | 13 | **0.769** | — | 0.615 | **1.000** |
| mid | noise | 13 | 0.231 | — | 0.795 | 1.000 |
| late | noise | 13 | 0.000 | — | 0.890 | 0.615 |

- **sham (no-op 恒等性): 16/16 bit 不変 (1.000)** — hook が自由生成ループで正しく
  動くことを実モデルで確認。
- **事前登録 H15 判定 全項目 TRUE**: ROUGE 増分 ≥ 0.15 / flip 半減 (restoration 0.909) /
  onset 過半消失 (0.727) / 後期窓 ≈ 無効果 (増分 −0.009) / noise で flip 誘発 (0.769)。
- **質的例 (gsm8k_00107, 正解 3)**: 無パッチ typo は別解法へ逸脱し誤答
  ("Let $h_i$ be the number of hours…")。早期 denoise patched は clean 解法へ復帰
  ("Let $x$ be the number of 30-minute episodes…")、正解 3、ROUGE 0.94、分岐は
  token 52 まで後退。「早期層を直せば CoT の逸脱そのものが消える」を直接図示。

## 本番

`scripts/exp15/prod/run_queue.sh` を setsid で起動 (M3×B2, n≤300/設定,
早期/中期/後期 × denoise/noise, sham 付き, `run_with_gpu.sh` 経由, 冪等)。
GPU プールは並行系統と共有。結果は本メモに追記予定 (下記「本番集計」節)。

<!-- 本番集計 (追記予定): 設定別 ROUGE 増分・flip 減少・onset 消失表 + 3 モデル一般化の H15 最終判定 -->

## 防御的読み替え (design implication T2)

本実験が成立する — 摂動語スパンの**早期層 residual 表現を clean 値に正すだけで
CoT の逸脱が消え、下流 (推論全体 + 答え) が正解へ戻る — という事実は、主結果
(損傷→分岐→搬送の因果閉鎖) の記述にとどまらず、**早期層限定の内部介入 T2 の
存在証明**として読み替えられる。すなわち「タイポ頑健化の是正に必要な介入標的は
モデル深さのごく浅い一帯に局在し、全層を触る必要がない」。実効窓 `[0,6)` は
Gemma-3-4B (6/34 = 17.6%) / Mistral-7B (6/32 = 18.8%) / Llama-3.2-3B
(6/28 = 21.4%) いずれでも**深さの先頭 ~20% 以内**に収まり、exp8 の層別回復
プロファイル (早期 residual で最大・深さ方向に単調減衰・最終層 ≈0) とも整合する。
したがって設計含意として、**訓練時の是正介入 (例: 早期層に限定した軽量アダプタ・
表現正則化・部分ファインチューニング) の標的を早期 ~20% の層に絞れば足りる**ことを
示唆する。これは全層更新に伴う (a) 既得能力の忘却リスクと (b) 学習・計算コストを
削減する根拠になりうる。本段落は本文の主張を変えるものではなく、あくまで
防御含意 (design implication T2) としての位置づけである。
