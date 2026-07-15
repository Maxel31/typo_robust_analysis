# 実験2: R_C 標的削除の要因計画 — 設計メモ (dev notes)

担当ブランチ: `exp/02-target-deletion`(develop 分岐、環境修正 4 コミットを exp/04 から cherry-pick 済み)。
正典: `docs/experiment_plan.md` §4 実験2 / §2.2 共通道具箱 / §3.4 共通規約 / §7 実装マッピング。

## 1. 要因セル一覧

CoT 編集はすべて **clean CoT の答え句前 prefix**(`loo_scorer.split_generated_text` の
`cot_text`)上で行い、選ばれた**語タイプの全出現**を操作する(計画 §4-2-1「全出現を操作」)。
編集後 prefix + 答えトリガー(`trigger_text`)を Q_clean の下で teacher-forcing し、
答えスパンのみ短生成 → 基準腕(無編集 prefix の再生成)との flip 判定。

| 軸 | 水準 | 備考 |
|---|---|---|
| 標的 | `top_rc` / `matched_random` / `bottom_rc` (Anti) / `top_loo` (修正B) / **`top_rc_unrestricted` / `stratum_matched_random`(2026-07-15 追加)** | ランキングはアーカイブ既定 R_C(`_cot.pt`。§6.1 の理由で clean 正解母集団では fixed-target と構成上一致)。`--rc_source` で差替可能 |
| 操作 | `delete` / `mask`(「…」置換=文法破壊統制) / `replace`(同品詞・同頻度帯別語) | 無制限腕は delete のみ |
| 用量 | k ∈ {1, 2, 4} | 候補不足のサンプルは腕単位で skip(理由記録)、min(k,残り) には切り詰めない(用量反応の解釈を守る) |
| 層 | `content`(主) / `numeric`(別枠) / **`all`(無制限腕の集計ラベル)** | 数値・演算語の削除は答えの直接破壊になりうる準自明なので**別層として選定・集計とも分離**(計画 §4-2-1)。無制限腕は層をまたいで選定するため第3の集計ブロック `strata.all` に隔離 |

### 主対比の「両建て」構成(2026-07-15 ユーザー決定)

スモーク#2 の発見(内容語削除は全用量でほぼ 0%、数値削除は k=4 で 91.7%、
R_C 上位語はほぼ数値・演算語=実験6の観測と一致)を受けて、主対比を両建てに更新:

1. **無制限 top-R_C 腕を主対比に追加**: `top_rc_unrestricted` = 数値・演算語を
   含む R_C 純粋上位 k 語の削除(事前予測「top 15〜30% vs random 3〜6%」に直接
   対応する腕)。統制は `stratum_matched_random` = **層内マッチのランダム語**
   (数値標的には数値語を、内容語には内容語を、既存の頻度×文字長マッチ
   [MATCH_SCHEDULE] を層内適用で非復元抽出)
2. **層別腕(内容語 top/matched + 数値 top)も維持**: 機構分解として併記

実装: `select_top(stratum=None)`(無制限選定)+ `select_stratum_matched_random`
(語ごとに同一層でマッチ)。`core_arms()` は両建て 15 腕(delete、k∈{1,2,4})、
`full_grid_arms()` は 33+6=39 腕。`aggregate_results` は CONTRAST_PAIRS =
{(top_rc, matched_random), (top_rc_unrestricted, stratum_matched_random)} を
セル (op,k,stratum) ごとに自動対比。テスト 314 passed(301→+13)。

### 本番の設定セル(計画どおり)

- **コア対比(M5×B5、昇格済み)**: baseline + {top_rc, matched_random} × delete × k=1(content 層)
- **完全グリッド(Gemma-3-4B × B2)**: baseline + 標的3(top_rc/matched_random/bottom_rc) × 操作3 × k3 = 27 腕
- **LOO 腕(M3×B2、修正B)**: top_loo × delete × k∈{1,2,4}(ランキングは exp/06 `loo_scorer`、`deletion_mode="occurrence"` 主定義)
- **数値層(別枠報告)**: numeric top_rc × delete × k∈{1,2,4}(完全グリッド設定に併設)
- **回復曲線(M3×B2、副実験)**: セルC構成(Q_typo + clean CoT 先頭 p% 強制、p∈{0,25,50,75,100})→自由生成→回復。ジャンプ位置と top-1 R_C 語初出位置の一致率を並べ替え検定

### 標的選定の規約

- 候補プール = 切断後 prefix の語タイプ(`loo_scorer.extract_word_types`、端句読点剥がし)
- content 層: 数字・演算語でない / ストップワードでない / 正規化後 2 文字以上(→ 選択肢文字 A–J は自動排除)/ 英字を含む
- numeric 層: 数字を含む語 + 演算語(`= + - * / × ÷ %`)
- 答え句内トークンは切断により構成上排除。切断後 prefix に答え句断片が残る場合は `residual_answer_in_prefix` フラグ(主分析から除外、感度分析で込み)
- ランキング側は `expand_multiword_entries` + `normalize_word` で語タイプに正規化、タイプ重複は最大スコア採用(既存 `top_k_jaccard_by_token` と同一規約)
- `matched_random`: top_rc 標的 1 語ごとに、プール(top 集合除外)から**文字長一致(±0→±1→±2…と緩和)× Zipf 頻度帯一致(±0.25→±0.5→±1→±2→∞と緩和)**で非復元抽出。seed は `(global_seed, sample_id)` から決定論的に導出(再実行で同一標的=冪等)
- `replace` 操作: wordfreq 上位語彙から同品詞・同頻度帯(|ΔZipf|≤0.5→緩和)の別語を抽選。POS タガーは注入可能(既定はヒューリスティック。spaCy `en_core_web_sm` は appendix extra 依存のため本番採用可否は open question)

## 2. 1 サンプルあたりの forward / 生成数見積り

| 設定 | 生成数/サンプル | 種別 |
|---|---|---|
| コア対比 | 3(baseline + 2 腕) | 短生成(答えスパンのみ、max_new_tokens≈32) |
| 完全グリッド+数値層+LOO腕 | 1 + 27 + 3 + 3 = 34 | 短生成 |
| LOO ランキング(インライン時) | +出現数(≈50–200 forward、生成なし) | 本番は exp/06 の results.json を `--loo_results` で供給(上流依存を明示) |
| 回復曲線 | 5 点 × 1 長生成(≤512 tok)/ flip 事例 | flip 事例 ≈300/設定 |

GPU 見積(RTX PRO 6000 級、バッチ 16、greedy):

- コア対比 M5×B5: 25 設定 × n≈750 × 3 短生成 ≈ 56k 短生成 → **≈0.5–1 GPU 日**(小型モデル多め)
- 完全グリッド Gemma-3-4B×B2 + 数値層 + LOO 腕: 2 設定 × n≈500 × 34 ≈ 34k 短生成 + M3×B2 LOO 腕残り 4 設定 × 500 × 4 ≈ 8k → **≈0.5–1 GPU 日**
- 回復曲線 M3×B2: 6 設定 × 300 事例 × 5 点 = 9k 長生成 → **≈1 GPU 日**
- 合計 ≈2.5–3 GPU 日(計画の見積と整合)

## 3. モジュール構成と既存資産への接続

| 新規 | 内容 | 依存 |
|---|---|---|
| `intervention/cot_editor.py` | delete / mask / replace の編集オペレータ(スパン単位、全出現) | `loo_scorer.extract_word_types` / `delete_spans` |
| `intervention/target_selector.py` | 層判定(content/numeric)・top/bottom/matched-random/LOO 標的選定 | `loo_scorer.normalize_word` ほか、wordfreq |
| `intervention/replacement.py` | 同品詞・同頻度帯の置換語サンプラ(タガー注入可) | wordfreq |
| `intervention/deletion_runner.py` | 腕仕様(ArmSpec)→編集→teacher-forcing 短生成→flip 判定。generate_fn 注入(実験1 runner と同型) | `evaluation/extractor.create_extractor`、`models/prompts` |
| `intervention/recovery_curve.py` | p% prefix 構築・回復判定・ジャンプ位置と R_C 初出位置の一致・並べ替え検定 | |
| `intervention/deletion_stats.py` | McNemar 厳密検定+対応 bootstrap リスク差 CI・用量反応の並べ替え単調性検定・腕別集計(層分離) | scipy |
| `scripts/exp2/run_target_deletion.py` | CLI(シャード `--start/--end`、進捗の原子的保存、sample_id ベース resume=冪等) | |
| `scripts/exp2/run_recovery_curve.py` | 回復曲線 CLI(baseline+perturbed の対を入力) | |

既存資産:

- 入力: アーカイブ `outputs/baseline/{model}_{bench}/results.json`(prompt 再構築は `run_loo_scoring.build_prompt` と同一規約)+ `importance_scores/{sid}_cot.pt`(R_C)。回復曲線は `outputs/perturbed/..._k4_importance` 等の typo 質問側も使用
- 生成: `models/wrapper.ModelWrapper.generate_batch`(左パディング greedy、実験1と同じ経路)。lxt 不要(`wrap_for_lxt=False`)、backward 不要
- flip 判定: `evaluation/extractor.create_extractor(benchmark)` を全腕で完全同一に適用(`trigger_text + 生成継続` に対して)
- 統計: 主推定量 = clean 正解条件付き correct→incorrect(§3.4-2)。対比は McNemar+リスク差 CI、用量反応と回復曲線は並べ替え検定

## 4. 判定・集計の定義(凍結)

- **flip**: 腕の抽出答え ≠ 基準腕(無編集 prefix 再生成)の抽出答え(strip 比較)。基準腕とアーカイブ抽出答えの一致率(`baseline_matches_archive`)を検証列として保存(greedy なら ≈1 のはず)
- **主推定量**: アーカイブ clean 正解(`is_correct`)サンプルに限定した flip 率(= correct→incorrect と同値になる構成)。全サンプル版は副次として summary に併記
- **層の分離**: summary は `strata.content` / `strata.numeric` に腕を分けて集計。混合しない
- **コア対比**: top_rc_delete_k1 vs matched_random_delete_k1 の McNemar(厳密二項)+ リスク差の対応 bootstrap 95% CI
- **用量反応**: 同一標的×操作の k∈{1,2,4} で、率の傾き(flip 率 vs k の最小二乗傾き)を統計量、サンプル内で用量ラベルを並べ替える permutation 検定
- **回復曲線**: ジャンプ位置 = 回復が最初に真になる格子点 p\*、区間 (p_prev, p\*]。一致 = top-1 R_C 語(content 層)の初出文字位置比が区間内。帰無分布 = 同一 CoT の content 候補から一様抽選した語で同判定(B=2000)

## 5. スモーク計画(1 ジョブ 30 分以内)

- Gemma-3-4B-it × GSM8K、アーカイブ baseline 先頭から clean 正解 n=24
- 腕: baseline + top_rc/matched_random × delete × k1(コア対比)+ top_loo × delete × k1(LOO インライン)+ numeric top_rc × delete × k1(別枠実演)
- 合格: (a) 腕別 flip 率が算出され top > random 方向、(b) results/summary スキーマが分析側(§4)と整合、(c) numeric 層が content 層と分離集計

## 5.5 CPU ドライラン検証(2026-07-15、GPU 不要部分)

アーカイブ実データ (gemma-3-4b-it_gsm8k、clean 正解先頭5件) で prepare_sample を検証:

- top_rc 標的: dollars / bolts / week / cups / total(答え句前 prefix の内容語、妥当)
- matched_random: ducks / robe / times / meal / wants(top と別語、長さ・頻度帯マッチ)
- numeric 層: 18 / 3 / 540 / 20 / 64 — **gsm8k_00000 では CoT 中の最終計算結果 18
  がそのまま標的になる**(= 準自明性の実例。別枠報告の設計根拠を裏づけ)
- skip 理由・residual フラグの動作確認済み。top_loo は ranking 未供給時に
  missing_ranking で腕 skip(設計どおり)

## 5.6 GPU スモーク結果 (2026-07-15)

**スモーク#1** (`results/smoke/exp2/gemma-3-4b-it_gsm8k_smoke/`): Gemma-3-4B-it ×
GSM8K、clean 正解 n=24、arms=smoke (コア対比 + LOO腕 + 数値層、各 delete k=1)、
LOO インライン。GPU 6、実行 2.2 分 (5.3s/sample、LOO インライン込み)。

- **基準腕の妥当性: matches_archive = 1.0 (24/24)** — greedy 再生成がアーカイブの
  抽出答えを完全再現(teacher-forcing 経路の検証として最強の合格線)
- 腕別 flip 率 (clean 正解条件付き): top_rc_delete_k1 = 0/24、
  matched_random_delete_k1 = 0/24、top_loo_delete_k1 = 0/24、
  **numeric_top_rc_delete_k1 = 1/24 (4.2%)**(例: '125' 削除 → 答え 125→96)
- 合格判定: (b) スキーマ整合 OK、(c) 数値層の分離集計 OK。
  (a) top>random の方向は k=1・n=24 では両腕 0 で未観測 → 用量を上げた
  スモーク#2 (arms=full、同一24サンプル) で確認
- 解釈メモ: GSM8K の R_C 上位はほぼ数値トークンであり (cot_top_k_words 参照)、
  内容語だけを k=1 で消しても計算結果の数値が CoT に残るため答えが復元される
  のは整合的。数値層 4.2% > 内容語層 0% はまさに「数値=準自明、別枠」の設計を
  裏づける方向

**スモーク#2 (用量反応の確認、状態=queued)**: 同一24サンプルで arms=full
(標的3×操作3×k{1,2,4} + 数値層 + LOO腕 = 33腕) を `setsid nohup` で投入済み
(ログ: `logs/exp2_smoke_full.log`、出力:
`results/smoke/exp2/gemma-3-4b-it_gsm8k_smoke_full/`)。GPU ロック輻輳のため
数時間待機中 (ロック待ちは正常)。完了後は summary.json の
`strata.content.arms.top_rc_delete_k4` と `dose_trends` で (a) の方向
(top>random、用量反応) を確認する。冪等 (resume 対応) なので再投入も安全。

## 6. 本番の設定リスト(確定版、2026-07-15 キュー起動)

| # | 設定 | CLI | 規模 | 見積 |
|---|---|---|---|---|
| P1 | コア対比 M5×B5 (25設定、両建て15腕) | `--arms core --clean_correct_only --n 500` | ~500×16短生成/設定 | ~0.5–1 GPU日 |
| P2 | 完全グリッド Gemma-3-4B×B2 (2設定、39腕) | `--arms full --loo_inline` | ~500×40短生成+LOO/設定 | ~0.3 GPU日 |
| P3 | LOO腕 残り M3×B2 (Llama-3.2-3B / Mistral-7B × B2 = 4設定) | `--arms loo --loo_inline` | ~500×4短生成+LOO/設定 | ~0.2 GPU日 |
| P4 | 回復曲線 M3×B2 (6設定) | `run_recovery_curve.py --n 300` | ~300 flip事例×5長生成/設定 | ~1 GPU日 |

合計 ≈2–2.5 GPU 日。スモーク#2 実測 7.6s/sample(34生成、gemma-3-4b)から
P1 は ≈3.5s/sample × 500 ≈ 30–60分/設定。

### 6.1 R_C ソースの決定(技術判断)

実験4完成済みの fixed-target 出力(exp-04-fixed-target worktree の
`results/prod/fixed_target/{model}_{bench}_k4_fixed_target/`)を精査した結果、
これは **摂動質問側の run**(perturbed_dataset 入力、generated_text = typo 質問下の
CoT、`_cot.pt` = その CoT の語ランキング)であり、本実験が編集する **clean CoT**
の語ランキングではない(アーカイブ baseline と generated_text 一致 86/1230)。
そのため本番の R_C ソースは**アーカイブ既定 `_cot.pt`**(clean run、標的=自答)を
採用。主分析母集団は clean 正解サンプルに限定されるため、そこでは
**既定 R_C ≡ fixed-target R_C(自答=正答で帰属先が一致)**であり計画の意図と
整合する。→ open question として記録(§8)。

LOO は exp/06 に本番 results.json が未出力(スモークのみ)のため、P2/P3 は
`--loo_inline`(occurrence 主定義)で自前計算。exp/06 本番完成後の照合は可能。

### 6.2 キュー運用

- シャード一覧: `uv run python scripts/exp2/make_queue.py` →
  `results/prod/exp2/queue/shards_active.tsv`(37行 = P1 25 + P2 2 + P3 4 + P4 6。
  検証用に gemma-3-4b × B2 のコア対比を先頭に配置)
- ワーカー: `scripts/exp2/queue_worker.sh`(exp1+3 方式: claim/failed/進捗JSON/
  STOP/一覧再読込、GPU は run_with_gpu.sh 経由、rc=86 PAUSED で退出、
  rc=124 ロック待ちタイムアウトは再試行)
- 起動: `cd <project> && WORKER_ID=w1 setsid nohup bash scripts/exp2/queue_worker.sh < /dev/null >> logs/exp2_queue/worker_w1.log 2>&1 &`
- 監視: `bash scripts/exp2/queue_status.sh` / 停止: `touch results/prod/exp2/queue/STOP`
- 失敗再試行: `rm results/prod/exp2/queue/failed/<name>`(CLI は sample_id resume)

## 6.5 ミニ検証(両建て core、Gemma-3-4B×GSM8K n=24、スモークと同一サンプル)

**合格(2026-07-15、GPU 5、2.5s/sample)**。出力:
`results/smoke/exp2/gemma-3-4b-it_gsm8k_core_smoke/`。

- 基準腕: matches_archive = **1.0 (24/24)**(スモーク#1 と同じ合格線を再現)
- **主対比(strata.all、clean 正解条件付き)**:
  top_rc_unrestricted = 4.2 / 16.7 / **66.7%** (k=1/2/4)、
  stratum_matched_random = 4.2 / 8.3 / 4.3%
- **k=4 対比: RD = 0.609、CI95 = [0.391, 0.826]、McNemar p = 1e-4 (b=14, c=0)**
  → 事前予測の方向(top ≫ random)を大差で確認
- 用量反応: 無制限 top は単調(slope 0.214、p=5e-4)、統制はフラット
- 層別腕はスモーク#2 を再現(content ≈0%、numeric 4.2/12.5/91.7%)
- 解釈メモ: 無制限 top 66.7% < 数値層 top 91.7% は、無制限 top-4 に内容語や
  '=' が混ざる希釈効果。統制が全用量 ≈4% に留まるのは「ランダムな数値を消しても
  flip しない」ことを意味し、効果が **R_C 上位の数値に特異的**である証拠
  (準自明性批判への直接の反駁材料)

CPU ドライラン(clean 正解先頭5件)では無制限 top4 = ['18','dollars','=','9'] 等
(数値・演算語優位)、層内マッチ = ['16','breakfast','-','4'](数値↔数値、
内容↔内容)で設計どおり。

## 6.6 本番キュー起動と最初のシャード検証(2026-07-15)

ワーカー w1 (sid 4129304)・w2 (sid 4130718) を setsid で起動。最初の完了2シャード
(いずれも P1 core、gemma-3-4b、0.8s/sample)を検証し**合格** — キュー続行:

| 設定 | n(clean正解) | matches_archive | 主対比 top (k=1/2/4) | 統制 | k=4 RD [CI] | p |
|---|---|---|---|---|---|---|
| mmlu | 473 | 1.000 | 10.8/15.4/20.7% | 1.5/1.5/2.1% | 0.172 [0.135,0.209] | 4e-19 |
| mmlu_pro | 479 | 0.996 | 16.7/20.7/25.5% | 2.1/2.3/2.4% | 0.209 [0.174,0.252] | 3e-26 |

- 事前予測帯(top 15〜30% vs random 3〜6%)に整合。用量反応単調 (p=5e-4)
- MMLU では content 層でも top>matched (k=4: 9.5% vs 1.7%, p<0.001) —
  GSM8K(内容語 ≈0%)とのベンチ差は「答え形式モデレーター」仮説と整合
- numeric 腕は候補不足 skip が多い (MMLU で 270–362/473) — 設計どおり別枠報告
- 続報(同日): arc RD=0.037 (p=5e-4)、csqa RD=0.063 (p=2e-7) — 方向維持のまま
  易ベンチで効果縮小(early answering の事前予想と整合)。gsm8k×gemma-3-1b は
  **RD=0.803 (p=8e-111)**(top 85.0% vs 統制 4.2%)

### 基準腕診断: 小型モデルの matches_archive 低下(gemma-3-1b×gsm8k = 0.816)

1B では teacher-forcing 境界での greedy 再生成が archive 答えから外れる事例が
92/500(すべて再生成側が不正解に逸れる)。4B は 0.994〜1.0。**対比は同一基準腕
との対比較なので内的整合は保たれる**うえ、record に `baseline.matches_archive`
が保存されているため下流で制限感度分析が可能。実測: matches_archive=True の
408件に制限すると主対比はむしろ強まる(k=4 top 88.7% vs 1.5%、RD=0.866、
p=5e-98)→ キュー続行、論文分析では matches_archive 制限版を感度分析として併記。

## 7. 実装中の技術判断(ユーザー判断不要と整理したもの)

1. 答え句分割は `loo_scorer.split_generated_text`(最後のマッチ採用)に統一 — LOO 腕・log-prob 系と規約を揃える。exp1 の cell_builder(最初のマッチ)とは異なるが、prefix 内答え句残留は `residual_answer_in_prefix` フラグで除外制御
2. 標的の「全出現操作」(計画の明文)と LOO ランキングの occurrence 主定義(ユーザー決定)は独立の事項 — 前者は編集、後者は順位付け
3. matched_random の照合変数は頻度(Zipf)+文字長のみ(計画 §4-2-2 の明文どおり。実験5 の 5 変数マッチングとは別物)
4. bottom_rc(Anti)は |score| でなく signed score の下位から選ぶ(負の relevance も「答えに寄与しない」側)ただし content 層フィルタ適用後
5. 無制限腕の操作は delete のみ(数値・演算語への同品詞 replace は定義不能、mask は数値でも文法破壊統制にならないため)。層別グリッドの mask/replace が操作アーチファクト統制を担う
6. smoke プリセットは両建て化後も旧 4 腕に固定(スモーク#1/#2 との比較可能性)

## 8. Open questions(ユーザー確認待ち、作業は継続)

1. **R_C ソース**: 計画 §4 実験2 は「実験4の fixed 版ランキングを参照」と明記するが、
   実験4の完成出力は摂動側 CoT のランキングであり clean CoT 編集には構成上使えない
   (§6.1)。本番はアーカイブ既定 R_C で起動済み(clean 正解母集団では fixed-target と
   一致するため実害なしと判断)。clean run の fixed-target 再計算を別途要するかは
   ユーザー判断。
2. **LOO 供給源**: exp/06 の本番 LOO results.json が未出力のため P2/P3 は
   `--loo_inline`(occurrence)で自前計算。exp/06 完成後にランキング一致の照合を
   するか、P2/P3 を exp/06 供給版で再実行するかはユーザー判断(コストは小)。
3. **replace 操作の POS タガー**: 既定ヒューリスティックのまま本番実行(P2 のみが
   replace を含む)。spaCy 昇格の要否は従来どおり open。
4. **stratum_matched_random の解釈**: 統制腕も数値語を削除するため、「ランダム数値
   削除でもある程度 flip する」ことが予想される。主対比の帰無は「同層・同頻度帯・
   同文字長のランダム語と同等」であり、これが計画の事前予測(3〜6%)より高く出た
   場合の紙面上の位置づけ(数値層の準自明性の定量化として報告)はユーザーと要相談。
