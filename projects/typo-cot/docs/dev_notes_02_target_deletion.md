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
| 標的 | `top_rc` / `matched_random` / `bottom_rc` (Anti) / `top_loo` (修正B) | ランキングは fixed-target R_C(実験4、上流)。スモークではアーカイブ既定 R_C(`_cot.pt`)で代用し `--rc_source` で差替可能に |
| 操作 | `delete` / `mask`(「…」置換=文法破壊統制) / `replace`(同品詞・同頻度帯別語) | |
| 用量 | k ∈ {1, 2, 4} | 候補不足のサンプルは腕単位で skip(理由記録)、min(k,残り) には切り詰めない(用量反応の解釈を守る) |
| 層 | `content`(主) / `numeric`(別枠) | 数値・演算語の削除は答えの直接破壊になりうる準自明なので**別層として選定・集計とも分離**(計画 §4-2-1) |

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

## 6. 本番の設定リスト案と GPU 見積

| # | 設定 | CLI | 規模 | 見積 |
|---|---|---|---|---|
| P1 | コア対比 M5×B5 (25設定、昇格済み) | `--arms core --clean_correct_only` | ~750サンプル×3短生成/設定 | 0.5–1 GPU日 |
| P2 | 完全グリッド Gemma-3-4B×B2 (2設定) | `--arms full --loo_results <exp6>` | ~500×34短生成/設定 | 0.5 GPU日 |
| P3 | LOO腕 残り M3×B2 (Llama-3.2-3B / Mistral-7B × B2 = 4設定) | `--arms loo --loo_results <exp6>` | ~500×4短生成/設定 | 0.2 GPU日 |
| P4 | 回復曲線 M3×B2 (6設定) | `run_recovery_curve.py` | ~300 flip事例×5長生成/設定 | ~1 GPU日 |

合計 ≈2.5 GPU 日(計画 §4 実験2 の 2.5〜3 日と整合)。全 CLI がシャード
(--start/--end)+resume 対応なのでロック待ち環境でも分割投入可能。
R_C ソースは実験4の fixed-target ランキング完成後に `--baseline_dir`(または
importance_scores 差し替え)で切替。LOO は exp/06 の results.json を供給。

## 7. 実装中の技術判断(ユーザー判断不要と整理したもの)

1. 答え句分割は `loo_scorer.split_generated_text`(最後のマッチ採用)に統一 — LOO 腕・log-prob 系と規約を揃える。exp1 の cell_builder(最初のマッチ)とは異なるが、prefix 内答え句残留は `residual_answer_in_prefix` フラグで除外制御
2. 標的の「全出現操作」(計画の明文)と LOO ランキングの occurrence 主定義(ユーザー決定)は独立の事項 — 前者は編集、後者は順位付け
3. matched_random の照合変数は頻度(Zipf)+文字長のみ(計画 §4-2-2 の明文どおり。実験5 の 5 変数マッチングとは別物)
4. bottom_rc(Anti)は |score| でなく signed score の下位から選ぶ(負の relevance も「答えに寄与しない」側)ただし content 層フィルタ適用後
