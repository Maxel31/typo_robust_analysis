# ARR August 2026 Resubmission: 実験計画と結果の詳細整理

本書は、resubmission 実験計画(experiment_plan.md)と各 worktree の開発メモ・結果データを統合し、論文執筆の参考として各実験の目的・手法・実装・予想結果・実際の結果を実行順に詳細にまとめたものである。

> **2026-07-19 更新**: Phase B(ERDC 拡張)の新規完了分 **実験11(連鎖媒介)・実験12(R_C組成)・
> 実験13(読み出し集中度)・実験14(no-CoT ショートカット)・実験17(行動修復)** の実際の結果を
> 巻末「Phase B (ERDC 拡張): 実験11-17 の実際の結果」節に追記。**敵対的レビュー A(A1/A2)・B(B1/B4/B5/B6)・
> C(実験6 ρ保持 修正版)・防御 D1** を「敵対的レビュー対応(2026-07-19)」節に統合(A1/A2 は既存の
> 実験4・実験1+3 節の監査段落を要約参照)。数値は各 `analysis/` 配下ソースから直接転記。

---

## 全体の背景: 改訂戦略

- **現状スコア**: R1 2.5 / R2 3.0 / R3 2.5 (short paper)
- **改訂の背骨**: 「相関診断の論文」から「介入で裏づけた媒介分析の論文」への格上げ
- **三層再編**: Surface (テキスト) / Attribution (答え関連トークン) / Representation (隠れ状態)
- **4批判グループ**: (1) 因果の飛躍 (2) 測定の信頼性 (3) 防御の主張過大 (4) 「なぜ」とスコープ不足
- **修正3系統**: A (Random-4 複製) / B (LOO 複製) / C (提示設計)

### 記号定義

- **M5**: Llama-3.2-1B / Llama-3.2-3B / Mistral-7B / Gemma-3-1B / Gemma-3-4B (instruction-tuned)
- **M3**: Gemma-3-4B / Llama-3.2-3B / Mistral-7B
- **B5**: GSM8K / MMLU / MMLU-Pro / ARC / CSQA
- **B2**: GSM8K + MMLU
- **flip**: clean 条件の答えと異なる答えになること。主推定量は clean で正解だったサンプルに限定した correct→incorrect

---

## Step 0: 資産棚卸しと凍結 [完了]

### 目的

以降の全実験の前提となる統合データ基盤を構築し、実験条件を凍結する。全25設定 (M5 x B5) の既存ログ (clean / LXT-4 / Random-4) を単一テーブルスキーマに統合し、prompt/seed/decoding 設定をレジストリとして凍結する。査読批判との直接対応はないが、全実験の「再利用トリック」の前提を整え、匿名リポジトリ公開 (Software=1 対策) と失敗率表の素材を提供する。

### 手法

1. 全25設定の既存ログ (clean / LXT-4 / Random-4 の生成、$R_Q$・$R_C$ 帰属、flip 判定、span 抽出失敗) を単一テーブルスキーマ (parquet) に統合
2. prompt/seed/decoding 設定をレジストリとして凍結
3. 指標に「本文/付録」ラベルを付与 (修正C)
4. Surface/Attribution 軸名を図表出力に最初から適用

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/data/master_table.py` | スキーマ・条件・main/appendix ラベルの凍結 |
| `src/typo_cot/data/master_builder.py` | 行構築ロジック (純粋関数) |
| `src/typo_cot/data/archive_reader.py` | アーカイブ読み取り層 (読み取り専用・sha256) |
| `src/typo_cot/analysis/reproduce.py` | 条件別精度・偏相関の再現集計 |
| `src/typo_cot/registry.py` | config レジストリ検証 |
| `configs/registry.yaml` | prompt hash / seed / decoding 凍結 |
| `scripts/step0_build_master_table.py` | 統合テーブル構築スクリプト |
| `scripts/step0_smoke_reproduce.py` | スモーク検証 |

スキーマは 1行 = 1サンプル x 1モデル x 1ベンチ x 1条件。列は `sample_id / model / benchmark / condition / question_text / cot_text / answer_span / answer_pred / answer_gold / flip / cot_rouge_l_f1 / cot_jaccard_top{k} / r_q / r_c / span_extract_ok / seed / prompt_id` + 追加列 (`is_correct`, `pattern`, `subset`, `original_question`, `perturbed_tokens`, `source_path`)。

凍結条件: seed=42, greedy decoding (do_sample=false, temperature=0.0, max_new_tokens=512)。指標ラベル: main = ROUGE-L, Jaccard@10, flip; 他は appendix。

### 予想される結果

全25設定 x 3生成条件のカバレッジ確認と、span 失敗全数表 (全体7.28%、最大31.16% = Mistral-7B x GSM8K) の再現。

### 実際の結果 [完了: 2026-07-14]

- **精度**: summary.json 150/150 セル一致、figures/table5.csv 90/90 セル一致
- **偏相関** (k=10, lxt4): 25/25 設定で $\rho(J|R)$・$\rho(R|J)$・$n$ が analysis と一致 (atol 1e-9)
- **span 除外整合**: 25/25 設定一致。union 除外率 全体 13.19%、最大 38.59% (Mistral-7B x GSM8K)
  - 注意: experiment_plan.md の「全体 7.28%・最大 31.16%」とは定義が異なる (rebuttal 期の lenient/per-pair 変種)
- **ハッシュ検証**: `--verify` で移行元 400 ファイルの sha256・150 parquet の行数 OK
- **合計**: 238,855 行 (25設定 x 6条件、うち clean 39,810行)

### wave2 拡張 [完了: 2026-07-18]

150 parquet (v1 25設定×6条件) に 67 セルを追加し **217 parquet / 342,653 行**
に拡張 (exp-step0 worktree、dev_notes_step0.md wave2 節、commit `a681fbf`)。

| グループ | セル | 行数 | 取込元 |
|---|---|---|---|
| v1 (既存) | 150 | 238,855 | アーカイブ (再ビルドで 150/150 byte 一致) |
| Anti-LXT-4 (k4_bottom_k) | 25 | 39,805 | アーカイブ perturbed (analysis 無 → flip/CoT 指標 NA) |
| MATH-500 再生成 (M5×3条件) | 15 | 7,500 | exp-10-scope outputs |
| Qwen2.5-7B (B5×3 + math×3) | 18 | 33,936 | clean(B5) はアーカイブ、他は exp-10-scope |
| R1蒸留 (gsm8k/math/mmlu×3条件) | 9 | 22,557 | exp-10-scope outputs (`<think>` 形式) |

- union 除外の意味論は `V1_UNION_CONDITIONS` (5摂動条件、bottom_k 除外) に凍結。
  R1 の `<think>` 形式は cot_text 列 = think 部、strict span 抽出 = `</think>` 後の
  answer_text のみ。registry に Qwen2.5-7B-Instruct / DeepSeek-R1-Distill-Qwen-7B /
  math prompt (hash 凍結) / anti_lxt4 / reasoning_prompts を追加。
- 検証: `--verify` 217 エントリ・移行元 520 ファイル OK。スモーク accuracy 175/175
  (anti_lxt4 25セル含む)、table5 90/90、偏相関 25/25、span 除外 25/25。テスト 201 passed。
- R1 の strict span 失敗は clean 11.3〜16.2% / 摂動 12.6〜28.0% (truncation +
  reasoning 系の書式ゆれで v1 モデルより高め)。

---

## 実験4: fixed-target 分析の全設定展開 [完了]

### 目的

内的軸の内生性 (R3-W2) を除去した測定を主指標に昇格させ、主図 Fig.3 を fixed-target 版に差し替える。帰属先を「typo 後に変わった答え」ではなく**元の答え**に固定することで、答えが変わると測定の当て先ごと変わるという内生性批判に対処する。

rebuttal の4設定で見つかった非対称 (MMLU で大幅減衰: $\rho$ -0.493→-0.175 (Gemma-3-4B), -0.618→-0.109 (Llama-3.2-3B) / GSM8K で不変: -0.509→-0.546, -0.529→-0.540) が「多肢選択形式に特有の目標依存」なのか答え形式一般の現象なのかを全設定で確定する。

**修正A/B適用点**: なし (修正C適用: 本文は fixed Jaccard@10 のみ・Attribution 軸への改名)。

### 手法

1. $R_C^{\text{fixed}}(c_j)$ の定義を式で凍結:

$$R_C^{\text{fixed}}(c_j) = \text{AttnLRP}\left(\text{logit}(a^{\text{clean}}) \mid \text{teacher-force}(Q_p, \text{CoT}_p, \text{trigger})\right)$$

摂動側の生成を答え位置まで teacher-force し、その位置での「元の答えトークン」の logit へ AttnLRP を適用した帰属。多肢選択 = 元の選択肢文字、GSM8K 等の自由記述 = 元の数値トークン列の平均。

2. **再利用トリック**: clean 側の $R_C$ は「生成答え = 元の答え」なので default 版と同値。摂動側も flip しなかった事例は答え文字列が一致するため同値。**再計算が必要なのは flip 事例 (約2〜3割) のみ** = 実質コストは元の帰属計算の約1/4。

3. flip 事例のみ AttnLRP 再計算 → fixed 版 CoT:Jaccard@{5,10,20} を算出し、flip との偏相関 $\rho(J|R)_{\text{fixed}}$ を全設定で計算 (bootstrap 95%CI + Holm)。

4. $\Delta\rho = \rho_{\text{fixed}} - \rho_{\text{default}}$ の paired bootstrap + Holm 検定。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/attribution/fixed_target.py` | 回答パターン検出、splice 計画、compare_cot_payloads、GPU 部 (build_prompt / analyze_cot_fixed) |
| `src/typo_cot/analysis/fixed_stats.py` | partial_corr_flip ($\rho(J|R)$)、bootstrap_partial_corr_ci、holm_adjust、paired_bootstrap_delta_rho ($\Delta\rho$) |
| `src/typo_cot/data/run_io.py` | アーカイブ run ディレクトリへの薄い読み取り層 |
| `scripts/run_fixed_target.py` | 全設定一般化ランナー (--flip_only) |
| `scripts/analyze_fixed_target_delta.py` | $\Delta\rho$ 全設定表の生成 (JSON+CSV, B=10,000, Holm) |

規約の凍結: ANSWER_PATTERNS は rebuttal 実装と同一 (テストで機械検証)。$\rho(J|R)$ は pingouin.partial_corr と厳密一致する残差 Pearson (一次偏相関)。

### 予想される結果 (壁打ち事前予測)

自由記述 (GSM8K+MATH) は $\Delta\rho \approx 0$ ($\rho \approx -0.5$ を維持)、多肢選択4種は +0.2〜+0.5 減衰しつつ全設定で Holm 有意残存。

### 実際の結果 [完了: 2026-07-14, 25設定]

**GPU スモーク**: rebuttal 4設定で $\rho$ の完全再現を確認。GSM8K: Gemma -0.5089→-0.5455 / Llama -0.5292→-0.5396。MMLU: Gemma -0.4932→-0.1750 / Llama -0.6181→-0.1088。

**本番 $\Delta\rho$ 表** (k=10, B=10,000, 全25設定): 事前予測とは大きく異なるパターンが判明。自由記述 vs 多肢選択の単純な二分ではなく、**モデル家族によって減衰方向が反転する**:

| 設定 | $\rho_{\text{default}}$ | $\rho_{\text{fixed}}$ | $\Delta\rho$ | $\rho_{\text{fixed}}$ Holm有意 |
|---|---|---|---|---|
| **自由記述 (GSM8K)** | | | | |
| Gemma-3-4B x GSM8K (n=1141) | -0.509 | -0.539 | -0.030 (n.s.) | Yes ($p$=6.8e-87) |
| Gemma-3-1B x GSM8K (n=962) | -0.495 | -0.476 | +0.019 (n.s.) | Yes ($p$=4.5e-54) |
| Llama-3.2-3B x GSM8K (n=1022) | -0.529 | -0.544 | -0.015 (n.s.) | Yes ($p$=2.9e-78) |
| Llama-3.2-1B x GSM8K (n=1033) | -0.566 | -0.730 | -0.164 (sig.) | Yes ($p$=4.3e-171) |
| Mistral-7B x GSM8K (n=810) | -0.338 | -0.499 | -0.162 (sig.) | Yes ($p$=5.6e-51) |
| **多肢選択 (MMLU 代表)** | | | | |
| Gemma-3-4B x MMLU (n=2552) | -0.493 | -0.174 | +0.319 (sig.) | Yes ($p$=1.1e-17) |
| Gemma-3-1B x MMLU (n=2487) | -0.465 | +0.029 | +0.494 (sig.) | No (Holm $p$=0.58) |
| Llama-3.2-3B x MMLU (n=2448) | -0.618 | -0.110 | +0.508 (sig.) | Yes ($p$=5.2e-7) |
| Llama-3.2-1B x MMLU (n=2388) | -0.631 | -0.057 | +0.574 (sig.) | Yes (Holm $p$=0.036) |
| Mistral-7B x MMLU (n=2591) | -0.294 | -0.366 | -0.072 (sig.) | Yes ($p$=2.1e-81) |
| **多肢選択 (ARC/CSQA/MMLU-Pro 代表)** | | | | |
| Llama-1B x ARC (n=1159) | -0.704 | -0.037 | +0.668 (sig.) | No (Holm $p$=0.58) |
| Llama-1B x CSQA (n=1201) | -0.685 | -0.124 | +0.561 (sig.) | Yes ($p$=1.6e-4) |
| Mistral x ARC (n=1145) | -0.308 | -0.389 | -0.082 (sig.) | Yes ($p$=1.9e-41) |
| Mistral x CSQA (n=1205) | -0.380 | -0.455 | -0.076 (sig.) | Yes ($p$=2.2e-61) |

主要な発見:
- **Gemma/Llama 家族の多肢選択**: $\Delta\rho > 0$ (大幅減衰)。多くの設定で $\rho_{\text{fixed}} \approx 0$ (相関がほぼ消失)
- **Mistral-7B は全設定で $\Delta\rho < 0$** (fixed で相関がむしろ強化)。GSM8K でも多肢選択でも同方向
- **GSM8K**: Gemma-4B/Llama-3B/Gemma-1B では $\Delta\rho \approx 0$ (予測通り)。Llama-1B/Mistral は $\Delta\rho < 0$
- **Holm 確定値 (2026-07-18, m=25, `analysis/holm_correction/`)**: $\rho_{\text{fixed}}$ (k=top10) は raw 有意 21/25 → **Holm 有意 19/25**。非有意6設定 = Llama-1B {arc, mmlu_pro}、Gemma-1B {arc, csqa, mmlu, mmlu_pro} (いずれも $\rho_{\text{fixed}} \approx 0$ の多肢選択)

### MATH-500 拡張 [完了: 2026-07-18, 6モデル]

exp-04 worktree `results/prod_math/delta_rho/delta_rho_table.json` (k=top10, B=10,000):

| 設定 | n | $\rho_{\text{default}}$ | $\rho_{\text{fixed}}$ | $\Delta\rho$ | $\Delta\rho$ p | $\rho_{\text{fixed}}$ Holm p |
|---|---|---|---|---|---|---|
| Gemma-1B x MATH | 162 | -0.254 | -0.236 | +0.018 | 0.69 | 0.0079 |
| Gemma-4B x MATH | 216 | -0.296 | -0.254 | +0.042 | 0.084 | 0.00065 |
| Llama-1B x MATH | 288 | -0.373 | -0.296 | +0.077 | 0.015 | 1.7e-06 |
| Llama-3B x MATH | 313 | -0.251 | -0.154 | +0.097 | 0.0002 | 0.013 |
| Mistral-7B x MATH | 295 | -0.162 | -0.306 | -0.144 | 0.0002 | 5.4e-07 |
| Qwen2.5-7B x MATH | 274 | -0.120 | -0.157 | -0.036 | 0.15 | 0.013 |

**判定: 自由記述 (MATH) で「内的軸頑健」が6モデル全てで再現**。$\rho_{\text{fixed}}$
は全6設定で Holm 有意に残存し、$|\Delta\rho| \leq 0.144$ で多肢選択の大幅減衰
(+0.42〜+0.67) は生じない。Mistral の $\Delta\rho < 0$ (fixed で強化) は MATH でも
再現し、第4家族 Qwen も同符号傾向 (n.s.)。

**A1 整合性監査 [2026-07-19, `analysis/a1_mistral_audit/`]**: GPU スモークは rebuttal 4設定
(Gemma/Llama) のみ再現検証したため、Mistral の $\rho_{\text{default}}$ を独立検算。
$\rho_{\text{default}}$ の入力 $j_{\text{default}}$ = `cot_metrics.jaccard.top10` は
`analyzer._compute_jaccard_metrics_by_token`(= `top_k_jaccard_by_token`, トークン文字列
dedup)で **`_cot.pt` の `token_scores` から**算出され、fef3958 で潰れた `word_scores` は
この経路に一切現れない。アーカイブ `token_scores` からトークンベース Jaccard@10 を
再計算した結果、Mistral 3設定 (GSM8K/ARC/CSQA) 各60サンプルで記録値と **完全一致**
(max_abs_diff = 0.0)、$\rho_{\text{default}}$ 再計算も -0.338/-0.308/-0.380 と一致。
**Mistral の $\rho_{\text{default}}$ は汚染なし・再計算不要。「Mistral だけ $\Delta\rho<0$
(逆方向)」= H4 family×format 交互作用はアーティファクトではなく無傷。** なお本文
「全設定で $\Delta\rho<0$」は MMLU-Pro のみ +0.006 (n.s.) の例外があり、正確には
「MATH 含む主要設定で $\Delta\rho<0$」。

---

## 実験5: マッチド統制の拡張 [完了]

### 目的

LXT 標的の優位が「重要だから」なのか「たまたま長い・珍しい・分割されやすい語だから」なのかの切り分け (R3-W3)。rebuttal のマッチは品詞・文字長のみだったため、5変数の層化マッチングに拡張し、残余交絡 (頻度・意味中心性・subword 断片化) を統制する。

**修正A/B適用点**: なし (修正C適用: 本文は Table 3 の1行、全表は付録)。

### 手法

1. **層化マッチ変数5種**: (1) 内容/機能語の別、(2) 文字長$\pm 1$、(3) Zipf 頻度ビン (wordfreq、0.5刻み)、(4) 摂動による分割数増分、(5) 意味中心性の近似 (質問文埋め込み all-MiniLM-L6-v2 との cos 類似ビン)
2. 緩和ラダー: L0 exact → L1 no_centrality → L2 caliper → L3 class_len → L4 any
3. 摂動注入は LXT と同一手続き
4. LXT-4 vs Matched-Rnd-4 の McNemar + リスク差 CI

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/perturbation/matched_sampler.py` | 5変数の層化マッチング (FeatureExtractor / MatchedTwinSampler / compute_smd_table) |
| `src/typo_cot/perturbation/matched_dataset.py` | Matched-Rnd-4 データセット作成 |
| `src/typo_cot/analysis/matched_control.py` | McNemar exact + リスク差 Wald 95% CI |
| `scripts/exp5/make_matched_twin_dataset.py` | データセット作成 CLI |
| `scripts/exp5/analyze_matched_control.py` | 統計集計 |

### 予想される結果 (壁打ち事前予測)

多肢選択系はフルマッチ後も LXT 優位が有意残存 (リスク差 1.5〜3pt)、GSM8K 系は差がさらに縮小・非有意が複数。

### 実際の結果 [完了: 2026-07-15, 25設定]

**マッチバランス**: 全25設定で embedding_enabled=true (5変数版)。class_match_rate 24/25 で $\geq 0.99$。char_len / split_increment は全設定 |SMD| < 0.1。zipf 5設定・centrality 6設定で |SMD| $\geq$ 0.25。

**McNemar 結果** (clean 正解条件付き cond_mcnemar_p):

| 設定 | $n$ | acc(clean) | acc(LXT) | acc(MR) | cond RD | cond $p$ | 有意 |
|---|---|---|---|---|---|---|---|
| Gemma-4B x GSM8K | 1319 | 0.835 | 0.782 | 0.809 | 0.035 | 0.0021 | Yes |
| Gemma-4B x MMLU | 2850 | 0.632 | 0.586 | 0.610 | 0.055 | 8.8e-7 | Yes |
| Llama-3B x GSM8K | 1319 | 0.705 | 0.640 | 0.654 | 0.038 | 0.020 | Yes |
| Llama-3B x MMLU | 2850 | 0.627 | 0.564 | 0.590 | 0.070 | 7.9e-9 | Yes |
| Mistral x GSM8K | 1319 | 0.433 | 0.400 | 0.413 | 0.060 | 0.0042 | Yes |
| Mistral x MMLU | 2850 | 0.644 | 0.581 | 0.611 | 0.055 | 3.4e-7 | Yes |
| Llama-1B x GSM8K | 1319 | 0.361 | 0.335 | 0.337 | 0.023 | 0.41 | **No** |
| Gemma-1B x GSM8K | 1318 | 0.404 | 0.330 | 0.325 | 0.041 | 0.092 | **No** |
| Gemma-1B x MMLU-Pro | 1400 | 0.154 | 0.152 | 0.155 | 0.056 | 0.21 | **No** |

- **有意残存**: 22/25 設定 (cond $p < 0.05$)。非有意は低精度モデル x GSM8K 系と Gemma-1B x MMLU-Pro
- risk_diff (LXT の追加低下) > 0: 24/25 設定
- **Holm 確定値 (2026-07-18, m=25, `analysis/holm_correction/exp5_holm.csv`)**: raw 有意 22/25 → **Holm 有意 16/25**。Holm で落ちる6設定 = Llama-3B x GSM8K、Mistral x {CSQA, MMLU-Pro}、Gemma-1B x {ARC, CSQA}、Gemma-4B x ARC

**GLMM 最終推定 [2026-07-18]** (`analysis/glmm_final/exp5_glmm_pooled.csv`,
error ~ condition + (1|item) + (1|setting)、39,809 item × 2条件):
cond (Matched-Rnd-4) = **−0.181 ± 0.014 logit (z=−12.5, OR 0.834)**、
Intercept (LXT-4) = +0.126 ± 0.009。クラスタロバスト marginal でも
cond = −0.0912 ± 0.0098 (z=−9.3, p=1.6e-20) と符号・有意性一致。
→ **5変数マッチングで表層特性を揃えても LXT-4 の誤答オッズは約1.20倍高い**。

---

## 実験7: 校正器3段ラダー [完了]

### 目的

R2-W1 (防御未評価) への完全回答と、論点1の新結論 (ボトルネック論) の実証。rebuttal の pyspellchecker 結果を3段の校正器強度に一般化し、「より強い校正器なら復元できるかもしれない」に先回りする。

新結論: 表層修復は完璧なら十分だが、修復の失敗は答えに重要なトークン (高 $R_Q$ 語) にちょうど集中する。

**修正A適用点**: 失敗の重要語集中を LOO 重要度でも再検証 (追加コスト $\approx 0$)。

### 手法

1. typo 入り質問を3段の校正器に通す: **pyspellchecker** (非文脈型) / **T5-large-spell** (ニューラル文脈型、neuspell は導入不可のため代替) / **Qwen2.5-7B-Instruct** (LLM 校正、温度0、評価モデルと別家族)
2. 入力側指標: token 復元率 / byte-identical 率 / 誤修正率 / 失敗トークンの $R_Q$ 偏在 (Mann-Whitney)
3. 校正後の質問で全モデルに解かせ、accuracy・対 clean の flip 回復を測定
4. **3面図**: 校正器強度 x {token 復元率, accuracy 回復, 高 $R_Q$ 語への失敗集中}

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/defense/correctors.py` | 校正器3段の統一インターフェース `correct(text) -> text` |
| `src/typo_cot/defense/restoration.py` | 語単位の復元/非復元/誤修正分類 |
| `src/typo_cot/defense/analysis.py` | flip サブセット集計・$R_Q$ 偏在 Mann-Whitney |
| `src/typo_cot/defense/sharding.py` | シャード管理 |

### 予想される結果 (壁打ち事前予測)

LLM 校正で復元率 90% 超、accuracy は大幅回復するが clean に 1〜3pt 届かず。復元の失敗は依然として高 $R_Q$ の固有名・専門語に集中。

### 実際の結果 [完了: 2026-07-19]

**pyspell 段の rebuttal 再現**: ロジック差 0件。fully_restored: GSM8K 185/1319 (14.03%)、MMLU 584/2850 (20.49%)。

**スモーク結果** (n=16, GSM8K, 評価=Gemma-3-4B):
| 指標 | neural (T5) | LLM (Qwen) |
|---|---|---|
| 語復元率 | 0.871 | 0.710 |
| byte-identical | 0.563 | 0.313 |
| byte-identical flip | **0/9** | **0/5** |

**本番評価生成 (25設定×3校正器) 完了**。全表は
`docs/all_results_by_setting.md` の実験7節。代表 (GSM8K):

| 設定 | clean | LXT-4 | spellfix | neuralfix | llmfix |
|---|---|---|---|---|---|
| Gemma-4B x GSM8K | 0.835 | 0.782 | 0.762 | 0.826 | 0.819 |
| Llama-3B x GSM8K | 0.705 | 0.640 | 0.627 | 0.695 | 0.688 |
| Llama-1B x GSM8K | 0.361 | 0.335 | 0.334 | 0.360 | 0.368 |
| Mistral x GSM8K | 0.433 | 0.400 | 0.386 | 0.427 | 0.422 |
| Gemma-1B x GSM8K | 0.404 | 0.330 | 0.356 | 0.403 | 0.389 |

neuralfix (T5-large-spell) は多くの設定で clean に近い accuracy を回復。spellfix は摂動時とほぼ同等か若干改善。llmfix は neuralfix と同等。

### within-run byte-identical 検証 [完了: 2026-07-19, 75設定完走]

本番評価生成はアーカイブ baseline とのクロスラン比較で再現性ノイズが乗るため、
clean 入力と校正後入力を**同一ラン・同一バッチ** (greedy) で生成して
「byte-identical 復元 → flip 0%」を正式測定 (exp-07 worktree
`docs/dev_notes_07_correctors.md` within-run 節)。

- byte-identical 総数 **45,641 ペア** (プロンプト厳密一致; 率 spellfix 10.5〜32.5% /
  llmfix 27.8〜64.4% / neuralfix 36.6〜83.8%)
- **within-run flip 0/45,641 (0.00%)** — 全 25設定×3校正器で 0。全ペアで生成
  テキストも byte 一致、生成失敗 0、シャード失敗 0
- 同一集合のクロスラン flip (参考ノイズフロア) = **9.56% (4,362/45,641)**
  (spellfix 9.38% / neuralfix 8.80% / llmfix 10.60%)。M3×B2 部分集合では
  9.53% (1,281/13,438)
- **結論**: byte-identical 復元 → flip 0% が within-run で厳密に成立 (greedy の
  理論どおり)。クロスラン比較で byte-identical 集合に見える flip (〜9.6%) は
  全量が再現性ノイズであり、flip 事例の精査対象なし

### R_Q 偏在: 校正ボトルネックの局在 (H7-4, 宿題2完結: 2026-07-19)

本体 `analysis/exp7_tables/`(rq_mannwhitney.csv 100行 / restoration_rates.csv /
flip_rates.csv、全表は `all_results_by_setting.md` 実験7補遺)。校正後生成
`perturbed_tokens[].importance_score` を語単位 R_Q(貪欲・語順保存マッチ最大値)に
集約し、復元失敗語 vs 復元語の R_Q 分布を Mann–Whitney で比較(プール25設定, Holm m=25)。

- 校正器別 平均 word 復元率: neural **0.886** > llm 0.734 > spellfix 0.663、
  平均 flip 率: neural 0.137 < llm 0.162 < spellfix 0.227。
- R_Q 偏在: プール25設定中 Holm 有意 **17/25**、AUC=P(R_Q_failed>R_Q_restored) 平均
  **0.539**(>0.5 が 23/25)、median(failed−restored) 平均 **+0.050**。
- **H7-4 判定 = 支持(方向一貫・効果は小)**: 復元失敗は高 R_Q 側へ偏るが AUC≈0.54 と
  効果量は小さく確率的偏り。ボトルネックの主因は復元率そのもので、byte-identical
  復元 → flip 0% が残余ギャップ=補正品質起因の決定的証拠。

---

## 実験1+3: CoT 移植 2x2 + KL 発散プロファイル [完了]

### 実験1の目的

偏相関では排除できない共通原因仮説に介入 (do-操作) で答え、typo→誤答を 4 セルのリスクに分解する。**主推定量は反事実的 restore 率** = 「TE で反転した事例のうち do(CoT:=clean)(セルC)で元の答えに復帰した割合」(反事実的に一意)。副次的に「CoT を固定しても残る直接効果 (DE)」「CoT 経由の間接効果 (IE)」を**セル別リスク**として提示する。批判(1) (3名共通) への唯一の直接回答。R3 が名指しした "forcing continuation from the clean CoT under perturbed inputs" の忠実な実装。因果媒介分解の枠組みは Vig et al. (2020) のニューロンレベル causal mediation analysis を **CoT テキスト (テキスト人工物) レベル**へ移設し防御設計に接続する点で位置づけられる。

**修正A適用点 (修正Aの本体)**: LXT-4 と Random-4 の両摂動条件で全手順を実施。

### 実験1の手法

**4セル構成**: $A = (Q_c, C_c)$ 基準 / $B = (Q_p, C_p)$ 総効果 TE / $C = (Q_p, C_c)$ 直接効果 DE / $D = (Q_c, C_p)$ 間接効果 IE

CoT を答え句直前で切断して teacher-forcing し、続きの答えスパン ($\leq 16$ トークン) のみを生成。**4 セル (A/B/C/D) は同一ラン・同一バッチ (greedy) で生成する**ため、セル間比較に再現性ノイズは乗らない (within-run ノイズ = 0; 実験7 の within-run 検証で byte-identical→flip 0/45,641 として確認済み)。

$$\text{TE} = P(\text{flip}|B), \quad \text{DE} = P(\text{flip}|C), \quad \text{IE} = P(\text{flip}|D)$$

$$\text{restore} = P(\lnot\text{flip}|C,\ \text{flip}|B) = \frac{\#\{B\text{ で flip} \land C\text{ で復帰}\}}{\#\{B\text{ で flip}\}}$$

restore が主推定量。TE/DE/IE は**セル別 flip リスク**であって効果の加法分解ではない (GLMM の $Q_p{\times}C_p$ 交互作用は負 $\approx-5.9$ = サブ加法的で $\text{DE}+\text{IE}\neq\text{TE}$)。したがって **IE/TE は記述的比率**であり「媒介割合 (proportion mediated)」とは呼ばない (非加法下で Pearl/VanderWeele 型の proportion-mediated は定義されない)。事前登録に従い、構造的除外を含めた**除外込み感度分析** (`flip_rate_sensitivity`) と、CoT が実際に変化した事例に条件付けた **IE (IE|ROUGE<1)** を併記する。

GLMM: $\text{flip} \sim Q_p \times C_p + (1|\text{item})$ で4セル+交互作用を同時推定。

### 実験3の目的

typo が CoT 生成のどの時点で推論を逸らすかを挙動レベルで特定。KL 上位位置と $R_C$ 上位語の重なりを検証。faithfulness 懸念への安価な回答。

### 実験3の手法

$(Q_c, C_c)$ と $(Q_p, C_c \text{強制})$ (実験1セルCの forward を共有) の2 forward で位置別 $\text{KL}(p_{\text{clean}} \| p_{\text{pert}})$、log-prob 低下、rank 低下を計算。発散オンセット = clean 実トークンの rank が閾値を超えて落ちる最初の位置。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/records.py` | PairRecord: clean x typo のサンプル対 |
| `src/typo_cot/intervention/archive_loader.py` | アーカイブ結合 → PairRecord |
| `src/typo_cot/intervention/cell_builder.py` | 答え句直前での CoT 切断 + 4セル teacher-forcing 入力構築 |
| `src/typo_cot/intervention/runner.py` | バッチ生成・抽出・TE 照合 |
| `src/typo_cot/intervention/analysis.py` | flip 表・bootstrap CI・GLMM |
| `src/typo_cot/intervention/divergence.py` | 位置別 KL・オフセット補正・precision@k |

### 予想される結果 (壁打ち事前予測)

**パターンX**: GSM8K で DE 小 (flip 3〜8%)、IE が TE の 7〜9割。多肢選択: DE がやや大 (TE の 2〜4割)。

### 実際の結果 [完了: 2026-07-15, M5 x B5 x 2条件 = 50設定]

**TE 再現一致率**: 非除外 pooled 98.23%。生成ペア 79,618 (4セル x 答えスパン)、主分析対象 39,275。

**代表設定 (Gemma-3-4B x GSM8K)**:

| 指標 | LXT-4 (n=1042) | Random-4 (n=1050) |
|---|---|---|
| TE | 9.79% | 5.71% |
| DE | 1.34% | 1.33% |
| IE | 9.60% | 5.81% |
| IE/TE | 0.98 | 1.02 |
| headline restore rate | 93.1% | 90.0% |

**Pooled (全設定, `outcomes.json` から再集計)**:
- **restore (主推定量)**: LXT-4 = **76.2%** (GSM8K 95.8% / MC 73.0%)、Random-4 = **72.2%** (GSM8K 95.4% / MC 67.5%)
- セル別リスク (記述): LXT-4 TE=23.9% / DE=7.4% / IE=19.7% (**記述的比率** IE/TE=0.83)、Random-4 TE=17.0% / DE=6.1% / IE=13.6% (IE/TE=0.80)
- **除外込み感度分析** (`flip_rate_sensitivity`, 構造的除外も含む): LXT-4 TE=26.0% / DE=8.1% / IE=21.8% (IE/TE=0.839)、Random-4 TE=18.9% / DE=6.5% / IE=15.4% (IE/TE=0.816) — 記述的 IE/TE は主集計とほぼ不変 (0.83→0.84, 0.80→0.82) で、restore 優位は除外設計のアーティファクトではない
- **IE|ROUGE<1** (CoT が実際に変化した事例に条件付けた IE): LXT-4 = 20.5%、Random-4 = 15.2%

**主要な発見**:
- **restore 優位 (間接経路支配, IE>DE) の分解構造が両摂動条件で 48/50 設定に成立** (修正Aの見出し論理)。
  例外は **Gemma-3-1B×CommonsenseQA の両条件のみ** (LXT-4: DE 25.2% > IE 22.8% /
  Random-4: DE 22.3% > IE 18.2%) — 1B×多肢選択で DE が IE を上回る唯一の設定
- **within-run ノイズは 0**: 4 セルは同一ラン・同一バッチ生成のため、付録で報告するクロスラン flip 9.56% (アーカイブ比較固有のノイズフロア) は本 4 セル設計に適用されない。GSM8K の DE≈1.3% はノイズフロアの発現ではない (実験7 within-run: byte-identical→flip 0/45,641)
- **パターンX 方向**: GSM8K では DE $\approx$ 1〜3%、restore 90〜99%
- 多肢選択では DE がやや大 (計画の予想どおり)
- GLMM: cot_typo 効果が支配的 (Gemma-4B x GSM8K LXT: coef=4.998 vs q_typo=1.352)

設定別の全50設定表 (TE/DE/IE/IE比/restore/TE照合、mmlu は p0+p1 統合) は
`docs/all_results_by_setting.md` の実験1節。

**DE の規模×形式依存 [宿題3(b), 2026-07-19]** (`analysis/exp1_de_refinement/`、
summary.json から n 加重で再集計、両摂動条件): DE は形式で二分される。

| 規模 | 自由記述 DE/TE (DE率) | 多肢選択 DE/TE (DE率) | MC restore |
|---|---|---|---|
| 1B | 0.044 (1.4%) | 0.499 (15.4%) | 61.1% |
| 3-4B | 0.101 (1.3%) | 0.284 (4.9%) | 77.8% |
| 7B | 0.063 (1.5%) | 0.314 (5.1%) | 76.5% |

- **自由記述 (GSM8K) では DE がほぼ消失**: 全規模で DE≈1.1〜1.8%、DE/TE≈0.04〜0.11、
  restore 94〜98% (CoT 側経路がほぼ全てを担う)。
- **多肢選択では DE が残り、1B で突出**: DE/TE は 1B (≈0.50) が 3-4B/7B (≈0.28〜0.32)
  の約1.7倍。ただし「規模に単調反比例」ではなく **1B→≥3B で急落しその後は横ばい**
  (7B は 3-4B とほぼ同水準、僅かに上)。restore も 1B×MC のみ 61% と低い。
- 上記 3(a) の例外 (Gemma-1B×CSQA) はこの「1B×MC で DE 増大」の最極端例。

**DE の解釈: 選択肢テキストへの直接経路 [宿題3(c) 予備検証, 2026-07-19]**
(`analysis/exp1_de_refinement/de_choice_shortcut.py`): 1B×MC で DE が大きいのは
「摂動語が選択肢ラベル/文言に直接ヒットする表層 short-cut」ではないかを、DE-flip 群
(セルC flip) vs 非flip 群で「摂動が選択肢テキストに触れる率」を比較して検証。
**DE-flip 群で有意に高い** (1B×MC pooled n=9024: 摂動語が選択肢語に一致 62.7% vs 55.4%,
OR=1.35, Fisher p=6e-07; 選択肢文字列が摂動された率 66.3% vs 56.7%, p=9e-12)。
選択肢が短い CommonsenseQA で最も明瞭 (69.1% vs 58.6%, OR=1.58, p=5e-05)、
文選択肢の arc/mmlu_pro では弱い (共通語のベース率が高いため)。**「選択肢テキスト経路」を
支持する予備証拠**。効果量は中程度 (絶対差 +7〜10pt) で、確証ではなく方向性の裏づけ。

**GLMM 最終推定 [2026-07-18]** (`analysis/glmm_final/`、statsmodels
BinomialBayesMixedGLM 変分ベイズ; 設定別 glmm 欄と 64/64 照合 max |Δcoef|=4.7e-06):

| pool | model | Intercept | q_typo | cot_typo | q_typo:cot_typo |
|---|---|---|---|---|---|
| lxt4 | (1\|item) | −11.851 ± 0.016 | +6.458 ± 0.021 | +8.624 ± 0.019 | −5.884 ± 0.026 |
| lxt4 | +(1\|setting) | −11.734 ± 0.017 | +6.542 ± 0.021 | +8.737 ± 0.019 | −5.962 ± 0.026 |
| rnd4 | (1\|item) | −11.961 ± 0.018 | +6.163 ± 0.023 | +7.846 ± 0.020 | −5.587 ± 0.028 |
| rnd4 | +(1\|setting) | −11.751 ± 0.018 | +6.223 ± 0.023 | +7.925 ± 0.021 | −5.643 ± 0.028 |
| all | +(1\|setting) | −12.440 ± 0.012 | +7.016 ± 0.015 | +8.975 ± 0.014 | −6.445 ± 0.019 |

pooled_all の (1|item) 単独は縮退のため不採用。クラスタロバスト marginal
(cluster(item)): C セル −2.53〜−2.75 / D−C +0.90〜+1.11 / B−C +1.15〜+1.36
(全て z≫3)。**結論: DE (Cセル) は小さく CoT 側タイポ (D, B) が flip を支配、
q×cot 交互作用は負 (サブ加法的)** — 両摂動条件で同型。

**拡張グリッド [進行中: 2026-07-18 検証シャード]**: Qwen2.5-7B x GSM8K
(LXT-4: TE 3.2% / DE 0.5% / IE 3.2%, n=217; Random-4: 3.6/0.8/4.0%, n=250)、
Gemma-4B x MATH (LXT-4: TE 10.2% / DE 5.1% / IE 9.2%, n=98; Random-4:
10.7/2.7/11.6%, n=112) — **IE 優位 (DE 小) の構造が第4家族・第2自由記述でも保持**。
Qwen x MMLU は importance 側シャード生成中。

**実験3の結果**:
- 68,462 divergence プロファイル、alignment 失敗 0件
- KL 上位10% 集中: mean 93.6% (LXT) / 88.5% (Random)
- flip 群の mean_kl_sum = 10.64 vs noflip 群 6.22 (Gemma-4B x GSM8K LXT)
- precision@k (KL 上位10 vs $R_C$ 上位10): mean 0.151 (null mean 0.334)
- 設定別の全50設定表 (KL_sum flip/noflip 群、prec@10、null、onset率) は
  `docs/all_results_by_setting.md` の実験3節。**KL_sum は flip 群 > noflip 群が
  50/50 設定で成立**、prec@10 は null と同水準以下 (KL 上位位置と $R_C$ 上位語は
  空間的に相補的)

**A1 整合性監査 [2026-07-19, `analysis/a1_mistral_audit/`]**: fef3958 で判明した
Mistral アーカイブ `_cot.pt` の `word_scores` 潰れ(全結合1語)が precision@10 に
波及していないかを独立検算。precision@10 の入力 $R_C$ 上位語 (`rc_words`) は
`archive_loader` 経由で **`results.json` の `cot_top_k_words`** を読む — これは推論時
(2025-05) に空白マーカー健全なトークンで生成された**非退化**ランキング(Mistral 全設定で
最小11-28語/サンプル・最大単語長14-15文字)であり、潰れた `_cot.pt` word_scores とは
別経路。保存済み divergence 出力(tokens+kl)と `cot_top_k_words` から precision@10 を
再計算した結果、Mistral 4設定 (GSM8K/ARC/CSQA/MMLU-Pro) 全 4,787 サンプルで stored 値と
**完全一致** (再計算平均 = 記録平均 0.156/0.314/0.385/0.240)。**Mistral の precision@10 は
汚染なし・再計算不要。**

**空間相補性の前半/後半分布 (H3, 宿題5完結: 2026-07-19)** (本体
`analysis/exp3_kl_rc_spatial/`、全60設定、`all_results_by_setting.md` 実験3節):
KL 大域平均位置 **0.359**(前半)vs $R_C$ **0.541**(後半)、Mann–Whitney p≈0。
設定単位平均で 前半KL率 = importance **0.749** / random 0.606、後半$R_C$率 0.620。
相補パターン(KL前半優位 かつ $R_C$後半優位)= **48/60 設定**。Wasserstein は
importance 0.285 > random 0.168、GSM8K > MMLU(構造化算術ほど二相分離が大)。
例外は Mistral($R_C$ 後半率≈0.50 で分離小)。**H3(Complementarity)= 支持(再確認)**。

**A2 restore「自明コピー」批判への反証 [2026-07-19, `analysis/a2_restore_audit/`]**:
「GSM8K の CoT 末尾は最終数値を含むので、セル C (typo質問+clean CoT強制) の restore は
答え抽出段が CoT 末尾の数値を書き写すだけの自明な帰結ではないか」という批判を、
M3×B2 (Gemma-4B/Llama-3B/Mistral-7B × GSM8K/MMLU, Random-4) の3点セットで検証。
TE-flip 事例のみ対象。

- **(i) リーク層別** (CPU, `leak_stratification.*`): restore 事例を「強制 clean CoT prefix に
  最終答え文字列が現れるか (leak)」で層別。**GSM8K は近ユニバーサルにリーク** (最終行に金答え
  数値 757/763=99%) — (i) 単独では copy 経路を排除できない。一方 **MMLU は答え文字も選択肢
  本文も現れない no-leak が n=789/1714 でも restore 0.772 [0.74,0.80]** (最寛容 leak =
  文字 anywhere ∪ option-text を除いても n=781 で 0.772 と不変)。MC ではコピーすべき答え文字列が
  存在しないのに復元 = **非自明**。
- **(ii) 結論剥ぎ** (GPU, `conclusion_strip*`): 金答えを載せた最終計算行を除去し自由生成 (256tok)。
  GSM8K leak群 restore 0.96→0.49 に低下するが、**除去後も 0.91 が非空の答えを算出し、restore
  失敗時ですら 0.83 が(誤った)数値を出力** (例 540→180 の算術誤り)。コピー機なら空/末尾数値を
  返すはずで、モデルは末尾行を再計算している。低下幅は算術力に依存 (stripped restore: gemma 0.80 >
  mistral 0.50 > llama 0.36) であってコピーの証拠でない。(注: max_new_tokens=16 版は生成予算不足で
  空出力が多発し交絡 → `conclusion_strip_tf16/` に保存、自由生成版を主とする)
- **(iii) 回復曲線** (GPU, `recovery_curve*`): clean CoT prefix の先頭 p% のみ強制し自由生成。
  GSM8K pooled p0=0.16 / p25=0.50 / p50=0.71 / p75=0.83 / p100=0.95、MMLU 0.16/0.47/0.54/0.66/0.83。
  **金答えは CoT 末尾付近にしか出ないため p=25〜50% では強制テキストに答えが無いのに restore
  0.50〜0.71** → 存在しない答えはコピー不能で、単調な段階的復帰は再導出の直接証拠。
- **判定**: 3点いずれも自明コピー説と両立せず、**restore は答えの丸写しでなく CoT が運ぶ
  再導出可能な推論内容の媒介** を支持 → **IE 優位はフォーマットの性質でなく機構的発見**。
  スコープ: GSM8K restore の絶対水準は最終読み上げ行の存在から恩恵を受けるが、これは「完了した
  計算の再実行」であって「答えのコピー」ではない。実装は TDD (`leak_audit` 25 tests +
  `build_cell_inputs(strip_conclusion_mode=)` 2 tests)、`scripts/rebuttal/a2_{leak_stratify,gpu_audit,finalize}.py`。

---

## 実験2: 重要 CoT トークンの削除介入 [完了: core 25設定]

### 目的

内的軸が「答えを因果的に決定するトークン集合」であることの介入による支持 (R3-W2)。deletion test による AttnLRP 検証 (R3-W3)。

**修正B適用点**: M3 x B2 に top-LOO 語の削除腕を追加。

### 手法

**要因計画**: 標的 x 操作 x 用量。標的: top-$R_C$ / matched_random / bottom-$R_C$。操作: delete / mask / replace。$k \in \{1, 2, 4\}$。

**両建て構成** (スモーク結果を受けて追加): 無制限 top-$R_C$ (数値含む) vs stratum-matched random + 層別 (content/numeric 分離)。

改変 CoT を $Q_{\text{clean}}$ の下で teacher-forcing → 答え生成 → flip 判定。McNemar + リスク差 CI、用量反応の単調性検定。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/cot_editor.py` | delete / mask / replace |
| `src/typo_cot/intervention/target_selector.py` | 層判定・標的選定 |
| `src/typo_cot/intervention/deletion_runner.py` | 腕仕様 → 編集 → 短生成 → flip |
| `src/typo_cot/intervention/deletion_stats.py` | McNemar・リスク差 CI・用量反応 |
| `src/typo_cot/intervention/loo_scorer.py` | LOO スコアラ (実験6から再利用) |
| `src/typo_cot/intervention/recovery_curve.py` | 回復曲線 |

### 予想される結果 (壁打ち事前予測)

top-$R_C$ 削除の flip 率はランダム削除の 3〜5倍 (15〜30% vs 3〜6%)。用量反応は単調。

### 実際の結果 [完了: 2026-07-18, core 25設定]

**スモーク (Gemma-4B x GSM8K, n=24, core 両建て)**:
- top_rc_unrestricted delete k=4: **66.7%** vs stratum_matched_random: **4.3%**
- k=4 McNemar: RD = 0.609, CI95 = [0.391, 0.826], $p$ = 1.2e-4
- 用量反応: 無制限 top は単調 (slope 0.214, $p$ = 5e-4)、統制はフラット
- content 層: 全用量で $\approx$ 0%。numeric top k=4: **91.7%**

**本番 core 25設定 完了** (全表 = `docs/all_results_by_setting.md` 実験2節)。
k=4 の top vs 統制は **24/25 設定で有意** (非有意は Mistral x ARC のみ、RD +0.013,
p=0.31)。リスク差は Gemma-1B x GSM8K +0.80 / Gemma-4B x GSM8K +0.63 /
Llama 系 +0.43〜+0.58 / Mistral x GSM8K +0.26 など。代表:

| 設定 | n_paired | top (k=1/2/4) | 統制 (k=1/2/4) | k=4 RD [CI] | $p$ |
|---|---|---|---|---|---|
| Gemma-4B x MMLU | 436 | 10.8/15.4/20.7% | 1.5/1.5/2.1% | +0.172 [+0.135,+0.209] | 4e-19 |
| Gemma-4B x MMLU-Pro | 449 | 16.7/20.7/25.5% | 2.1/2.3/2.4% | +0.209 [+0.174,+0.252] | 3.1e-26 |
| Gemma-1B x GSM8K | 478 | 75.6/62.0/85.0% | 4.4/4.4/4.2% | +0.803 [+0.766,+0.837] | 7.8e-111 |
| Mistral x GSM8K | 471 | 2.6/19.2/35.8% | 1.6/3.4/8.1% | +0.263 [+0.214,+0.314] | 8e-22 |

**重要な発見**: GSM8K の $R_C$ 上位はほぼ数値・演算語。内容語だけの削除は flip しないが、数値語の削除は壊滅的。MMLU では content 層でも top > matched (k=4: 9.5% vs 1.7%, $p$ < 0.001)。

**Mistral 5設定の復旧 [2026-07-18]**: 当初 Mistral の n_paired が 1〜2 に崩壊して
いた根本原因は、アーカイブ Mistral の `_cot.pt` の token_scores に空白マーカー
(先頭スペース / ▁) が無く、`tokens_to_words` の語境界検出が発火せず word_scores が
全文結合の1語に潰れていたこと (標的選定が候補と交差せず全腕
insufficient_candidates)。アーカイブは読み取り専用のため読み込み側で修復:
トークン列を既知テキストへ貪欲整合して token_scores から語ランキングを再構築
(`loo_scorer.rc_word_ranking_from_token_scores`、Gemma アーカイブと小数4桁一致で
検証; commit `fef3958`)。修復後の再走で実数値を取得 (旧結果は `*_core.broken`
退避)。

---

## 実験8: activation patching [完了: M3×B2×2条件]

### 目的

DE がどの層・部位の内部表現に担われるかの局在。R1 が名指しした "activation patching or causal tracing"。Representation 層の実証。

**修正A適用点**: flip ペアを LXT-4/Random-4 で半々 (コスト増なし)。

### 手法

セルC構成で位置整列。$\text{do}(\text{第}l\text{層の残差ストリーム} := \text{clean実行の値})$。部位3種 (質問スパン / CoT suffix / 答え位置) x 方向2種 (denoising / noising) x 層窓 (幅3, stride 3)。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/patching.py` | hook 管理・位置整列・層窓スイープ・パッチ付き生成 |
| `scripts/exp8/run_patching.py` | CLI (冪等、シャード対応) |

hook 対象: Gemma-3-4B (34層) / Llama-3.2-3B (28層) / Mistral-7B (32層)。

### 予想される結果

中間層帯 (深さ40〜70%) の摂動語スパン patch で回復がピーク。

### 実際の結果 [完了: 2026-07-17 本番 + 2026-07-18 Llama 再走集計]

スモーク (2026-07-15): Gemma-3-4B x GSM8K, n=16 ペア: 11/16 完了、3除外、0失敗。1ペア $\approx$ 40s (228 forward)。

**本番 M3×B2×2摂動条件 (12条件) 完走** (Llama 2設定は HF Hub 一時障害後に
`HF_HUB_OFFLINE=1` で再走)。完了ペア: gemma gsm8k 173 / gemma mmlu 576 /
mistral gsm8k 267 / mistral mmlu 608 / llama gsm8k 337 / llama mmlu 709
(failed 0、整合性検証 PASS)。主指標 = S2 KL recovery のセル median
(question_span, clean→pert)。詳細表は `docs/all_results_by_setting.md` 実験8節と
exp-08 worktree `docs/dev_notes_08_patching.md`。

主要結果:

1. **早期層 residual 局在が3モデルで再現**: 最良セルは12条件すべて
   residual[0,12) (うち10条件が [0,6))、median 回復 lxt4 0.59〜0.77 /
   rnd4 0.33〜0.52。深さ方向に単調減衰し最終窓 ≈0。ピークは Gemma [3,9) /
   Mistral [0,6) / Llama [3,6) — 3モデルとも早期 1/3 の層。
2. mlp は早期のみ正で residual に次ぐ (0-3窓 0.31〜0.62)。attn は ≈0 かつ
   最早期 [0,3) が一貫して負 (−0.03〜−0.34)。
3. **LXT-4 は Random-4 の 4.8〜10.1 倍の分布乖離 (KL_unpatched) かつ早期層
   1窓パッチの回復率 1.4〜2.3 倍** — 重要語摂動の効果は早期層の質問スパン表現に
   集中して書き込まれる。
4. MMLU の分岐ペアでは question_span への residual[0,3) パッチで flip の
   69〜83% が逆転。GSM8K は本 regime (max_new_tokens=16 + clean CoT 強制) で
   分岐ペア 0〜4% のため flip 系指標は使わない。

→ **主結論「質問タイポの効果は早期層の摂動語スパン residual 表現に局在し、
そこへの1窓パッチで過半が打ち消せる」が 3モデル×2ベンチ×2摂動条件の全12条件で
再現** (モデル間一般化完成)。

---

## 実験9: 内部修復分析 [完了]

### 目的

R1-W2「なぜ効く摂動と効かない摂動があるのか」への機構的回答。仮説: typo が無害なのは inner lexicon が摂動語を clean 語表現へ修復できた場合。

**修正A適用点**: Random-4 語を追加 (+0.5 GPU日)。

### 手法

$$\text{repair\_score} = \max_{l \geq 1} \cos(h_{\text{clean}}^l, h_{\text{typo}}^l)$$

logit lens: 各層の表現を語彙に射影。回帰: $\text{flip} \sim \text{repair\_score} + \text{split\_increment} + \text{zipf} + R_Q + (1|\text{item})$

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/repair/span_align.py` | difflib 整列 |
| `src/typo_cot/repair/lexicon_probe.py` | 層別 hidden・cos 類似・修復スコア・logit lens |
| `src/typo_cot/repair/features.py` | 分割増分・Zipf |
| `src/typo_cot/repair/regression.py` | GLM Binomial + クラスタロバスト SE |

### 予想される結果

修復スコアが flip の最強の負予測子。

### 実際の結果 [完了: 2026-07-15〜16, 50設定]

全60シャード完了。sanity_clean_pair 全 PASS。

| モデル | repair (flip) | repair (noflip) | 差 | lens_typo | lens_clean_self |
|---|---|---|---|---|---|
| Gemma-3-4B | 0.998 | 0.998 | 微小 | 0.24-0.36 | 0.90-0.97 |
| Gemma-3-1B | 0.990 | 0.990 | 微小 | 0.14-0.25 | 0.90-0.97 |
| Llama-3.2-3B | 0.68-0.76 | 0.70-0.76 | noflip > flip | 0.44-0.63 | 0.87-0.96 |
| Llama-3.2-1B | 0.68-0.75 | 0.72-0.76 | noflip > flip | 0.28-0.43 | 0.86-0.96 |
| Mistral-7B | 0.67-0.80 | 0.77-0.80 | noflip > flip | 0.28-0.41 | **0.27-0.50** |

Gemma は修復スコアが 0.99 近傍に飽和。Llama/Mistral は差が観測可能。Mistral の lens_clean_self が異常に低い (モデル固有)。

設定別 repair 係数 (50設定、パース修正済) は `docs/all_results_by_setting.md`
実験9節: 係数負 47/50、raw 有意 25/50、Holm (m=50) 有意 10/50 (lxt4 8 / random4 2)。

**「最強の負予測子」の検証 [宿題4, 2026-07-19]** (`analysis/exp9_covariate_comparison/`、
4共変量を z 標準化して同一 GLM に投入、cluster-robust SE、clean 正解条件付き。
アンカー Llama-3B×gsm8k×lxt4 で repair −0.2445/n=3485 を再現):

| pool | repair_score | split_inc | zipf_freq | r_q | 最大\|coef\| |
|---|---|---|---|---|---|
| Llama+Mistral (主報告, n=99,698) | **−0.165** | +0.003 | −0.095 | +0.123 | repair |
| 全5モデル (n=164,190) | **−0.128** | −0.021 | −0.099 | +0.079 | repair |
| Gemma単独 (n=64,492) | **−0.387** | −0.020 | −0.118 | +0.028 | repair |
| 拡張 非Gemma (LM+Qwen+math, n=167,251) | **−0.282** | +0.005 | −0.127 | +0.120 | repair |
| 拡張 全6モデル (72設定, n=233,976) | **−0.142** | −0.047 | −0.109 | +0.109 | repair |

**判定: pooled/条件別レベルで「最強の負予測子」を支持** — 主報告・全モデル・Gemma単独・
条件別 (lxt4/random4)・Qwen/MATH 拡張後の再 pooled まで **14/14 の集計で repair_score が
最大の |標準化係数|** (z≈−13〜−33, p<1e-36, 一貫して負)。**ただし限定付き**: 個別設定では
repair が単独最大になるのは 27/72 (38%; base 20/50) にとどまり、残りは zipf_freq (19)・
split_increment (15; MATH で突出)・r_q (11; Mistral×MATH) が上回る。repair が pool で
勝つのは符号一貫性 (64/72 設定で負) による。→ **「(集計水準で) 最強の負予測子」は成立、
設定単位では最強とは限らない**旨を併記。

**拡張グリッド [進行中: 2026-07-18 検証シャード]** (exp-09 worktree
`results/exp9/summary_*.json`、sanity_clean_pair 全 PASS):

| 設定 | 条件 | n | repair (flip) | repair (noflip) | lens_typo | lens_clean_self |
|---|---|---|---|---|---|---|
| Qwen2.5-7B x GSM8K | lxt4 | 1294 | 0.791 | 0.810 | 0.098 | 0.047 |
| Qwen2.5-7B x GSM8K | random4 | 1282 | 0.795 | 0.800 | 0.143 | 0.039 |
| Gemma-4B x MATH | lxt4 | 242 | 0.998 | 0.998 | 0.271 | 0.897 |
| Gemma-4B x MATH | random4 | 243 | 0.998 | 0.998 | 0.350 | 0.916 |

noflip > flip の方向は Qwen でも保持。Gemma の repair 飽和は MATH でも再現。
Qwen は lens_clean_self が低い (Mistral と同型のモデル固有現象)。

---

## 実験6: 帰属手法比較 + LOO 再構成 [帰属比較(i–iii)完了 / LOO(iv) 本番進行中]

### 目的

AttnLRP 依存性の最終処理。(iv) LOO で帰属なしでも内的軸を再構成できることを示す (修正Bの本体)。

### 手法

**LOO 定義**: $\text{LOO}(w) = \log P(a^{\text{clean}} | Q, C, \text{trigger}) - \log P(a^{\text{clean}} | Q, C \setminus w, \text{trigger})$

主定義: 出現ごと削除 → タイプへ平均集約 (案B)。感度分析: 全出現一括削除 (案A)。先行研究調査 (Li et al. 2016) に基づく。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/intervention/loo_scorer.py` | split/extract/sequence_logprob/batched/score_sample_loo |
| `scripts/run_loo_scoring.py` | CLI (--deletion-mode {occurrence,type}) |

### 予想される結果

$\rho(J|R)$ の符号・有意性は全手法 + LOO でも保持。

### 実際の結果 [帰属比較完了: 2026-07-18〜19]

**LOO スモーク** (Gemma-3-4B x GSM8K clean, n=16):
- LOO vs $R_C$ Jaccard@10: 案B = **0.460** / 案A = 0.449
- 案B vs 案A Top-10 Jaccard: mean 0.755 (定義変更でランキングは概ね保持)
- 変種数 x2.42 (出現数/タイプ数)。本番 M3 x B2: $\approx$ 8〜10 GPU時間

**LOO 本番 完了分**: gemma-3-4b-it x GSM8K clean (occurrence, n=300):
mean LOO-vs-$R_C$ Jaccard@10 = **0.432** (median 0.429) — スモーク値をほぼ維持。
残りシャードは冪等キューで進行中。

**帰属手法ファミリー比較 (i) G×I / (ii) IG (m=16) / (iii) rollout
[完了: 36シャード = M3×B2×2条件×3手法]** (exp-06 worktree
`results/attribution_family/`、総サンプル 10,800 中スコア 10,569、エラー 0):

- 各手法 vs AttnLRP $R_C$ の mean Jaccard@10 は **0.15〜0.43** (全表は
  `docs/all_results_by_setting.md` 実験6節)
- **clean と LXT-4 でほぼ不変** (最大差 0.021) — 手法間 overlap は摂動に安定
- 最良手法はモデル依存: Gemma/Llama x GSM8K は IG、Llama x MMLU は rollout、
  Mistral は両ベンチとも G×I (LOO ≈0.43 が上限帯)

**$\rho(J|R)$ 保持表 (H6, 宿題1完結: 2026-07-19)**
(本体 `analysis/exp6_rho_preservation/`、全表は `all_results_by_setting.md` 実験6節):

- **重要な補正**: exp-06 の既存集計の "$\rho(J_\text{method}|R)$"(0.55〜0.85, 18/18有意)
  は実体が **Spearman$(J_\text{method}@10,\ \text{ROUGE-L})$** で、実験4の $\rho(J|R)$
  =「flip 目的・ROUGE-L 統制の**偏相関**」とは別統計だった。全手法を実験4/Step0 と
  同一手続き(`reproduce.py::_partial_corr`)の $\text{partial\_corr}(J@10,\text{flip}|R)$
  に揃えて再算出(Holm m=30)。
- **LOO(帰属フリー)は全6設定で負・Holm有意**($-0.18$〜$-0.44$)= 修正Bの中核を
  確証。勾配系(IG/G×I)は GSM8K で有意・MMLU で減衰非有意。**rollout は偏相関
  ほぼ0で 0/6**(Spearman は最大だが flip 予測の追加説明力なし)。
- 符号は 24 代替セル中 21 で保持。**H6 判定 = 条件付き支持**(LOO 完全保持・勾配系は
  自由記述のみ・rollout 棄却)。詳細は hypothesis_registry.md H6。

---

## 実験10: スコープ拡張 [生成完了 / 分析進行中]

### 目的

外的網羅性への直接回答。(1) Qwen2.5-7B (第4家族) (2) MATH-500 (第2自由記述) (3) R1蒸留系 (実験1・3のみ) (4) 自然 typo 条件。

### 実装

| モジュール | 役割 |
|---|---|
| `src/typo_cot/models/reasoning.py` | R1蒸留サポート (think/answer 分離、答え抽出チェーン) |
| `src/typo_cot/sharding.py` | シャード管理 |
| `scripts/run_inference_reasoning.py` | R1蒸留用シャード生成 |

### 実際の結果 [生成完了: 2026-07-18 / 分析進行中]

- **R1蒸留** (DeepSeek-R1-Distill-Qwen-7B): lxt 2.1 で $R_Q$ 計算確認。greedy/seed42/bf16、max_new_tokens 4096-8192。GPU スモーク 3/4 正解。
- **MATH-500**: アーカイブ精度 (参考): Gemma-1B 26.8% / Gemma-4B 44.4% / Llama-1B 22.2% / Llama-3B 30.0% / Mistral 12.8% / Qwen 49.8%。全面新規再生成完了 (スモーク 4/4 アーカイブと byte-identical)。

**生成完了分の accuracy** (clean / LXT-4 / Random-4; 全表は
`docs/all_results_by_setting.md` 実験10節):

- **R1蒸留**: gsm8k 0.848 / 0.811 / 0.827、math 0.734 / 0.650 / 0.610、
  mmlu 0.697 / 0.662 / 0.668
- **Qwen2.5-7B (第4家族, B5+math)**: 例 gsm8k 0.896 / 0.863 / 0.887、
  csqa 0.831 / 0.722 / 0.781、math 0.498 / 0.418 / 0.434 —
  LXT-4 の低下 ≧ Random-4 の低下が全6ベンチで成立
- **MATH-500 再生成 (M5+Qwen)**: clean はアーカイブ参考値と一致 (例 Gemma-4B
  0.444, Qwen 0.498)。LXT-4 の低下 ≧ Random-4 が 5/6 モデル (例外は Gemma-4B の
  同水準 0.368 vs 0.364)

**④ 自然typo A/B [完了: 2026-07-18]** (gemma-3-4b-it、標的語固定で合成 LXT-4 vs
GitHub Typo Corpus 分布、k=4; `analysis/exp10_natural_typo/ab_comparison.{json,md}`):

- flip率 (正→誤): gsm8k A 0.116 vs B 0.109 (McNemar p=0.526)、mmlu A 0.202 vs
  B 0.200 (p=0.914) — **flip 率・Δ精度は typo 操作分布に対して頑健**
- flip 一致は中程度 (Jaccard 0.34〜0.36): flip する個々のサンプルは分布に依存
- wave2 として master table へ取込済 (Step 0 参照)。R1/Qwen/MATH の実験1系
  分析は拡張グリッド (実験1節) で進行中

---

## Phase B (ERDC 拡張): 実験11-17 の実際の結果 [2026-07-19]

ERDC 連鎖(Encode–Repair–Divert–Carry)の**段間リンク**を閉じ、異質性を 3 モデレータ
(M1 修復・M2 R_C組成・M3 読み出し集中/ショートカット)に還元する。以下は再解析中心の Tier 1/2 完了分。
判定の正典は `hypothesis_registry.md`(H11–H17)。

### 実験11: 連鎖媒介 (H11, G→S2) [完了: 2026-07-19]

**手法**: サンプル×摂動語の repair(exp-09)、KL_sum(exp-01-03 divergence)、flip(TE flip)を
`sample_id` 結合。設定内 2 段回帰: 第1段 OLS `KL_sum ~ repair_min + Zipf + split + R_Q`、
第2段ロジット `flip ~ repair_min (+ KL_sum) + 統制`。媒介率 PM = (a−a′)/a。pooled は
`BinomialBayesMixedGLM` と設定固定効果ロジット。ソース `analysis/exp11_chain_mediation/`
(`mediation_pooled.json`、サンプル表 85,802行/60設定)。

**実際の結果**: **H11 SUPPORTED**。core5(主分析、50設定)で第1段 `KL_sum~repair_min` が
**負有意 35/50 (70%)**、pooled 媒介率 **PM = 0.577 (GLMM) / 0.578 (FE)**、設定中央値 0.523(≥0.5)。
KL_sum の flip 係数 **+0.505**。→ 修復(弱リンク)の flip への効果の約 **58%** が分岐(KL_sum)経由で、
G→S2 リンクを支持。MC タスクで最強(第1段 28/30 = 93% 負有意)。**反例(連鎖の分岐)**: MATH シャード
11設定は第1段負有意 0/11・媒介率中央値 **−0.40**(MATH では repair↑ が KL_sum を下げず、修復が分岐を
介さず読み出し段へ直接効く別経路)。感度 repair_mean で PM 0.638。Qwen は Track C dedup-on 上書きの検証扱い。

### 実験12: R_C 組成 (H12, M2) [完了: 2026-07-19]

**手法**: 各設定 clean 側 R_C top-10 を {conclusion / numeric / content / function} に分類
(POS 代理 + テンプレートマッチ、優先 conclusion>numeric>content>function)。組成シェアを
Δρ(top10, exp-04)・削除RD(exp-02 k=4)と回帰。**Mistral は必須の再構築ローダー**(word_scores 退化を
token_scores 貪欲整列で回避、31設定中 22 で発火)。ソース `analysis/exp12_rc_composition/`。

**実際の結果**: **強形 REFUTED / 機構 SUPPORTED**。R_C top-10 は内容語支配で、結論句シェアは
Gemma/Llama MC で平均 **0.130**(予測 >0.5 に対し 0/16 通過)、全31設定で \|r(結論句, Δρ)\| = **0.184**
(≥0.7 不成立)。一方 **MC 20設定に限定**すると r(結論句シェア, Δρ) = **+0.705 (p=0.0005)** で |r|≥0.7 を満たし、
削除RD も結論句シェアと **r=+0.516 (p=0.008)**(内容語+数値とは r=−0.001)。家系対比: Gemma(結論句 0.169,
Δρ +0.252, 正 92%)・Llama(0.136, +0.383, 83%)・Mistral(**0.012**, **−0.088**, 17%)。→ 答え定型は
少数派だが Δρ を弁別する軸として機能(閾値 >0.5 を誤較正として修正、M2 を方向的モデレータとして保持)。

### 実験13: 読み出し集中度 (H13, M3) [完了: 2026-07-19]

**手法**: R_C 分布の Gini(LOO 全語)を、削除RD(exp-02、content-scope 主 / all-scope 併記)と rank-corr。
10 LOO 設定 = M3 × {gsm8k, mmlu}。ソース `analysis/exp13_readout_concentration/`(`exp13_summary.json`、
`setting_table.csv`)。

**実際の結果**: **事前登録形 REFUTED / scope 一致機構 SUPPORTED**。家系 Gini 順位は
**Gemma 0.855 > Llama 0.803 > Mistral 0.790**(事前登録 Llama>Gemma>Mistral とは Gemma/Llama 逆転)。
rank-corr(LOO Gini, RD_content) = **−0.564 (p=0.09)**(scope 不一致で逆符号)。だが scope を合わせると
rank-corr(LOO Gini, RD_all) = **+0.782 (p=0.008)**(≥0.7 を満たす)。集中が numeric/機能語由来のとき
(Gemma gsm8k, Gini 0.93 だが RD_content ≈0)content 削除に効かない。**Mistral 二重乖離**を定量化: 観察的集中
(LOO Gini 0.87/0.71、内容語質量シェア)は Llama と同程度でも RD_content が桁違いに低い
(mmlu **0.026** vs Llama-3B **0.479**、gsm8k 0.005 vs 0.332)= 因果読み出しが冗長/分散(削除に強い)。

### 実験14: no-CoT ショートカット (H14, 残差DE) [完了: 2026-07-19]

**手法**: no-CoT 条件(空 CoT で即答強制)flip と DE(exp-1 セルC)の設定横断 rank-corr、サンプル OR。
設定数 72(回帰採用 n=60)。ソース `results/exp14_nocot/analysis/`(`h14_summary.json`、`settings.csv`)。

**実際の結果**: **リテラル NOT-SUPPORTED / Simpson 機構 SUPPORTED**。全設定 rank-corr(noCoT_flip, DE)
= **−0.04 (p=0.79)**(≥0.7 不成立)だが、サンプル OR(Mantel-Haenszel)= **8.85**(crude 10.12, >3)。全体 ρ≈0 は
**Simpson 型**で、層別すると **MC のみ ρ=+0.726 (p<0.001, n=40)**・**生成のみ ρ=+0.633 (n=20)**。
鋭い予測 Gemma-1B×CSQA(DE>IE の唯一設定)は全設定 rank 9/60(top25% 外)だが **MC 内 rank 2/40 (top5%)**。
noCoT_flip は IE とも連動(全 +0.578、MC +0.755)→ 「DE = 直接読み出し成分だがタスク横断の単一指標でない」。
→ 二層タクソノミー(H16 フォールバック)と整合。

### 実験17: 行動修復 (H17, M1 行動形) [完了: 2026-07-19, DeepSeek-R1-Distill-Qwen-7B]

**手法**: R1 蒸留 CoT の自己訂正マーカー(cue/明示訂正)を検出、flip との共起 OR(strict-cue / broad)。
R_Q 五分位別マーカー率。ソース `analysis/exp17_behavioral_repair/`(`raw_output.txt`)。

**実際の結果**: **H17 REFUTED(逆方向)**。事前登録は「訂正→flip 抑制(OR<1)」だったが、strict-cue OR は
**MATH 2.76 [1.76,4.33] / GSM8K 2.96 [2.12,4.14] / MMLU 1.98 [1.70,2.30]**(broad でも 2.63 / 2.25 / 1.48)、
全 task で CI が 1 を除外 → マーカーは flip と**共起**(タイポに気づき解釈を彷徨う「難儀」信号で成功修復ではない;
手動 FP 監査 17/20 が真陽性)。R_Q 単調性なし(MATH markC は五分位で平坦)、R1×MATH 逆転も importance/random で
マーカー率が等しく行動非対称なし。→ **M1 を表現レベル(実験9 隠れ状態コサイン)に一本化**、逆転は摂動トークンの
構造的性質(Track C)に帰属。

---

## 敵対的レビュー対応 (2026-07-19)

投稿前の敵対的自己レビューへの対応。測定信頼性(A)・交絡(B)・帰属法非依存(C)・防御上限(D)を機械的に検証。

### A1 Mistral 整合性監査 [`analysis/a1_mistral_audit/`]

commit `fef3958` の word_scores 潰れ(Mistral の `_cot.pt` はトークン文字列に空白マーカーが無く
`tokens_to_words` の語境界が発火せず 1 語に結合)が実験4/実験3へ波及しないかの監査。**波及なし**:
潰れは **exp-02 の word-ranking 経路(`rc_word_ranking_from_cot_pt`)に限局**。実験4 ρ_default は
`token_scores` 経路(word_scores 不使用)で、アーカイブから j_default 再計算 180/180 完全一致・
ρ_default 再計算 −0.338/−0.308/−0.380 が doc と一致。実験3 precision@10 は `results.json` の
`cot_top_k_words`(推論時算出、非退化)経路で 4787/4787 サンプル完全一致。→ **「Mistral だけ Δρ<0」
(H4 family×format)と実験3 の Mistral 異常パターン(第2主結論候補)はアーティファクトではなく無傷**。
(既存の実験4・実験1+3 節の監査段落を参照。)

### A2 restore「自明コピー」批判への反証 [`analysis/a2_restore_audit/`]

「GSM8K の CoT 末尾は最終数値を含むので restore は末尾数値の丸写しでは」への反証(3点セット)。
(i) **リーク層別**: MMLU で答え文字も選択肢本文も現れない **no-leak n=789 で restore 0.772 [0.74,0.80]**
(最寛容 leak 除外でも n=781, 0.772)→ コピーすべき答え文字列が無いのに復元 = 再導出。
(ii) **結論剥ぎ**: 最終計算行を除去し自由生成すると leak 群 restore 0.96→0.49 に低下するが、除去後も
GSM8K 0.91 / MMLU 0.99 が非空の答えを算出、restore 失敗時ですら 0.83/0.98 が(誤った)数値を出力
(例 540→180 の算術誤り)→ コピーでなく再計算。
(iii) **回復曲線**(先頭 p% 強制): GSM8K p0=0.16→**p25=0.50→p50=0.71**→p75=0.83→p100=0.95、
MMLU p0=0.16→p25=0.47→p50=0.54→p75=0.66→p100=0.83。**p=25〜50% では答え数値はまだ強制テキストに無いのに
restore 0.50〜0.71** → 存在しない答えはコピー不能。→ **「自明コピー」説を棄却、IE 優位は機構的発見**。
(既存の実験1+3 節の A2 段落を参照。)

### B 群 交絡・自明化チェック [`analysis/{b1_exp2_edit_balance,b4_exp3_entropy,b5_natural_typo_correctors,b6_choice_letter_bias}/`]

- **B1(実験2 の削除操作/位置交絡)**: R_C 選択性は**削除操作特異**(Llama-3B mmlu: top delete 0.771 →
  **replace 0.047**、matched 0.226→0.011)。ただし edit_pos+n_spans 統制後も is_top OR≫1 有意
  (content_k4 OR=10.5, p=1e-28)→ grammar 統制後も R_C 選択性は残る。
- **B4(実験3 相補性のエントロピー機械成分)**: 回避の **数値タスク 64% / MC 6%**(pooled ALL 50%)が
  実トークン surprisal で説明。ただし数値タスクは entropy 層別 null 統制後も残差回避が高度有意
  (gsm8k Wilcoxon p=3.3e-94 等)→ 相補性は弱まるが完全には崩れない。
- **B5(自然 vs 合成 typo)**: neural(T5)校正の優位は**自然 typo でも保持**(gsm8k nat +0.335 /
  mmlu nat +0.229 vs pyspell)、自然/合成の word_restoration 差はほぼ 0 → 反証成功。
- **B6(DE=セルC の選択肢文字バイアス)**: DE flip は**第1選択肢(A)へ系統偏り**(pooled 4-opt χ²=39.3,
  **p=1.5e-8**、A share 0.358 vs base 0.244;16 model×bench 中 11 で most-over=A)。効果量 Cramér V ≈0.12–0.16。

### C = 実験6 ρ(J|R) 保持(帰属比較 修正版)[`analysis/exp6_rho_preservation/`]

記法バグ(旧集計は Spearman(J@10, ROUGE-L) を ρ(J|R) と誤称)を修正し、Step0 と同一手続きの
**partial_corr(J@10, flip | ROUGE-L)** で再算出。LOO(帰属フリー)は **6/6 設定で負・Holm 有意**、
G×I 2/6、IG 3/6、**rollout 0/6**(Spearman 最大だが偏相関はほぼ0)。→ 核心(内的安定性→頑健性が特定の
帰属法のアーティファクトでない)は**帰属フリー LOO で強く支持**、勾配系は自由記述で支持・MC で減衰。
H6 は**手法・フォーマット依存の条件付き支持**(hypothesis_registry.md 参照)。

### D1 防御オラクル(重要語優先校正の上限)[`exp-20-defense` worktree]

因果地図の処方箋を手法非依存の上限として測る。摂動語の上位 k 語だけ clean に戻す oracle /
下位 k の inverse / ランダム k の random ×(k=1,2,3)、端点 k0=full-typo・k4=clean。
**データ構築完了(6設定 = M3×B2, flip サブセット × 11 条件、`results/d1_datasets/`)**: separable 保持率
0.924–0.992(gemma gsm8k 127/128・mmlu 348/364、Llama gsm8k 193/195・mmlu 424/445、Mistral gsm8k 138/147・
mmlu 352/381)。端点は「同一入力の再生成」でバイト一致(環境整合性チェック)。**生成・回復曲線は進行中**。
事前登録判定「律速=重要語復元精度」= pooled k=1 で **oracle 回復率 > random > inverse** かつ oracle vs
inverse 有意。→ 実験7 の R_Q 偏在(校正ボトルネックの局在)の含意を上限として実証する設計。

### サイズ梯子(進行中)[`exp-19-size-ladder` worktree + `analysis/size_ladder_results/`]

主分析(25–31設定)とは統計分離した確証レプリケーション層。**Gemma-3-12B × GSM8K clean acc = 92.2%**
(1216/1319、baseline 実測)。**Gemma-3-27B の R_Q 経路確定(2026-07-19)**: AttnLRP +
`--freeze_params --grad_checkpointing`(経路 a)で backward 通過を確認(feasibility: gen_peak 51.7GB /
rq_peak 61.4GB, rq_ok=true, rq_nonzero_relevance=64)。

---

## 付録: 実験間の依存関係

```
Step 0 (全実験の前提)
  +-- 実験4 (fixed-target R_C)
  |     +-- 実験2 (R_C 標的選定)
  |     +-- 実験6 (fixed-target プロトコル)
  +-- 実験5 (R_Q 参照)
  +-- 実験7 (R_Q 参照)
  +-- 実験1 (生成ログ再利用)
  |     +-- 実験3 (forward 共有)
  |     +-- 実験8 (DE 規模・flip ペア・セルC)
  +-- 実験9 (摂動語スパン・R_Q・flip)
  +-- 実験10 (並行可)
```

## 付録: 完了状態 (2026-07-19 時点)

| 実験 | ステータス | 主要成果物 |
|---|---|---|
| Step 0 | **完了 (wave2 済)** | master table 217 parquet / 342,653行 |
| 実験4 | **完了 (MATH拡張済)** | $\Delta\rho$ 25設定表 + MATH 6モデル表、Holm 19/25 |
| 実験5 | **完了** | McNemar 25設定表 (Holm 16/25)、GLMM pooled |
| 実験7 | **完了** | 25設定×3校正器 + within-run flip 0/45,641 |
| 実験1 | **完了 (拡張進行中)** | flip表 50設定、79,618ペア、GLMM pooled 確定 |
| 実験3 | **完了** | 68,462 divergence profile、50設定表 |
| 実験2 | **完了** | core 25設定 (Mistral 復旧済)、24/25 で有意 |
| 実験8 | **完了** | M3×B2×2条件 12条件、早期層 residual 局在 |
| 実験9 | **完了 (拡張進行中)** | 60シャード + Qwen/MATH 4設定、係数表 50設定 |
| 実験6 | **帰属比較完了 / LOO進行中** | G×I/IG/rollout 36シャード、LOO 本番1設定 |
| 実験10 | **生成完了 / 分析進行中** | R1/Qwen/MATH 精度表、自然typo A/B、wave2 取込 |
| 実験11 | **完了 (H11 SUPPORTED)** | 連鎖媒介 pooled PM 0.577、第1段負有意 35/50 |
| 実験12 | **完了 (H12 強形棄却/機構支持)** | R_C組成 31設定、MC r(結論句,Δρ)=+0.705 |
| 実験13 | **完了 (H13 事前形棄却/scope機構)** | Gini表 10設定、Gini vs RD_all +0.782 |
| 実験14 | **完了 (H14 リテラル不支持/Simpson機構)** | no-CoT 72設定、MC ρ=+0.726、OR=8.85 |
| 実験17 | **完了 (H17 REFUTED)** | 行動修復 OR 2-3 共起、M1 表現レベル一本化 |
| 敵対的レビュー A/B/C | **完了** | A1 監査(汚染なし)、A2 restore 反証、B1/4/5/6、C=実験6 ρ保持修正 |
| 防御 D1 | **データ構築完了 / 生成進行中** | 6設定 separable 保持率 0.92–0.99、回復曲線待ち |
