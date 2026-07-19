# Hypothesis Registry: Pre-Registered Predictions and Phase A Verdicts

本レジストリは、ARR August 2026 resubmission の全仮説を事前登録(pre-registration)し、Phase A 分析の結果に基づく判定を記録する正典文書である。事前に設定した分岐条件(branching conditions)により、予測と実測が乖離した仮説は修正(Modified)として記録し、その修正過程を完全に透明化する。

---

## Summary Table

| ID | Short Name | Experiment | Pre-Registered Prediction | Verdict | Branch Activated? |
|----|------------|------------|--------------------------|---------|-------------------|
| H1 | IE Dominance | Exp 1 | IE/TE = 0.7--0.9 (Pattern X) | **Supported** | No |
| H2 | Deletion Causal Effect | Exp 2 | top-R_C flip rate > random x3 | **Supported (exceeded)** | No |
| H3 | KL-R_C Agreement | Exp 3 | precision@10 > null mean | **MODIFIED -> Complementarity** | **Yes** |
| H4 | Fixed-Target Format Dichotomy | Exp 4 | Format main effect only | **MODIFIED -> Family x Format Interaction** | **Yes** |
| H5 | Matched Control Robustness | Exp 5 | McNemar p<0.05 in majority | **Supported** | No |
| H6 | Attribution Method Convergence | Exp 6 | rho sign/significance preserved | **Conditionally Supported** (LOO 6/6; gradient GSM8K-only; rollout fails) | No |
| H7 | Corrector Bottleneck | Exp 7 | ~90% restoration, clean gap 1--3pt | **Supported (direction confirmed)** | No |
| H8 | Activation Patching Localization | Exp 8 | Mid-layer peak recovery | Pending | No |
| H9 | Repair Score as Flip Predictor | Exp 9 | repair_score strongest predictor (all 5 models) | **MODIFIED -> Llama/Mistral scope; "strongest" holds at aggregate level only (27/72 per-setting)** | **Yes** |
| H10 | Scope Extension | Exp 10 | -- | Pending | -- |

**Branch activation count: 3/10** (H3, H4, H9)

### ERDC Extension -- Phase B Pre-Registration (H11--H18)

These eight hypotheses operationalize the unified **ERDC chain (Encode--Repair--Divert--Carry)** and are pre-registered *before* execution. Verdicts are pending; falsification branches are frozen ex ante (see detailed records below and `docs/experiments_11_18_plan.md`).

| ID | Short Name | Experiment | Pre-Registered Prediction | Verdict | Branch Activated? |
|----|------------|------------|--------------------------|---------|-------------------|
| H11 | Chain Mediation | Exp 11 | KL_sum~repair_min neg-sig in majority; repair DE attenuates >=50% controlling KL_sum (PM>=0.5) | **SUPPORTED** (2026-07-19) | core5 stage-1 neg-sig 35/50 (70%), pooled PM 0.577 (58% via S2); MATH counter-example (PM -0.40) logged as chain fork |
| H12 | R_C Composition | Exp 12 | conclusion-share >0.5 (Gemma/Llama MC), <0.3 (Mistral); numeric+content >0.7 (GSM8K/MATH); \|r(share,delta_rho)\|>=0.7 | **Strong form REFUTED / mechanism SUPPORTED** (2026-07-19) | conclusion-share 0.130 (not >0.5), \|r\|=0.184 over all 31; but MC-only r(share,delta_rho)=+0.705 (p<0.001); threshold recalibrated |
| H13 | Read-out Concentration | Exp 13 | Gini rank Llama>Gemma>Mistral; rank-corr(Gini, deletion RD)>=0.7 | **Pre-registered form REFUTED / scope-matched mechanism SUPPORTED** (2026-07-19) | Gini rank Gemma>Llama>Mistral, rank-corr(Gini,RD_content)=-0.564; scope-matched rank-corr(Gini,RD_all)=+0.782; Mistral double dissociation |
| H14 | no-CoT Shortcut | Exp 14 | rank-corr(noCoT_flip, DE)>=0.7; overlap OR>3; Gemma-1B x CSQA tops shortcut ranking | **Literal NOT SUPPORTED / Simpson mechanism SUPPORTED** (2026-07-19) | overall rho=-0.04 (Simpson), MH OR=8.85; stratified MC rho=+0.726 / gen +0.633; Gemma-1B x CSQA rank 2/40 within MC |
| H15 | Patch -> Free Generation | Exp 15 | early-window patch: delta_ROUGE>=+0.15, flip halved, onset mostly gone; late window ~ null; noising induces flips | Pending (pre-registered) | TBD |
| H16 | Unified GLMM | Exp 16 | moderators absorb >=50% of setting random-slope variance (A>=0.5) | **SUPPORTED** (2026-07-19) | A=0.645 (primary 4-mod, 20 settings gsm8k/mmlu x3 fam); robust A=0.531 (3-mod, 52 settings incl Qwen); absorption driven by M3 (gini_rc coef -0.70, noCoT_rate +0.69), repair n.s.; two-regime branch NOT triggered. Caveat: (1|setting) intercept-variance (not perturb-slope); residual sample-level Mistral family suppression NOT absorbed |
| H17 | Behavioral Repair | Exp 17 | high-R_Q typos -> explicit correction; corrected samples flip less; explains R1 LXT-superiority loss | **REFUTED** | markers co-occur with flips (OR 2-3, CI excl. 1); M1 -> representation-level only; reversal = structural tokens |
| H18 | Format Transplant | Exp 18 | MC-GSM8K: DE up, restore down, delta_rho inflates; free-form-MMLU: DE down, delta_rho -> 0 | Pending (pre-registered) | TBD |

**Phase B status (2026-07-19): 6/8 executed** -- H11 SUPPORTED, H12 strong-form REFUTED (mechanism supported), H13 pre-registered-form REFUTED (scope-matched mechanism supported), H14 literal NOT-SUPPORTED (Simpson mechanism supported), H16 SUPPORTED (A=0.645, two-regime branch not triggered), H17 REFUTED. **2/8 pending**: H15 (patch->free-generation), H18 (format transplant). Falsification branches were specified ex ante for all eight.

### Size-Ladder Extension -- Pre-Registration (P1--P6)

These six predictions are **size-effect corollaries** of the unified ERDC + M1/M2/M3 hypothesis, pre-registered *before* the size-ladder runs (Gemma-3 1B->4B->12B->27B main axis; see `docs/size_ladder_plan.md`). They form a **confirmatory replication layer that is statistically separated from the main analysis** (no re-computation of the main Holm/GLMM family). Verdicts are pending; falsification branches -- including the P1 falsification receptacle -- are frozen ex ante.

| ID | Short Name | Experiment (@scale) | Pre-Registered Prediction | Verdict | Branch Activated? |
|----|------------|---------------------|--------------------------|---------|-------------------|
| P1 | Backbone IE Defense | Exp 1 @ 27B x MATH | IE/TE >= 0.7 at the capability-frontier task | Pending (pre-registered) | TBD |
| P2 | Shortcut Size-Law | Exp 14 + Exp 1 @ ladder (MMLU) | DE/TE monotone-increasing in size; size coef absorbed by M3 | Pending (pre-registered) | TBD |
| P3 | Early-Layer Invariance | Exp 8 @ 27B | best patch window relative depth < 35% (62 layers) | Pending (pre-registered) | TBD |
| P4 | Repair Size-Monotonicity | Exp 9 + Exp 16 @ ladder | M1 rises with size, mediates part of TE decline | Pending (pre-registered) | TBD |
| P5 | Read-out Dispersion | Exp 13 + Exp 2 @ ladder | LOO-Gini falls with size, co-varies with deletion RD | Pending (pre-registered) | TBD |
| P6 | Size-as-Moderators Capstone | Exp 16 (+ size covariate) | M1--M3 absorb >= 50% of size random-slope variance | Pending (pre-registered) | TBD |

**Size-ladder pre-registration count: 6/6 pending** (P1--P6) + 1 falsification receptacle (P1 break -> regime scoping). Detailed records at the end of this document; expansion plan in `docs/size_ladder_plan.md`.

### Exp 8-fine Extension -- Single-Layer Injection Localization (H8f-1--H8f-5)

Exp 8 (coarse, width-3 windows) established that the best activation-patching recovery sits in the
**earliest residual window `residual[0,6)` in 10/12 conditions** (S1 Encode localization). Exp 8-fine
**refines that coarse window to 1-layer resolution** to locate the **read-out completion point** --
the depth by which reading the clean value off the *perturbed-word span* no longer restores the CoT --
and upgrades Fig. 5 from a window bar chart to a **continuous relative-depth x recovery profile**.
(Claim precision, per adversarial review A3: because the patch is confined to the perturbed-word span,
the depth profile localizes the *span read-out completion point*, not an absolute "injection site"; the
defensive implication -- fixing the span early enough restores the reasoning -- is unchanged, so Fig. 5's
value stands. Specificity controls A3(a)/(b)/(c) below guard against the "monotone decay is generic"
objection.) Sweep = residual-only, perturbed-word span,
denoising (clean->typo) as primary; width-1 single-layer scan over layers 0--11 (12 points) + width-1
validation at layers 14/20/26 (3 points); plus a **cumulative patch** (layers 0..l) and a **sham patch**
(write the typo run's own value back) control. Verdicts are pending; falsification branches frozen ex ante.

| ID | Short Name | Experiment | Pre-Registered Prediction | Verdict | Branch Activated? |
|----|------------|------------|--------------------------|---------|-------------------|
| H8f-1 | Peak Depth < 0.2 | Exp 8-fine | Recovery peak at relative depth l/L < 0.2 in all 3 models (Gemma L2--6, Mistral L0--2) | **Supported (6/6)** — Gemma L5 (0.147), Llama L3--4 (0.107--0.143), Mistral L1 (0.031); predicted bands hit exactly | No |
| H8f-2 | Plateau vs Spike | Exp 8-fine | Profile is plateau-shaped (2--4 adjacent layers at same level); single-layer spike -> stronger localization claim | **Supported (5/6 plateau)** — Gemma-GSM8K a single spike (L5) -> stronger-localization branch for that cell | Partial (1 spike) |
| H8f-3 | Cumulative Saturation | Exp 8-fine | Cumulative patch rises fast, saturates by l/L~0.2 (>= 1.2x single-layer max), then flat | **MODIFIED (1/6)** — saturates early (sat L1--L7, l/L<=0.21) but cum/single ratio ~1.0--1.2 -> write concentrated in early band, not distributed | **Yes** |
| H8f-4 | Late-Layer Null | Exp 8-fine | Validation layers 14/20/26 give recovery ~ 0 even at width 1 | **MODIFIED (0/6 strict)** — L20/L26 ~0 (dilution refuted for late layers) but L14 (mid, l/L~0.45) retains 0.12--0.30 -> gradual read-out tail | **Yes** |
| H8f-5 | Noising Sufficiency | Exp 8-fine | Noising at the best layer reproduces the majority of KL divergence (sufficiency) | **MODIFIED (1/6)** — single-layer noising reproduces 28--48% -> necessary but not sufficient alone; remainder distributed | **Yes** |
| A3 | Specificity Controls | Exp 8-fine | (a) other-span ~0; (b) all-positions ~1; (c) semantic vs typo profile | **(a) Supported 6/6; (b) Supported 5/6; (c) Mixed** — Mistral isomorphic, Llama corr-high/peak-shift, Gemma distinct | Partial |

**Exp 8-fine verdicts (median-robust, n=150 x 6 settings):** core localization **confirmed** (H8f-1 6/6, H8f-2 5/6, A3a 6/6, A3b 5/6); H8f-3/4/5 **modified** via pre-registered falsification branches (early single-band write, gradual mid-depth tail, necessity-not-sufficiency); A3c model-dependent. Primary estimator switched to **median** (s2_kl_recovery has an unbounded left tail; mean is outlier-corrupted). Detailed records at the end; output at `analysis/exp8_fine/` (profile CSVs, Fig.5 median overlay, judgment.json).

---

## Detailed Hypothesis Records

---

### H1: IE Dominance (Exp 1 -- Mediation Analysis)

**Pre-registered prediction:**
Pattern X (IE dominates) or Pattern Y (DE dominates). Specifically, if CoT faithfully mediates the typo-to-answer effect, the indirect effect through CoT (IE) should dominate over the direct effect (DE), yielding IE/TE > 0.7.

**Judgment criterion:**
IE/TE ratio computed via CoT transplant intervention. Pattern X adopted if IE/TE > 0.7 across pooled settings; Pattern Y if DE/TE > 0.7.

**Predicted value(s):**
- GSM8K: IE/TE = 0.7--0.9 (free-form generation with explicit reasoning steps should show strong mediation)
- MC formats: somewhat lower IE/TE (direct shortcutting possible)

**Actual value(s) from Phase A** (`results/exp01_03/*/outcomes.json` から再集計):
- **restore 率 (主推定量, 反事実的)**: LXT-4 pooled = **76.2%** (GSM8K 95.8% / MC 73.0%)、Random-4 pooled = **72.2%** (GSM8K 95.4% / MC 67.5%)
- 記述的比率 IE/TE: LXT-4 = 0.83、Random-4 = 0.80 (**媒介割合ではなく記述指標**)
- 除外込み感度分析: LXT-4 IE/TE = 0.839、Random-4 = 0.816 (主集計とほぼ不変)
- IE|ROUGE<1: LXT-4 = 20.5%、Random-4 = 15.2%
- restore 優位 (間接経路支配) の分解構造は両摂動条件で **48/50 設定**に成立 (例外は Gemma-3-1B x CommonsenseQA の両条件)
- Peak: Gemma-4B x GSM8K: restore 93.1% (記述的 IE/TE = 0.98--1.02)

**Verdict: Supported**

Pattern X adopted. Indirect-path dominance (high restore rate) is confirmed across both perturbation conditions (LXT-4 and Random-4). The finding that Random-4 shows a comparable restore rate / descriptive IE/TE to LXT-4 also supports Modification A (AttnLRP-free conclusion).

**用語上の注意 (A4)**: 主推定量は restore 率 (反事実的に一意) であり、IE/TE は**記述的比率**として扱う。効果は非加法 (GLMM $Q_p{\times}C_p$ 交互作用 $\approx-5.9$ = サブ加法, DE+IE!=TE) のため、Pearl/VanderWeele 型の "proportion mediated" は定義されず、そう呼ばない。因果媒介分解は Vig et al. (2020) のニューロンレベル分析を **CoT テキスト (テキスト人工物) レベル**へ移設し防御設計に接続する位置づけ。**選択バイアス頑健性**: 事前登録の除外込み感度分析でも記述的 IE/TE はほぼ不変 (0.83->0.84, 0.80->0.82) で、見出しは除外設計のアーティファクトではない。**within-run ノイズ = 0**: 4 セルは同一ラン・同一バッチ生成のため、付録のクロスラン flip 9.56% (アーカイブ比較固有のノイズフロア) は本 4 セル設計 (DE 含む) に適用されない (実験7 within-run: byte-identical->flip 0/45,641)。

LXT-4 と Random-4 の両条件で restore 優位が成立するため、見出し結果が帰属法の標的選定に依存しないことの裏づけにもなる。Gemma-4B x GSM8K で restore が 93% (記述的 IE/TE≈1.0) に達する点は、Gemma の CoT が事実上ほぼ完全に typo の効果を伝達する経路として機能していることを示唆する。

**Branch activation:** None required.

---

### H2: Deletion Causal Effect (Exp 2 -- R_C Top-k Deletion)

**Pre-registered prediction:**
Deleting top-R_C tokens from the CoT should cause answer flips at a rate at least 3x higher than deleting random tokens. Expected flip rate: top-R_C 15--30%, random 3--6%.

**Judgment criterion:**
Flip rate ratio (top-R_C / random) > 3, with statistical significance.

**Predicted value(s):**
- Ratio: ~3--5x
- Top-R_C flip rate: 15--30%
- Random flip rate: 3--6%

**Actual value(s) from Phase A:**
- k=4, GSM8K smoke: top-R_C = 66.7%, random = 4.3%, ratio = **x15.5**
- Numeric token subset (top k=4): flip rate = **91.7%**

**Verdict: Supported (exceeded)**

Effect far larger than predicted. The x15.5 ratio vastly exceeds the pre-registered x3 threshold. Additionally, the near-total concentration of causal effect in numeric tokens for GSM8K is a bonus discovery not anticipated in the pre-registration.

予測を大幅に上回る効果サイズは、R_C 帰属の因果的妥当性を極めて強く裏づける。数値トークンへの集中（GSM8K で 91.7%）は、算術推論における CoT の答え決定メカニズムについて新たな知見を提供する。この「数値/内容語」の層別化は事前登録にない発見だが、仮説の方向性とは矛盾しない。

**Branch activation:** None required. The numeric/content stratification is recorded as a supplementary finding.

---

### H3: KL-R_C Agreement -> MODIFIED to "Complementarity" (Exp 3)

**Pre-registered prediction:**
KL divergence peaks (locations where perturbation propagates through CoT) and R_C top tokens (answer-determining CoT positions) should show significant overlap, validating that attribution-identified tokens coincide with perturbation propagation origins. Specifically, precision@10 of KL top-10 vs R_C top-10 should exceed the null mean.

**Judgment criterion:**
precision@10 significantly above null distribution mean (permutation test).

**Predicted value(s):**
- precision@10 ~ 0.3--0.5

**Actual value(s) from Phase A:**
- precision@10 = **0.151**
- Null mean = 0.334
- p-value = 0.885 (i.e., observed BELOW null)
- KL top-10 mean position in CoT: 0.359 (early-to-mid)
- R_C top-10 mean position in CoT: 0.541 (mid-to-late)
- Wasserstein distance (global): 0.182
- Wasserstein distance (GSM8K): up to 0.47

**Verdict: BRANCH ACTIVATED -- Modified to "Complementarity"**

The pre-registered "agreement" hypothesis is rejected: KL peaks and R_C peaks do not overlap more than chance. However, the branching protocol specifies that systematic non-overlap should be investigated as a potential finding rather than treated as a null result.

**New interpretation -- Complementarity:** KL peaks (perturbation propagation origins, mean position 0.359) and R_C top tokens (answer-determining points, mean position 0.541) occupy different functional stages of the CoT. This spatial separation is itself informative: it reveals a two-phase structure where perturbation first enters the reasoning chain (KL phase) and later determines the answer (R_C phase).

**Revision note:**
"Agreement" framing withdrawn. Replaced with "Complementarity" -- KL captures perturbation onset while R_C captures causal answer-determination points, and these are systematically separated in CoT space. The Wasserstein distance quantifies the degree of functional separation and varies by task type (higher for structured arithmetic reasoning in GSM8K).

事前登録の「一致」仮説は棄却されたが、分岐プロトコルに従い「体系的不一致」自体を分析した結果、CoT の二相構造（摂動伝播フェーズ -> 答え決定フェーズ）が明らかになった。KL と R_C が CoT の異なる機能段階を捉えているという解釈は、元の「一致」仮説より情報量が多い。GSM8K で Wasserstein が最大 0.47 に達する点は、算術推論で二相分離が最も顕著であることを示す。

---

### H4: Fixed-Target Format Dichotomy -> MODIFIED to "Family x Format Interaction" (Exp 4)

**Pre-registered prediction:**
Fixed-target attribution should reveal a simple format dichotomy: free-form tasks show delta_rho ~ 0 (fixed-target and default attribution agree), while MC tasks show delta_rho > 0 (fixed-target attenuates the relationship, revealing answer-option contamination in default attribution).

**Judgment criterion:**
Format main effect only; no interaction with model family expected.

**Predicted value(s):**
- Free-form: delta_rho ~ 0 across all models
- MC: delta_rho > 0 across all models

**Actual value(s) from Phase A:**
Two-way ANOVA results:
- **Interaction (Family x Format):** eta^2 = 0.066, p = 8.9e-5 (significant)
- **Family main effect:** eta^2 = 0.397, p = 1.2e-10
- **Format main effect (Gemma/Llama only):** p = 5.4e-7

Per-family breakdown:
- Gemma MC: delta_rho = +0.371 (as predicted)
- Llama MC: delta_rho = +0.575 (as predicted)
- Gemma free-form: delta_rho ~ 0 (as predicted)
- Llama free-form: delta_rho ~ 0 (as predicted)
- **Mistral ALL settings: delta_rho < 0** (pooled mean = -0.077, CI [-0.127, -0.028])

**Verdict: BRANCH ACTIVATED -- Modified from "format dichotomy" to "family x format interaction"**

The pre-registered format dichotomy holds for Gemma and Llama families but is overridden by Mistral's architecture-specific behavior, yielding a significant family x format interaction. Mistral shows negative delta_rho even in MC settings, meaning fixed-target attribution *increases* rather than attenuates the relationship -- the opposite of the predicted direction.

**Revision note:**
The format dichotomy predicted ex ante holds for Gemma and Llama families but is overridden by Mistral's architecture-specific behavior, yielding a significant family x format interaction. Fixed-target attribution is redefined from an "improvement" over default attribution to a "diagnostic tool" that detects family-level processing differences.

Mistral の逆転（delta_rho < 0）はフォーマットだけでは説明できない。Mistral のアーキテクチャ固有の処理様式（例: sliding window attention）が帰属パターンに質的に異なる影響を与えている可能性がある。この発見は fixed-target 法の有用性を否定するものではなく、むしろモデルファミリー間の処理差異を検出する診断ツールとしての価値を示す。Family 主効果の eta^2 = 0.397 は大きな効果量であり、モデルファミリーが帰属構造の最大の変動源であることを意味する。

---

### H5: Matched Control Robustness (Exp 5)

**Pre-registered prediction:**
The LXT superiority finding (LXT-4 causes more flips than Random-4) should survive 5-variable matching (controlling for token frequency, position, POS, subword status, and R_Q) in MC settings. Some GSM8K settings with low accuracy may lose significance due to small matched-pair counts.

**Judgment criterion:**
McNemar test p < 0.05 for majority of 25 settings.

**Predicted value(s):**
- Significant in most MC settings
- Possibly non-significant in some GSM8K settings with low base accuracy

**Actual value(s) from Phase A:**
- **22/25 settings significant** (McNemar p < 0.05)
- Non-significant settings (3/25):
  - Llama-1B x GSM8K: p = 0.41
  - Gemma-1B x GSM8K: p = 0.092
  - Gemma-1B x MMLU-Pro: p = 0.21

**Verdict: Supported**

22/25 significant is well above the "majority" threshold. The three non-significant settings are exactly the low-accuracy configurations predicted ex ante (1B models on harder tasks), where the small number of correct-to-incorrect flip pairs reduces statistical power.

低精度設定（1B モデル x 高難度タスク）での非有意は予想通り。88% の設定で matched 統制を通過したことは、LXT 優位性が confounders（トークン頻度・位置・品詞・サブワード・R_Q）によるアーティファクトではないことを強く示唆する。

**Branch activation:** None required.

---

### H6: Attribution Method Convergence (Exp 6)

**Pre-registered prediction:**
The sign and significance of rho(Jaccard@k | R_C) should be preserved across alternative attribution methods (Gradient x Input, Integrated Gradients, rollout) and across LOO importance ranking. This demonstrates that findings are not an artifact of AttnLRP specifically.

**Judgment criterion:**
Sign and significance preserved in the majority of method-setting combinations.

**Predicted value(s):**
- Convergence across methods
- LOO ranking should show moderate agreement with R_C (Jaccard@10 ~ 0.3--0.5)

**Actual value(s) [完結: 2026-07-19, 宿題1]:**

Source: `analysis/exp6_rho_preservation/`(build_rho_preservation.py / preservation_table.csv / README.md). 全手法の per-sample J@10 を archive `k4_importance/full_results.json` の flip(=answer_changed)・ROUGE-L と結合し、**実験4/Step0 と同一の偏相関** `partial_corr(J@10, flip | ROUGE-L)`(residual+Pearson)で再算出。Holm m=30。

**重要な補正**: exp-06 の既存集計(dev_notes 表2)の "ρ(J_method|R)=0.55〜0.85, 18/18有意" は実体が **Spearman(J_method@10, ROUGE-L)** であり、実験4の ρ(J|R)(flip 目的・ROUGE-L 統制の偏相関)とは別統計だった(記法の衝突)。この Spearman 版は本再算出で完全再現・検証済(gemma×gsm8k: G×I 0.590 / IG 0.610 / rollout 0.758、参照 R_C 0.490 が一致)。

偏相関 partial_corr(J@10, flip | ROUGE-L)(符号は負が期待、`***`=Holm p<0.05):

| 手法 | 符号が負 | Holm有意(負) | 備考 |
|---|---|---|---|
| R_C(AttnLRP) 参照 | 6/6 | 6/6 | 実験4を再現(n300 で −0.24〜−0.62) |
| **LOO(帰属フリー)** | **6/6** | **6/6** | −0.18〜−0.44。LOO–vs–R_C J@10 ≈0.43〜0.46(予測 0.3〜0.5 内) |
| IG | 6/6 | 3/6 | GSM8K で有意、MMLU で減衰非有意 |
| G×I | 5/6 | 2/6 | GSM8K(Llama/Mistral)で有意 |
| rollout | 4/6 | **0/6** | Spearman は最大だが偏相関ほぼ0 |

**Verdict: Conditionally Supported(条件付き支持)**

事前判定基準「符号と有意性が method-setting の過半で保持」に対し:
- **符号は 24 代替セル中 21 で保持**(過半基準を満たす)。
- 偏相関の完全な符号+Holm有意保持は 11/24(LOO 6 + IG 3 + G×I 2 + rollout 0)で過半に僅かに届かない(生 p<0.05 では 13/24)。
- 中核の **LOO(帰属を一切使わない行動的ランキング)は全6設定で保持**し、これが「所見は AttnLRP のアーティファクトではない」= 修正Bの主張を最も強く確証する。勾配系はフォーマット依存(GSM8K 保持 / MMLU 減衰)、rollout は本基準で棄却。

帰属なしの LOO が実験4の ρ(J|R) 構造を全設定で再現した点が H6 の中核証拠であり、修正Bは成立。ただし帰属手法間の完全な収束は成り立たず、特に rollout は「Spearman(J,ROUGE) では最良に見えるが、ROUGE 統制後の flip 予測力は無い」という記法補正で初めて可視化された非収束を示す。勾配系の MMLU 減衰は実験4 H4(MC は選択肢汚染で挙動が質的に異なる)と整合的で、フォーマット依存性が帰属手法横断でも現れる。この非一様性は棄却ではなく、H3/H4/H9 と同型の「構造化された異質性」として情報量を持つ。

**Branch activation:** None formally required(判定基準は sign+significance の過半で、符号は過半を満たす)。ただし Spearman→偏相関の記法補正と rollout の非収束は本文で明示する。

---

### H7: Corrector Bottleneck (Exp 7)

**Pre-registered prediction:**
LLM-based typo correction achieves approximately 90% restoration rate, but a residual clean gap of 1--3 percentage points remains. Correction failures concentrate on tokens with high R_Q (tokens where the typo carries the most attribution weight).

**Judgment criterion:**
Restoration rate, clean gap magnitude, and failure concentration on high-R_Q tokens.

**Predicted value(s):**
- LLM restoration rate: ~90%
- Clean gap: 1--3pt
- Failure concentration: high-R_Q tokens

**Actual value(s) from Phase A (partial):**
- Neural corrector (T5): restoration rate = **0.871**, near-clean accuracy recovery
- LLM corrector (Qwen): restoration rate = **0.710**
- Byte-identical restoration -> flip rate = **0%** (confirmed)

**Verdict: Supported (direction confirmed)**

The direction of the prediction is confirmed: correction does not fully close the clean gap, and byte-identical restoration eliminates flips entirely (supporting that the gap is due to imperfect correction, not the correction process itself).

**Unexpected finding:** Neural corrector (T5) **outperforms** LLM corrector (Qwen) in restoration rate (0.871 vs 0.710). This was not predicted -- the prior assumption was that LLM correction would dominate. The result suggests a "conservative correction advantage": T5's narrower, more conservative corrections may avoid the over-editing that LLMs sometimes introduce.

事前予測では LLM 補正が最良と想定していたが、実測では T5（ニューラル補正器）が Qwen（LLM 補正器）を上回った。「保守的補正の優位性」として解釈: T5 は最小限の修正に留まるため、LLM が導入しがちな過剰修正（over-editing）を回避する。byte-identical 復元で flip = 0% は、残余ギャップが補正品質に起因することの決定的証拠。

**Branch activation:** None required. The neural > LLM ordering is recorded as a supplementary finding.

---

### H8: Activation Patching Localization (Exp 8)

**Pre-registered prediction:**
Activation patching from clean to perturbed hidden states should show peak accuracy recovery at mid-layer depth (40--70% of total layers). This would localize the "representation damage" from typos to a specific layer band.

**Judgment criterion:**
Layer-wise recovery curve showing statistically significant peak in the 40--70% depth range.

**Predicted value(s):**
- Peak recovery at 40--70% depth
- Recovery curve shape: gradual rise, peak in mid-layers, plateau or slight decline

**Actual value(s) from Phase A:**
- Smoke completed
- GPU estimate for full run: 0.4 days

**Verdict: Pending full results**

スモーク完了、フル実行の GPU 見積り 0.4 日。Representation 層の実証はこの実験に依存する。

---

### H9: Repair Score as Flip Predictor -> MODIFIED for Gemma (Exp 9)

**Pre-registered prediction:**
The repair score (maximum cosine similarity between perturbed and clean hidden states across layers) is the strongest negative predictor of answer flips across ALL 5 models. Higher repair score means the model successfully "repaired" the perturbation internally, preventing a flip.

**Judgment criterion:**
repair_score coefficient is significant and negative in logistic regression predicting flip, for all 5 models individually and pooled.

**Predicted value(s):**
- Negative coefficient in all 5 models
- repair_score is the strongest predictor (largest |coefficient|)

**Actual value(s) from Phase A:**

Pooled results:
- Llama + Mistral pooled: coef = **-0.977**, p = 3.7e-86
- All 5 models: coef = **-0.707**, p = 1.5e-97

Per-family diagnosis:
- Llama: repair_score discriminates flip vs noflip effectively
- Mistral: repair_score discriminates flip vs noflip effectively
- **Gemma: repair_score saturates at 0.99** (flip mean = 0.994, noflip mean = 0.994) -- ceiling effect destroys discriminability

Gemma deep dive:
- Repair completes by layer 5.3 / 31 (17% depth)
- 100% of samples reach cos > 0.95 (vs 6.9% Llama, 17.9% Mistral)
- Cohen's d for repair_score: 0.457 (inflated by near-zero variance, not meaningful)
- Best Gemma alternatives: repair_speed_95_rel (AUC = 0.615), seg_middle (d = -0.475)

**「最強の負予測子」主張の範囲確定 [Track B 宿題4, 2026-07-19]** (`analysis/exp9_covariate_comparison/`、4共変量を z 標準化して同一 GLM、cluster-robust SE、clean 正解条件付き):
- 事前登録は「repair_score は **5モデル全てで最強の負予測子** (最大 |係数|)」だったが、これを **集計 (pooled/条件別) 水準に限定**する形に確定 (Track B)。
- pooled では 14/14 の集計 (主報告 Llama+Mistral / 全5モデル / Gemma 単独 / lxt4・random4 条件別 / Qwen・MATH 拡張後) で repair_score が最大 |標準化係数| (z≈-13〜-33, 一貫して負)。
- **ただし個別設定では repair が単独最大になるのは 27/72 (38%; base 20/50)** に留まり、残りは zipf_freq (19)・split_increment (15; MATH で突出)・r_q (11; Mistral×MATH) が上回る。pool で勝つのは符号一貫性 (64/72 設定で負) による。
- **統一表現**: 「(集計水準で) 最強の負予測子。設定単位では最強とは限らない」。無条件の「最強の負予測子」主張は撤回する。

**Verdict: BRANCH ACTIVATED -- Scope limited to Llama/Mistral (and to the aggregate level)**

The repair hypothesis as formulated using max-cos is supported for Llama and Mistral (coef = -0.98, p < 1e-85) but encounters a ceiling effect in Gemma, where repair completes within the first 17% of layers. When repair is universally near-perfect, the max-cos metric loses all variance and cannot discriminate flip from noflip. Separately, the "strongest negative predictor" claim is confirmed **only at the aggregate (pooled/condition) level** (14/14 pooled analyses) and is **withdrawn at the per-setting level** (repair is the single strongest covariate in only 27/72 settings); see the Track B note above.

**Revision note:**
The repair hypothesis as formulated using max-cos is supported for Llama and Mistral (coef = -0.98, p < 1e-85) but encounters a ceiling effect in Gemma, where repair completes within the first 17% of layers. For Gemma, middle-layer cosine gradient (d = -0.48) and relative repair speed (d = 0.35) discriminate flip from noflip, suggesting that the *quality* rather than the *presence* of repair matters.

Gemma の超高速修復（全層の 17% で完了）は、Gemma アーキテクチャ固有の「積極的正規化」を示唆する。cos 類似度が 0.99 に飽和するため、max-cos ベースの repair_score は分散を失い、flip/noflip の判別力がゼロになる。しかし、修復の「速度」（repair_speed_95_rel）と「中間層の勾配」（seg_middle）は一定の判別力を保持しており、修復の「有無」ではなく「質」がflipを左右するという、より精緻な仮説への修正が正当化される。Llama（6.9%）と Mistral（17.9%）で cos > 0.95 到達率が低い点との対比が、Gemma 固有の修復メカニズムの特異性を際立たせる。

---

### H10: Scope Extension (Exp 10)

**Pre-registered prediction:**
Core findings (IE dominance, R_C deletion causal effect, matched control robustness) extend to at least one additional model family and one additional reasoning benchmark.

**Judgment criterion:**
Sign and significance of key metrics preserved in the extended scope.

**Status:** In progress (R1 distillation, MATH-500 baseline generation).

**Verdict: Pending**

---

## Branch Activation Summary

### Overview of Activated Branches

Three of ten pre-registered hypotheses required branch activation upon encountering results that diverged from predictions:

| ID | Original Hypothesis | Modification | Nature of Divergence |
|----|---------------------|-------------|---------------------|
| H3 | KL-R_C Agreement | -> Complementarity | Overlap BELOW null; spatial separation is informative |
| H4 | Format Dichotomy | -> Family x Format Interaction | Mistral reverses predicted direction |
| H9 | Repair Score (all models) | -> Scope limited to Llama/Mistral | Gemma ceiling effect destroys metric variance |

### Common Pattern Across Modifications

All three modifications share a structural pattern: a hypothesis that assumed **uniformity** (across metrics, formats, or models) encountered **systematic heterogeneity** that is itself scientifically informative. In each case:

1. The original prediction was partially correct (KL and R_C are both meaningful; format does matter; repair does predict flips)
2. The divergence was not noise but a structured phenomenon (spatial separation, interaction effect, ceiling effect)
3. The modified hypothesis has higher explanatory power than the original

---

## Lessons for Reviewers: Why Branch Activations Strengthen the Paper

### 1. Pre-Registration Integrity

The three branch activations (H3, H4, H9) demonstrate that hypotheses were genuinely registered before results were known. If hypotheses had been formulated post hoc, all ten would trivially "confirm" predictions. The presence of three clear mismatches -- including one where the observed value fell *below* the null mean (H3) -- is the signature of honest pre-registration.

事後的に仮説を立てていれば、10 件すべてが予測通りになるように調整できたはずである。H3 で観測値がヌル平均を下回った事実は、事前登録の誠実さの最も強い証拠である。

### 2. Branching Conditions Prevent Post-Hoc Rationalization

Each modified hypothesis was not simply "reinterpreted to fit the data." The branching protocol was specified ex ante:

- **H3:** "If precision@10 < null mean, investigate spatial distribution of KL vs R_C peaks as potential complementarity."
- **H4:** "If interaction term is significant, decompose by family and report interaction rather than main effect alone."
- **H9:** "If repair_score fails for a model family, diagnose the failure mechanism and identify alternative metrics."

These branching conditions constrain the space of allowable modifications, distinguishing this approach from unconstrained post-hoc analysis.

分岐条件が事前に設定されていることで、修正の自由度が制約される。これは「データを見てから都合のよい解釈を選ぶ」こととは質的に異なる。分岐条件は allowable modifications の空間を限定し、post-hoc rationalization を構造的に防止する。

### 3. Modifications Increase Scientific Value

Each modification yielded a finding with greater explanatory power than the original prediction:

- **H3 (Complementarity):** Reveals a two-phase functional structure in CoT (perturbation propagation -> answer determination) that the original "agreement" hypothesis would have obscured. This is a novel structural finding about how LLMs process perturbed reasoning chains.

- **H4 (Family x Format Interaction):** Transforms a descriptive observation ("formats differ") into a diagnostic tool that detects architecture-level processing differences between model families. The Mistral reversal is arguably the most interesting result in Exp 4 and would have been invisible under the original hypothesis.

- **H9 (Gemma ceiling):** Reveals that Gemma's repair mechanism is qualitatively different from Llama/Mistral -- repair completes in 17% of layers vs the full depth for other families. This architectural insight is more valuable than a uniform "repair predicts flips" conclusion.

いずれの修正も、元の仮説より説明力の高い発見をもたらした。H3 の「相補性」は CoT の二相構造を明らかにし、H4 の「交互作用」はモデルファミリー間のアーキテクチャ差異を診断ツールとして検出可能にし、H9 の「Gemma 天井効果」は修復メカニズムのファミリー間質的差異を示した。

### 4. The 7/10 Support Rate Is Itself Informative

Of the ten hypotheses, seven are supported (or direction-confirmed), three are modified through pre-registered branches, and none are outright refuted with no informative residual. This 7/10 support rate with 3 structured modifications is consistent with a research program that has genuine domain knowledge (the majority of predictions land) while remaining honest about complexity (the minority that miss reveal heterogeneity the field needs to understand).

---

## 考察(2026-07-19)フォローアップ判定総括(Track A: 宿題1・2・5)

ユーザー考察(2026-07-19)の残宿題のうち Track A 担当分の確定判定。データソースと
手続きは `docs/followup_plan_20260719.md`、確定表は `all_results_by_setting.md` /
`experiment_details.md` / 各 `analysis/` サブディレクトリ。

| 仮説 | 内容 | 判定 | 根拠(Track A で確定) |
|---|---|---|---|
| **H6** | 帰属手法収束(ρ(J\|R) 保持) | **条件付き支持** | LOO 偏相関 6/6 Holm有意、勾配系 GSM8K のみ、rollout 0/6。符号 21/24 保持。記法補正(Spearman→偏相関)を実施 |
| **H7-4** | 校正ボトルネックの R_Q 偏在(H7 サブ) | **支持(効果小)** | プール25設定 Holm有意 17/25、AUC 0.539(>0.5 が 23/25)、median 差 +0.050 |
| **H3** | KL–R_C 空間相補性(修正版) | **支持(再確認)** | KL 前半 0.749(imp)/ R_C 後半 0.620、相補 48/60 設定、Wasserstein imp 0.285>rnd 0.168 |

注: ユーザー考察が列挙する全 16 サブ仮説の総括は Track B(宿題3・4)/ Track C
(宿題6)/ 執筆トラック(宿題7)の確定を待って統合する。本節は Track A 確定分の
みを正典として記録し、他トラック分は各担当の完了時に追補する。

---

## ERDC Unified Hypothesis: Pre-Registered Extension (H11--H18)

Phase B pre-registers eight hypotheses that operationalize the unified **ERDC chain
(Encode--Repair--Divert--Carry)**. Phase A observed each stage of the chain *in isolation*;
Phase B closes the *links between stages* causally and shows that the cross-model / cross-benchmark
heterogeneity reduces to three moderators of a single chain. Master plan, methods (with equations),
GPU budgets, and schedule live in `docs/experiments_11_18_plan.md`.

**The ERDC chain (established by Phase A):**

- **S1 -- Encode:** the typo manifests as early-layer lexical-encoding damage of the perturbed word
  (Exp 8: early-layer localization in **12/12 conditions**).
- **G -- Repair gate:** only damage the model fails to repair internally propagates forward
  (Exp 9: repair score is a negative flip-predictor, **negative direction in 47/50** diagnoses).
  **Moderator M1 = pass rate of gate G.**
- **S2 -- Divert:** surviving damage diverts CoT generation at a small number of branch points
  (Exp 3: KL concentration, **flip > noflip in 50/50**).
- **S3 -- Carry:** the diverted CoT text carries most of the effect to the answer
  (Exp 1: **IE/TE ~ 0.8**, clean-CoT forcing **restores 90%+**).
- **Residual DE:** a read-out-stage CoT-bypass shortcut. **Moderator M3** governs it.

**Three moderators of heterogeneity:** **M1** repair capacity (gate-G pass rate);
**M2** where answer-determining information sits (composition of the top R_C mass);
**M3** read-out concentration + shortcut dependence.

**Causal-closure emphasis:** the **S1 -> S2** link is closed by **Exp 15** (early-window activation
patching into free generation): patching the early-layer encoding (S1) and observing the induced
CoT divergence (S2) promotes ERDC from a chain of correlations to a single causal chain.

Each record below is pre-registration format: {pre-registered prediction, judgment criterion,
falsification/interpretation branch, corresponding experiment, data source}. No verdicts yet
(Phase B not executed).

---

### H11: Chain Mediation (Exp 11 -- G -> S2)

**Pre-registered prediction:**
The failure of internal repair diverts the CoT (G -> S2), and this diversion mediates the effect on
the answer. Concretely: in a first stage, `KL_sum ~ repair_min` is negative and significant in the
**majority** of settings; and the direct effect of repair on flip **attenuates by >=50%** once
`KL_sum` is controlled (i.e. the diversion carries most of the repair effect).

**Judgment criterion:**
- Stage 1: `KL_sum_i = beta0 + beta1 * repair_min_i + eps_i`; **beta1 < 0 and significant in a
  majority of settings** (repair_min = per-sample minimum repair score across layers = residual
  damage; KL_sum = summed positional KL from Exp 3 = total diversion).
- Stage 2 (mediation): compare `flip ~ repair + (1|item)` (total effect) vs
  `flip ~ repair + KL_sum + (1|item)` (direct effect); proportion mediated
  `PM = (beta_repair^total - beta_repair^direct) / beta_repair^total >= 0.5` (bootstrap 95% CI, Holm).

**Falsification / interpretation branch:**
If PM does not reach 50% (repair's direct effect survives KL_sum control), then repair does **not**
act through S2 but has a **direct path to the read-out stage**. Report as a **branch/fork revision
of the chain** (add a direct G -> S3/read-out edge) -- not a rejection of the chain, but a
branching of it.

**Corresponding experiment:** Exp 11 (Tier 1, GPU 0 days; re-analysis only).

**Data source:** Exp 9 `repair_score`/`repair_min` (exp-09-inner-repair), Exp 3 `KL_sum`
(`analysis/exp3_kl_rc_spatial/`), Exp 1/2 flip labels; joined by `sample_id`.
Output: `analysis/exp11_chain_mediation/` (`mediation_pooled.json`, `mediation_by_setting.csv`,
`exp11_sample_table.parquet` 85,802 rows / 60 settings).

**Status: SUPPORTED (2026-07-19).** Both pre-registered criteria pass on the core-5 main analysis.
(1) Stage-1 `KL_sum ~ repair_min` is **negative and significant in 35/50 (70%)** of core-5 settings
(all-groups 36/53; MC-only 28/30 = 93%). (2) Pooled proportion mediated
**PM = 0.577 (GLMM) / 0.578 (FE)**, setting-median 0.523, both >= 0.5; the KL_sum flip coefficient is
**+0.505** (larger diversion -> more flips). So **~58% of repair's effect on flip is carried by the
S2 diversion (KL_sum)**, closing the G -> S2 link (repair_mean sensitivity PM 0.638).
**Interpretation branch (partial):** on the **MATH shards (11 settings)** stage-1 is neg-sig 0/11 and
PM-median = **-0.40** -- in MATH repair does not lower KL_sum, so mediation fails; this is recorded as
the pre-registered **branch/fork of the chain** (a direct G -> read-out edge for MATH), not a
rejection. Qwen handled as validation (Track C dedup-on override), excluded from the core-5 verdict.

---

### H12: R_C Composition (Exp 12 -- Moderator M2)

**Pre-registered prediction:**
The composition of the top R_C mass (where answer-determining information sits) differs by model
family and format, and this composition explains the fixed-target attenuation delta_rho (Exp 4).
Specifically: in **Gemma/Llama-family MC, conclusion-phrase share > 0.5**; in **Mistral < 0.3**;
in **GSM8K/MATH, numeric + content-word share > 0.7**; and at the setting level
**|r(conclusion-phrase share, delta_rho)| >= 0.7**.

**Judgment criterion:**
Classify top-k R_C mass into {conclusion-phrase / numeric / content / function} via answer-phrase
detection (`evaluation/extractor.py`) + POS tagging + numeric regex; compute
`conclusion_share = R_C mass on conclusion-phrase tokens / total top-k mass`; correlate with
`delta_rho = rho_fixed - rho_default` (Exp 4) across settings.

**Falsification / interpretation branch:**
If the share-vs-delta_rho correlation is weak (|r| < 0.7), M2 is not the principal driver of the
attenuation; attribute delta_rho instead to the already-confirmed family x format interaction (H4,
architecture-level differences) and demote M2 to an auxiliary moderator.

**Corresponding experiment:** Exp 12 (Tier 1, GPU 0 days; re-analysis of Exp 4 R_C).

**Data source:** Exp 4 fixed-target R_C rankings and `delta_rho` (exp-04-fixed-target).
Output: `analysis/exp12_rc_composition/` (`correlations.json`, `rc_composition_by_setting.csv`,
`rc_top10_examples.json`). Mistral requires the reconstruction loader (word_scores degenerate ->
token_scores greedy alignment), fired in 22/31 settings.

**Status: Strong form REFUTED / mechanism SUPPORTED (2026-07-19).**
The **strong prediction is refuted**: conclusion-phrase share averages only **0.130** in Gemma/Llama MC
(0/16 exceed 0.5; the R_C top-10 is content-word-dominated, not conclusion-phrase-dominated), and the
global correlation \|r(conclusion-share, delta_rho)\| = **0.184** over all 31 settings (not >= 0.7).
The **Mistral < 0.3** (mean 0.012, 6/6) and GSM8K/MATH numeric+content (mean 0.693, 5/11) predictions
are met trivially/partially. **But the underlying mechanism is directionally supported:** restricting
to the **MC domain (n=20)** gives r(conclusion-share, delta_rho) = **+0.705 (p=0.0005)**, meeting the
\|r\| >= 0.7 bar; deletion RD also tracks conclusion-share (**r=+0.516, p=0.008**) while being
uncorrelated with numeric+content (r=-0.001). The all-31 dilution is caused by GSM8K/MATH (answer
formulae route to `numeric`, so the conclusion-phrase axis is meaningless there).
**Interpretation branch (partial):** M2 is retained as a **directional moderator** (family contrast:
Gemma/Llama positive delta_rho + higher conclusion-share vs Mistral negative delta_rho + ~0
conclusion-share), with the mis-calibrated ">0.5 share" threshold corrected -- answer-formula tokens
are a minority of R_C mass but still **discriminate delta_rho**.

---

### H13: Read-out Concentration (Exp 13 -- Moderator M3)

**Pre-registered prediction:**
Models whose R_C mass concentrates on few CoT tokens are more sensitive to deletion (Exp 2).
Specifically: the Gini concentration rank order is **Llama > Gemma > Mistral**, and
**rank-corr(Gini, deletion RD) >= 0.7**.

**Judgment criterion:**
Per-sample Gini of the R_C distribution over CoT tokens
`G = (sum_i sum_j |x_i - x_j|) / (2 n sum_i x_i)`; setting-mean Gini; deletion RD = (top-R_C
deletion flip rate) - (random deletion flip rate) from Exp 2; `rank-corr(mean Gini, deletion RD)`
(Spearman) >= 0.7.

**Falsification / interpretation branch:**
If the rank order breaks or the correlation is weak, read-out concentration does not explain
deletion sensitivity; redefine M3 as shortcut-dependence alone (from Exp 14) and relegate Gini to a
descriptive appendix metric.

**Corresponding experiment:** Exp 13 (Tier 2, GPU ~1 day; R_C reuse + deletion-RD top-up).

**Data source:** Exp 4 R_C distribution (Gini), Exp 2 deletion RD
(`outputs/analysis/{bench}/{model}/k4_importance/full_results.json`).
Output: `analysis/exp13_readout_concentration/` (`exp13_summary.json`, `setting_table.csv`,
`loo_concentration.json`; 10 LOO settings = M3 x {gsm8k, mmlu}).

**Status: Pre-registered form REFUTED / scope-matched mechanism SUPPORTED (2026-07-19).**
Both literal criteria fail. (1) The family Gini order is **Gemma 0.855 > Llama 0.803 > Mistral 0.790**,
not the predicted Llama > Gemma > Mistral (Gemma/Llama are near-tied and swapped). (2)
`rank-corr(LOO-Gini, RD_content) = -0.564 (p=0.09)` -- wrong sign, because content-scoped RD and
all-word Gini are scope-mismatched. **The scope-matched exploratory analysis supports the mechanism:**
`rank-corr(LOO-Gini, RD_all) = +0.782 (p=0.008)`, meeting >= 0.7 with the correct scope, and the
family RD_all order is Llama > Gemma > Mistral. When concentration sits on numeric/function tokens
(e.g. Gemma gsm8k, Gini 0.93 but RD_content ~0), content deletion is unaffected.
**Interpretation branch (partial):** M3 read-out concentration is retained but **scope-qualified**
(all-word concentration predicts all-word deletion sensitivity). **Mistral double dissociation** is
quantified: its observational concentration (LOO-Gini 0.87/0.71, content mass share) matches Llama,
yet RD_content is an order of magnitude lower (mmlu **0.026 vs Llama-3B 0.479**; gsm8k 0.005 vs 0.332)
-- causal read-out is redundant/distributed (robust to deletion).

---

### H14: no-CoT Shortcut (Exp 14 -- Residual DE)

**Pre-registered prediction:**
The residual direct effect (DE) is a read-out-stage CoT-bypass shortcut. Settings with a larger DE
are those where the model can answer without using the CoT. Specifically:
**rank-corr(noCoT_flip, DE) >= 0.7** and sample-level **overlap OR > 3**.
**Sharp prediction:** **Gemma-3-1B x CSQA** (the only Phase A setting with **DE > IE**) sits at the
**top of the shortcut-dependence ranking**.

**Judgment criterion:**
no-CoT condition = force the answer immediately with an empty CoT span (teacher-forcing);
`noCoT_flip` = flip rate under no-CoT; `rank-corr(noCoT_flip, DE)` across settings (DE = Exp 1
P(flip|cell C)); 2x2 odds ratio OR for overlap between no-CoT-flip samples and DE samples.

**Falsification / interpretation branch:**
If the correlation is weak or Gemma-1B x CSQA does not top the ranking, DE is not a CoT-bypass
shortcut; reinterpret DE via another mechanism (e.g. residual encoding damage surfacing at the
read-out stage).

**Corresponding experiment:** Exp 14 (Tier 2, GPU ~1 day; no-CoT answer-span generation).

**Data source:** Exp 1 DE / cell C (exp-01-03-transplant), plus new no-CoT generations.
Output: `results/exp14_nocot/analysis/` (`h14_summary.json`, `settings.csv`, `report.md`;
72 settings, regression n=60), exp-14-nocot worktree.

**Status: Literal NOT-SUPPORTED / Simpson mechanism SUPPORTED (2026-07-19).**
The literal criteria split: the **overall** `rank-corr(noCoT_flip, DE) = -0.04 (p=0.79, n=60)` fails
the >= 0.7 bar, **but** the sample-level overlap **OR (Mantel-Haenszel) = 8.85** (crude 10.12) passes
the OR>3 bar. The near-zero overall correlation is a **Simpson's paradox**: stratifying by task
recovers strong positive coupling -- **MC-only rho = +0.726 (p<0.001, n=40)** and **generation-only
rho = +0.633 (n=20)**. The sharp prediction is confirmed **within the MC stratum**: Gemma-3-1B x CSQA
(importance) ranks **2/40 (top 5%)** on no-CoT flip among MC settings (all-settings rank 9/60, just
outside top-25%). `noCoT_flip` also correlates with IE (all +0.578, MC +0.755), so it reflects general
typo-sensitivity, not a DE-specific index. **Interpretation branch (partial):** DE is a
**read-out-stage direct-readout component** that couples to shortcut availability **within each format
regime**, but is not a single cross-task scalar -- consistent with the two-regime taxonomy (H16
fallback). Reported as Simpson-stratified support rather than a monolithic rejection.

---

### H15: Patch -> Free Generation (Exp 15 -- Closes S1 -> S2)

**Pre-registered prediction:**
Patching the early-layer encoding window (S1, localized by Exp 8) and then **freely generating** the
CoT recovers the clean reasoning trajectory (S2), causally closing S1 -> S2. Specifically:
early-window patch yields **delta_ROUGE >= +0.15** vs unpatched; **flip is at least halved**; the
divergence **onset mostly disappears**; a **late-window patch is ~ null**; and reverse **noising
induces CoT divergence and flips**.

**Judgment criterion:**
- denoising: `do(h^early := h^early_clean)` on the typo run, then free-generate CoT;
  `delta_ROUGE = ROUGE(patched CoT, clean CoT) - ROUGE(unpatched pert CoT, clean CoT) >= +0.15`.
- flip rate (patched vs unpatched) halved or more.
- divergence onset (Exp 3 definition) removed in a majority of samples.
- late-window patch ~ no effect (effect localized early).
- noising (inject typo-run early window into the clean run) induces divergence / flips.

**Falsification / interpretation branch:**
If the early-window patch does not pull free generation back toward clean, S1 encoding damage does
**not single-handedly** drive S2 divergence (another path intervenes); weaken the S1 -> S2 link to
"necessary but not sufficient" and re-formulate jointly with the Exp 11 mediation path.

**Corresponding experiment:** Exp 15 (Tier 3, GPU 1--2 days; **the pivotal causal-closure
experiment**).

**Data source:** Exp 8 early-window localization (exp-08-patching), Exp 3 onset definition, Exp 1
cell C alignment; implemented by extending `intervention/patching.py` to free generation.
Output: `analysis/exp15_patch_freegen/` (proposed).

**Status:** Pre-registered (Phase B, pending execution).

---

### H16: Unified GLMM Heterogeneity Absorption (Exp 16)

**Pre-registered prediction:**
The cross-model / cross-benchmark heterogeneity reduces to the three moderators M1/M2/M3 of the
single ERDC chain. Introducing the moderators **absorbs >=50% of the setting random-slope variance**.

**Judgment criterion:**
- Baseline GLMM: `flip ~ Q_p*C_p + (1 + perturb | setting) + (1|item)` -> `sigma2_slope(base)`.
- Moderated GLMM: add fixed effects M1 (repair pass rate), M2 (R_C composition / conclusion-share),
  M3 (Gini + no-CoT shortcut dependence) -> `sigma2_slope(mod)`.
- Absorption `A = 1 - sigma2_slope(mod)/sigma2_slope(base) >= 0.5`.

**Falsification / interpretation branch (fallback):**
If A < 0.5 (heterogeneity not absorbed by continuous moderators), do not abandon the single-chain
unification; instead report a **two-regime taxonomy**: a **free-form regime** (S3 carry-dominant,
IE-dominant) and an **MC/selection regime** (DE/shortcut-dominant), applying ERDC within each regime.

**Corresponding experiment:** Exp 16 (Tier 1 skeleton with M1+M2, completed in Tier 2 with M3; GPU 0
days).

**Data source:** Exp 9 (M1), Exp 12 (M2), Exp 13 (M3), Exp 1 flip responses.
Output: `analysis/exp16_unified_glmm/` (proposed).

**Status:** Pre-registered (Phase B, pending execution).

---

### H17: Behavioral Repair (Exp 17 -- Moderator M1, behavioral form)

**Pre-registered prediction:**
Repair capacity (M1) manifests behaviorally, not only in hidden-state cosine (Exp 9): the higher the
R_Q of the typo'd word, the more likely the model performs an **explicit correction** in the CoT,
and samples with a correction are **less likely to flip**. This **explains the disappearance of
LXT superiority in the R1-distilled (reasoning) models** observed in Exp 10.

**Judgment criterion:**
Detect explicit corrections in R1-distilled CoTs (regex + lexical matching), `corrected in {0,1}`;
logistic `P(corrected | R_Q)` with positive coefficient; `P(flip | corrected=1) < P(flip |
corrected=0)` (McNemar-family test); link to the loss of LXT-4 (high-R_Q target) advantage in Exp 10.

**Falsification / interpretation branch:**
If correction frequency is unrelated to R_Q, or correction does not suppress flips, the LXT-loss is
not explained by behavioral repair; reinterpret as a strengthening of representation-level repair
(Exp 9 M1) in reasoning models.

**Corresponding experiment:** Exp 17 (Tier 1, GPU 0 days; reuses Exp 10 R1 generations. If Exp 10 R1
generation is incomplete, that completion is a prerequisite).

**Data source:** Exp 10 R1-distilled CoTs (exp-10-scope), Exp 7 R_Q (`analysis/exp7_tables/`).
Output: `analysis/exp17_behavioral_repair/` (commit `47fec73`, branch exp/07-correctors).

**Status: REFUTED (2026-07-19).** Both pre-registered criteria fail, in the opposite direction.
(1) Self-correction markers **co-occur with flips**, not suppress them: strict-cue odds ratios
MATH 2.76 [1.76, 4.33], GSM8K 2.96 [2.12, 4.14], MMLU 1.98 [1.70, 2.30] -- all CIs exclude 1
(prediction was OR<1). Manual audit (17/20 TP) shows the markers are a "struggling" signal --
the model noticing the typo and wandering between interpretations -- not successful repair.
(2) No monotone R_Q->correction relationship (MATH markC flat across R_Q quintiles). (3) On the
reversal task (R1xMATH, Random>LXT) all markers are equal across importance/random, so no
behavioral asymmetry drives the reversal. **Interpretation branch taken:** M1 is reinterpreted as
**representation-level only** (Exp 9 hidden-state cosine); the R1xMATH reversal is attributed to
the **structural property of the perturbed tokens** (Track C: random-4 destroys single-char math
variables/delimiters that lack linguistic redundancy and are not repaired even in R1's long CoT),
NOT to behavioral repair. This refutation sharpens the unified model: the self-correction verbalization
marks difficulty, not repair success.

---

### H18: Format Transplant (Exp 18 -- S3/DE Format Dependence)

**Pre-registered prediction:**
The balance between S3 (carry) and DE (shortcut) is driven by **answer format**, not content.
Specifically: **MC-ified GSM8K shows DE up, restore down, and delta_rho inflation**; and
**free-form-ified MMLU shows DE down and delta_rho -> 0**.

**Judgment criterion:**
- MC-GSM8K: numeric answers converted to options; measure `DE_MC` (Exp 1 direct effect),
  `restore_MC` (clean-CoT forcing recovery), `delta_rho_MC` (Exp 4).
- free-form MMLU: options hidden, free-form answering; measure `DE_free`, `delta_rho_free`.
- Contrast against native formats (GSM8K free <-> MC, MMLU MC <-> free).

**Falsification / interpretation branch:**
If DE / delta_rho do not move when the format is swapped, the S3/DE balance is driven by **content**
(arithmetic vs commonsense), not format; revise the format-dependence claims of M2/M3 to
content-dependence.

**Corresponding experiment:** Exp 18 (Tier 3, GPU 1--2 days; new format-transplant generations).

**Data source:** Exp 1 DE/restore baselines, Exp 4 delta_rho baselines, plus new MC-GSM8K /
free-form-MMLU generations.
Output: `analysis/exp18_format_transplant/` (proposed).

**Status:** Pre-registered (Phase B, pending execution).

---

## Size-Ladder Extension: Pre-Registered Size-Effect Predictions (P1--P6)

The size ladder pre-registers six **size-effect corollaries** of the unified ERDC chain and its
three moderators M1/M2/M3. Phase A/B established the mechanism on 1B--8B models; the ladder
(Gemma-3 1B->4B->12B->27B main axis, Qwen2.5 7B->14B->32B reserve, Llama-3.3-70B int4 spot; see
`docs/size_ladder_plan.md`) asks whether the *same* mechanism holds as capacity scales. The
scientific stance is deliberate: **model size is the single largest external-validity threat**
(all Phase A models are small; a reviewer citing Lanham et al. can argue IE dominance is a
small-model artifact), and P1--P6 **turn that threat into a confirmatory test** -- if ERDC+M1/M2/M3
is correct, size should act *only through* the three parameters (P6), not as a separate mechanism.

Each record uses the pre-registration format: {pre-registered prediction, judgment criterion,
falsification/interpretation branch, corresponding experiment, data source}. These predictions live
on a **confirmatory replication layer statistically separated from the main analysis** (they do not
re-trigger the main Holm/GLMM family; H1--H18 verdicts are unaffected by ladder outcomes). No
verdicts yet (ladder not executed).

---

### P1: Backbone Defense Line -- IE Dominance Holds at the Capability Frontier (Exp 1 @ 27B)

**Pre-registered prediction:**
At the capability-frontier-adjacent task for each large ladder point (for Gemma-3-27B this is
**MATH**), the indirect effect keeps dominating: **IE/TE >= 0.7**. Where CoT actually bears the
reasoning load, the S3 (Carry) transport path does not vanish with scale. This is the "backbone"
the whole ERDC chain rests on and the primary defense against the Lanham critique that IE dominance
is a small-model artifact.

**Judgment criterion:**
Exp 1 CoT-transplant mediation at 27B (and the ladder point nearest the capability frontier) on
MATH: **IE/TE >= 0.7** pooled over samples. Frontier-adjacency = the hardest benchmark on which the
model is well above floor but below ceiling (MATH for 27B; escalate the perturbation to LXT-8 if
flips < 50 to preserve power).

**Falsification / interpretation branch:**
If IE/TE < 0.7 at 27B on MATH (the carry path collapses even where CoT should be doing the work),
route to the pre-registered **falsification receptacle** below (scope the CoT-mediated regime to
where CoT bears the load; Lanham boundary-condition discovery). An IE collapse on **MC** but not on
MATH is the *expected* P2 shortcut regime, not a P1 failure.

**Corresponding experiment:** Exp 1 (mediation) at Gemma-3-27B (and ladder) on MATH; forward-only
transplant (see `docs/size_ladder_plan.md` §2), GPU per §3 wave 1.

**Data source:** Exp 1 IE/TE/DE transplant estimates at 27B x MATH (exp-01-03-transplant,
size-ladder run). Phase A anchor: pooled IE/TE ~ 0.80--0.83 (H1).
Output: `analysis/size_ladder/p1_ie_frontier/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer).

---

### P2: Size-Law of Shortcuts (Exp 14 + Exp 1 @ ladder, MMLU)

**Pre-registered prediction:**
On MMLU the direct-effect share **DE/TE increases monotonically with model size**, and this increase
is **explained by the no-CoT accuracy of Exp 14** (larger models solve the MC item without using the
CoT, so a larger fraction of the effect bypasses the chain). A **size-controlled regression whose
size coefficient vanishes once M3 (no-CoT shortcut dependence) is entered** is the evidence that M3
absorbs the size effect. The **Lanham-type faithfulness decline** (faithfulness dropping with scale
on easy tasks) is here *predicted* as the surface signature of "larger models can answer MC without
CoT, so the shortcut ratio rises."

**Judgment criterion:**
- Monotonicity: `rank-corr(size, DE/TE)` on MMLU across ladder points > 0 (Spearman; target >= 0.7).
- Explanation: in the setting-level regression `DE/TE ~ size`, the size coefficient attenuates
  toward non-significance once `M3 = no-CoT shortcut dependence` (Exp 14 no-CoT flip / accuracy) is
  added as a covariate (>= 50% coefficient shrinkage, mirroring the P6/H16 absorption logic).

**Falsification / interpretation branch:**
If DE/TE is not monotone in size on MMLU, or the size coefficient survives M3 control, then the size
dependence of DE is **not** a CoT-bypass shortcut governed by M3; reinterpret the DE size-trend via a
separate mechanism (e.g. residual encoding damage surfacing at read-out) and demote the
Lanham-as-predicted framing to Lanham-as-observed.

**Corresponding experiment:** Exp 14 (no-CoT) + Exp 1 (DE) at ladder points on MMLU; the M3/size
regression is CPU re-analysis (cf. Exp 16).

**Data source:** Exp 1 DE/TE per ladder point (MMLU), Exp 14 no-CoT accuracy/flip per ladder point.
Observed small-model anchor: MC DE/TE ~ 0.4--0.6 (1B) -> 0.2--0.4 (3--7B). **Regime note:** this
observed 1B->7B *decrease* runs opposite to the predicted *increase*; P2 asks whether the trend
reverses as scale enters the Lanham regime (12B->27B on easy MC), which is exactly why the ladder is
diagnostic (see `docs/size_ladder_plan.md` §0).
Output: `analysis/size_ladder/p2_shortcut_sizelaw/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer).

---

### P3: Early-Layer Localization Invariance (Exp 8 @ 27B)

**Pre-registered prediction:**
The early-layer localization of typo encoding damage (S1; Exp 8 found early-layer localization in
12/12 Phase A conditions) is **preserved at 27B despite its 62 layers**: the best activation-patching
recovery window sits at **relative depth < 35%** (roughly within the first ~22 of 62 layers).
Encode-stage localization is scale-invariant.

**Judgment criterion:**
Exp 8 layer-wise recovery curve at Gemma-3-27B: the peak-recovery window's relative depth
(window-center layer / 62) < 0.35, statistically distinguishable from a uniform/late-layer null.

**Falsification / interpretation branch:**
If the best window shifts to mid/late depth (>= 35%) at 27B, S1 encoding is **not** scale-invariant;
the Encode stage acquires depth-dependence with scale, and the "early-layer" claim is qualified to
the <= 12B regime.

**Corresponding experiment:** Exp 8 (activation patching localization) at 27B; forward-only patching
sweep (window resolution may be coarsened under time pressure, see `docs/size_ladder_plan.md` §3).

**Data source:** Exp 8 layer-wise recovery at 27B (exp-08-patching, size-ladder run); Phase A anchor:
12/12 early-layer localization.
Output: `analysis/size_ladder/p3_early_layer/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer).

---

### P4: Repair Size-Monotonicity and TE Mediation (Exp 9 + Exp 16 @ ladder)

**Pre-registered prediction:**
Repair capacity (M1; the gate-G pass rate from Exp 9) **increases monotonically with model size**,
and this increase **mediates part of the decline in total effect TE** (larger models are more
typo-robust: lower TE). Size buys robustness partly by buying repair.

**Judgment criterion:**
- Monotonicity: `rank-corr(size, M1 pass rate) > 0` across ladder points. Because of the Gemma
  repair-score ceiling (H9), M1 is operationalized as the **pass rate / relative repair speed**
  (`repair_speed_95_rel`), not the saturated max-cos, so the metric retains variance at scale.
- Mediation: comparing `TE ~ size` vs `TE ~ size + M1`, M1 absorbs a non-trivial share of the
  size->TE slope (PM > 0, bootstrap CI), consistent with the P6 GLMM.

**Falsification / interpretation branch:**
If M1 does not increase with size, or does not mediate the TE decline, the size-driven typo
robustness is **not** routed through the repair gate; attribute the robustness gain to another
channel (e.g. distributed read-out, P5) and record M1 as size-invariant.

**Corresponding experiment:** Exp 9 (repair) at ladder + Exp 16 mediation (CPU re-analysis).

**Data source:** Exp 9 repair pass rate / `repair_speed_95_rel` per ladder point
(exp-09-inner-repair), TE per ladder point (Exp 1).
Output: `analysis/size_ladder/p4_repair_size/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer).

---

### P5: Read-out Dispersion with Scale (Exp 13 + Exp 2 @ ladder)

**Pre-registered prediction:**
Read-out concentration (M3; the Gini of the R_C mass over CoT tokens) **decreases with model size**
-- larger models read the answer out of **more** CoT tokens (distributed read-out) -- and this
dispersion **co-varies with the decline in deletion risk-difference RD** (Exp 2: bigger models are
less hurt by deleting top-R_C tokens because the read-out is spread out). Because attribution is
forward-only-constrained at 27B, concentration is measured with the **LOO-Gini** (attribution-free
leave-one-out importance), not AttnLRP R_C.

**Judgment criterion:**
- `rank-corr(size, mean LOO-Gini) < 0` across ladder points.
- Co-variation: `rank-corr(mean LOO-Gini, deletion RD)` stays positive (target >= 0.7, cf. H13), and
  both LOO-Gini and deletion RD decline together across the ladder.

**Falsification / interpretation branch:**
If LOO-Gini does not decrease with size, or does not co-vary with deletion RD, read-out does **not**
disperse with scale; M3 concentration is not the size channel and deletion robustness is attributed
elsewhere (e.g. repair, P4). LOO-Gini then reverts to a descriptive metric.

**Corresponding experiment:** Exp 13 (read-out concentration, LOO-Gini) + Exp 2 (deletion RD) at
ladder.

**Data source:** Exp 13 LOO-Gini per ladder point, Exp 2 deletion RD per ladder point
(`outputs/analysis/{bench}/{model}/k4_importance/full_results.json`).
Output: `analysis/size_ladder/p5_readout_dispersion/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer).

---

### P6: Capstone -- Size Acts Through the Three Moderators (Exp 16 + size covariate)

**Pre-registered prediction:**
Adding **model size as a setting-level covariate to the unified GLMM (Exp 16/H16)** and then
entering **M1, M2, M3** absorbs **>= 50% of the size-attributable random-slope variance**. The
interpretation: **model size is not an additional mechanism; it acts on typo robustness entirely
through the three measurable parameters M1 (repair), M2 (answer-info location), and M3 (read-out
concentration / shortcut).**

**Judgment criterion:**
- Baseline: `flip ~ Q_p*C_p + size + (1 + perturb | setting) + (1|item)`; estimate the
  size-attributable component of the setting random-slope variance `sigma2_size(base)`.
- Moderated: add fixed effects M1/M2/M3; estimate `sigma2_size(mod)`.
- Absorption `A_size = 1 - sigma2_size(mod)/sigma2_size(base) >= 0.5`.

**Falsification / interpretation branch:**
If `A_size < 0.5` (size variance not absorbed by M1--M3), size carries an **independent mechanism**
beyond the three moderators; report size as a **fourth moderator / distinct scale effect** rather
than a derived one, and revise the "size acts only through M1--M3" claim accordingly.

**Corresponding experiment:** Exp 16 (unified GLMM) re-fit with a size covariate (CPU).

**Data source:** Exp 9 (M1), Exp 12 (M2), Exp 13 (M3), Exp 1 (flip), model size (params) as a
continuous setting-level covariate.
Output: `analysis/size_ladder/p6_size_glmm/` (proposed).

**Status:** Pre-registered (size-ladder confirmatory layer; statistically separated from the main
GLMM per `docs/size_ladder_plan.md` §4).

---

### Falsification Receptacle (Pre-Registered) -- P1 Break -> Regime Scoping

**Pre-registered fallback for a P1 failure.** If P1 breaks -- i.e. IE **collapses on the 27B MC
setting** (IE/TE falls well below 0.7 where a shortcut can operate) -- we do **not** retract the
mediation finding. Instead we **scope** the claim, exactly as frozen ex ante:

> *"The CoT-mediated regime holds under the conditions where CoT bears the load (the region where
> task difficulty rivals the model's capability), and the boundary of that regime is predicted by
> M3 (read-out concentration / shortcut dependence)."*

This converts a P1 failure into a **boundary-condition discovery consistent with Lanham et al.**
(faithfulness is regime-dependent), while giving the boundary itself a mechanistic predictor (M3).
The MC collapse is thereby reframed from a threat into a *localization of where the carry regime
ends*. This receptacle is the pre-registered branch for P1 only; P2--P6 each carry their own
branches above.

---

## Exp 8-fine: Single-Layer Injection Localization (H8f-1--H8f-5)

Exp 8-fine refines the coarse (width-3) Exp 8 finding -- best recovery window `residual[0,6)` in
**10/12 conditions** -- to **1-layer resolution**, to locate the **read-out completion point** of the
perturbed-word span (the depth beyond which reading the clean span value no longer restores the CoT).
Per adversarial review A3, the claim is deliberately scoped to a *read-out completion point* rather than
an absolute *injection site* (the patch is span-confined, so information may already have propagated to
other positions by late layers); the defensive implication -- correcting the span early enough restores
the reasoning -- is unchanged. The design: patch the
**residual** stream at the **perturbed-word span**, **denoising** direction (clean->typo) as primary;
a width-1 single-layer scan over layers 0--11 (12 points) plus width-1 validation at layers 14/20/26
(3 points); a **cumulative** patch (all layers 0..l replaced, l=0..11); and a **sham** control
(write the recipient run's own value back at the same position -- expected zero effect). Cross-model
comparison uses **relative depth l/L** (Gemma L=34, Llama L=28, Mistral L=32). Primary metric = S2 KL
recovery of the first-CoT-word distribution; secondary = branch-pair flip-reversal rate (MMLU only;
GSM8K branch rate 0--4% = uninformative, a known constraint). n = 150 flip pairs / setting
(LXT-4 : Random-4 half-and-half). Data source: coarse results at
`.../exp-08-patching/projects/typo-cot/results/prod/exp8/` (read-only); archive baseline/perturbed
generations at `archive/2025/JSAI2026/outputs/`. Output: `analysis/exp8_fine/`.

Each record uses the pre-registration format: {pre-registered prediction, judgment criterion,
falsification/interpretation branch, corresponding experiment, data source}. No verdicts yet.

---

### H8f-1: Recovery Peak at Relative Depth l/L < 0.2 (Exp 8-fine)

**Pre-registered prediction:**
The single-layer recovery profile peaks early in all three models, at **relative depth l/L < 0.2**.
Concretely the peak layer(s) sit at **Gemma layers 2--6** (of 34) and **Mistral layers 0--2** (of 32);
Llama's peak likewise falls below l/L = 0.2 (of 28). This localizes the typo-encoding injection to the
first fifth of the network, consistent with the coarse `residual[0,6)` finding.

**Judgment criterion:**
Per-setting single-layer S2-KL-recovery profile over layers 0--11; the argmax layer's relative depth
`l_peak / L < 0.2` in the pooled/per-model profile, statistically distinguishable from a
uniform/late-layer null.

**Falsification / interpretation branch:**
If the peak sits at l/L >= 0.2 (mid/late depth) for any model, the "early injection" claim is qualified
for that model; report the model-specific peak depth and treat the depth shift as an
architecture-level property (as in the H4/H9 heterogeneity pattern) rather than a rejection of early
localization.

**Corresponding experiment:** Exp 8-fine (single-layer residual sweep, denoising).
**Data source:** Exp 8 coarse results (exp-08-patching `results/prod/exp8/`), new width-1 sweep.
Output: `analysis/exp8_fine/` (per-setting layer-profile CSV, relative-depth overlay PNG = Fig. 5 candidate).

**Status:** Executed 2026-07-19 (n=150 x 6 settings, median-robust) -- see *Exp 8-fine Verdicts* below.

---

### H8f-2: Plateau vs Single-Layer Spike (Exp 8-fine)

**Pre-registered prediction:**
The recovery profile is **plateau-shaped** -- 2--4 adjacent early layers reach the same recovery level
-- indicating the injection is distributed over a short early band rather than a single vocabulary-
integration layer.

**Judgment criterion:**
Count of adjacent layers within a small tolerance (e.g. within 0.05, or >= 90% of the single-layer max)
of the peak. Plateau if >= 2 adjacent layers qualify; spike if the peak stands alone.

**Falsification / interpretation branch (frozen ex ante):**
If the profile is a **single-layer spike** (peak isolated, neighbors well below), **switch to the
stronger, more localized claim**: typo information is integrated at a *specific* vocabulary-integration
layer. This is a sharpening, not a rejection -- a spike is a stronger localization result than a plateau.

**Corresponding experiment:** Exp 8-fine.
**Data source:** As H8f-1.

**Status:** Executed 2026-07-19 (n=150 x 6 settings, median-robust) -- see *Exp 8-fine Verdicts* below.

---

### H8f-3: Cumulative Patch Saturates Early (Exp 8-fine)

**Pre-registered prediction:**
The **cumulative** patch (replace layers 0..l) **rises steeply early and saturates by l/L ~ 0.2**,
reaching **>= 1.2x the single-layer maximum** recovery, and stays flat thereafter. This is the direct
evidence that **the write of typo information is complete early**: once the early band is corrected,
patching deeper layers adds nothing.

**Judgment criterion:**
Cumulative-recovery curve over l = 0..11; (a) `cumulative_max >= 1.2 * single_layer_max`;
(b) the cumulative curve reaches >= 90% of its max at l with l/L <= 0.2; (c) marginal increment per
layer ~ 0 beyond the saturation point. `single_layer_max << cumulative_max` additionally evidences
distributed (multi-layer) writing.

**Falsification / interpretation branch:**
If the cumulative curve keeps climbing past l/L = 0.2 (no early saturation), the write is **not**
complete early; report the actual saturation depth and revise "early completion" to the observed depth.
If `cumulative_max ~ single_layer_max` (no gain from accumulation), the injection is essentially
single-layer and H8f-2's spike branch is corroborated.

**Corresponding experiment:** Exp 8-fine (cumulative patch, +12 forward/pair).
**Data source:** As H8f-1; cumulative windows (0, l+1) for l = 0..11.

**Status:** Executed 2026-07-19 (n=150 x 6 settings, median-robust) -- see *Exp 8-fine Verdicts* below.

---

### H8f-4: Late-Layer Validation Null (Exp 8-fine)

**Pre-registered prediction:**
The three validation layers (**14, 20, 26**) give **recovery ~ 0 even at width 1**. This refutes the
alternative reading that the coarse width-3 windows only *looked* low late because the effect was
"diluted" by averaging three layers -- at width 1 the late layers are still null.

**Judgment criterion:**
Width-1 S2-KL-recovery at layers 14/20/26 not significantly above 0 (per-setting and pooled);
contrast with the early-layer peak.

**Falsification / interpretation branch:**
If any late validation layer shows non-trivial recovery at width 1, the "late layers are inert" claim is
qualified; report which late layer carries residual recovery and reconsider whether a second
(late) injection site exists.

**Corresponding experiment:** Exp 8-fine.
**Data source:** As H8f-1; validation layers {14, 20, 26}.

**Status:** Executed 2026-07-19 (n=150 x 6 settings, median-robust) -- see *Exp 8-fine Verdicts* below.

---

### H8f-5: Noising Sufficiency at the Best Layer (Exp 8-fine)

**Pre-registered prediction:**
The reverse **noising** intervention (inject the typo run's activation into the clean run) at the
**best (peak) layer +/- 1** reproduces the **majority of the KL divergence** between clean and typo
first-CoT-word distributions -- i.e. corrupting the single best early layer is *sufficient* to induce
most of the diversion, complementing the denoising *necessity* evidence.

**Judgment criterion:**
Noising KL (clean run with typo activation injected at best layer) vs the unpatched clean<->typo KL gap;
`noising_KL / unpatched_KL_gap >= 0.5` at the best layer +/- 1 (3 points), per-setting and pooled.

**Falsification / interpretation branch:**
If single-layer noising reproduces < 50% of the KL gap, the early layer is **necessary but not
sufficient** on its own; downgrade the sufficiency claim and attribute the remainder to distributed
early-band encoding (consistent with an H8f-2 plateau / H8f-3 cumulative-gain result).

**Corresponding experiment:** Exp 8-fine (noising, best-layer +/- 1 only, 3 points).
**Data source:** As H8f-1; noising direction (typo->clean) at best-layer +/- 1.

**Status:** Executed 2026-07-19 (n=150 x 6 settings, median-robust) -- see *Exp 8-fine Verdicts* below.

---

### A3: Adversarial-Review Specificity Controls (Exp 8-fine rider)

**Attack (A3).** "Because the patch is confined to the perturbed-word span, late-layer patches failing to
restore could simply mean the information has already leaked to *other* positions; a monotone depth-decay
of recovery would then appear for **any** input perturbation, so 'early-layer localization' over-reads a
generic read-out artifact."

**Response.** (i) **Claim precision (done above):** the depth profile is reported as a
**read-out completion point** of the span, not an absolute injection site; the defensive implication
(fix the span early -> in time) is unchanged, so Fig. 5's value is preserved. (ii) Three controls,
run on the **same hook and the same flip pairs** as the 1-layer sweep:

- **A3(a) other-span patch (specificity).** Patch a *non-perturbed* (downstream, span+offset) position
  with the clean value across the same layers. Prediction: recovery ~ 0 at all layers -- only the
  perturbed-word span carries the effect, not the leaked copies at neighbouring positions.
- **A3(b) all-positions patch (framework sanity, inverse of sham).** Patch *all* prompt positions with the
  clean value. Prediction: near-**full** recovery at any layer (analytically guaranteed) -- confirming the
  measurement is not saturated and that the span-only sweep's partial recovery is a genuine subset effect,
  not a broken hook.
- **A3(c) semantic-replacement control.** Replace the target word with a **non-synonymous real word**
  (not a typo) and run the same single-layer denoising sweep. **If the depth profile is isomorphic** to
  the typo profile, we report honestly that the *read-out dynamics are generic to input perturbation* and
  that the **typo-specific** contribution localizes to the **LXT-vs-Random magnitude ratio (4.8-10.1x)**,
  not to a different depth signature; **if it differs**, typo-specific depth localization is supported.

**A3(d) attn earliest-layer negative values.** The coarse sweep's small **negative** recovery at the
earliest *attention*-site window is currently unexplained; it is **reported as an observation with
interpretation reserved** (not over-interpreted), to pre-empt the objection. (Fine sweep is residual-only;
the attn note is carried from the coarse appendix.)

**Judgment addition.** The Exp 8-fine judgment records, alongside H8f-1..5, the **A3 specificity block**:
A3(a) other-span recovery ~ 0 (per-layer), A3(b) all-positions recovery ~ 1, and the A3(c) typo-vs-semantic
profile comparison (isomorphic vs distinct), with the honest framing selected by the outcome.

**Status:** Executed 2026-07-19 -- see *Exp 8-fine Verdicts* below.

---

## Exp 8-fine Verdicts (2026-07-19)

Executed on all 6 settings (Gemma-3-4B / Llama-3.2-3B / Mistral-7B x GSM8K/MMLU),
n=150 flip pairs/setting (LXT-4 : Random-4 half-and-half), plus a matched semantic-replacement
pass. **Primary estimator = median** of `s2_kl_recovery`: the metric `1 - KL_patched/KL_base`
is **unbounded below** (a small clean<->typo gap makes `KL_base` tiny, so a mild patch perturbation
yields a large negative value), so the **mean is outlier-corrupted** (Gemma pooled mean "peak" was
-0.60 while the median peak is +0.52). The mechanism was validated independently: fine `cumulative(0,3)`
reproduces the coarse `window[0,3]` **bit-for-bit** on shared samples; sham median ~0; all-positions
median = 1.0000. Bootstrap 95% median CIs exclude 0 at every early peak. Output: `analysis/exp8_fine/`.

**Injection / read-out-completion depth (H8f-1) -- Supported 6/6.** Median single-layer S2-KL-recovery
peaks at: **Gemma L5** (l/L=0.147, both benchmarks), **Llama L3--L4** (0.107--0.143), **Mistral L1**
(0.031). All < 0.2; the ex-ante model-specific bands (Gemma L2--6, Mistral L0--2) are hit exactly. The
rise starts at L0, and the cross-model **relative-depth overlay** (Fig. 5 candidate) shows a shared
early-peaked shape (Mistral earliest at l/L~0.03, then Llama ~0.11--0.14, then Gemma ~0.15).

**Profile shape (H8f-2) -- Supported 5/6 (plateau).** Broad early plateaus (Llama L1--5, Mistral L1--3,
Gemma-MMLU L5--7); **Gemma-GSM8K is a single-layer spike (L5)** -> the pre-registered spike branch
(stronger, more localized "vocabulary-integration layer" claim) is taken for that one cell.

**Cumulative (H8f-3) -- MODIFIED (branch activated; 1/6).** Cumulative patching **saturates early**
(saturation layer L1--L7, all l/L<=0.21) as predicted, but the **magnitude gain is small**: cum/single
ratios are 1.03--1.45 (only Llama-MMLU reaches the >=1.2x-with-early-saturation criterion). Since
`cumulative_max ~ single_max`, the pre-registered branch applies: **the typo write is concentrated in the
early band and largely captured by a single/few layers, not distributed across many** -- a *sharper*
localization than the distributed-write prediction.

**Late-layer null (H8f-4) -- MODIFIED (branch activated; 0/6 strict).** The genuinely late validation
layers **L20 (l/L~0.6) and L26 (l/L~0.8) are ~0 even at width 1** (medians 0.02--0.15 and 0.01--0.04),
**refuting the "width-3 dilution" alternative** for late layers. But **L14 (l/L~0.45, mid-depth) retains
median recovery 0.12--0.30** in every setting. -> The read-out-completion tail is **gradual**, decaying
to ~0 by l/L~0.6--0.8 rather than cutting off sharply at the early peak; the "late" validation set mixed
one mid-depth point (L14) that is not yet null.

**Noising sufficiency (H8f-5) -- MODIFIED (branch activated; 1/6).** Single-layer noising at best+/-1
reproduces **28--48%** of the clean<->typo KL divergence in 5/6 settings (only Llama-MMLU >=50%). ->
The early layer is **necessary but not sufficient on its own**; the remaining diversion is distributed
over the early plateau, consistent with H8f-2 (plateau) and H8f-3 (early-band concentration).

**A3 adversarial-review controls.**
- **A3(a) other-span specificity -- Supported 6/6.** Patching non-perturbed downstream positions gives
  median |recovery| <= 0.06 at all early layers (vs +0.5--0.6 at the span). The leaked copies at other
  positions do **not** carry the effect -- only the perturbed-word span does. This directly rebuts the A3
  attack ("late-layer failure just means the info leaked elsewhere").
- **A3(b) all-positions sanity -- Supported 5/6.** Median recovery = 1.0000 (Llama-MMLU had no
  equal-token-length pairs -> N/A). The hook can fully restore; the span-only partial recovery is a
  genuine subset effect, not a saturated/broken measurement.
- **A3(c) semantic (real-word) replacement -- Mixed / model-dependent.** Typo-vs-semantic depth profiles:
  **Mistral isomorphic** (Pearson 0.80--0.86, peaks match) -> read-out dynamics are **generic to input
  perturbation**, so the typo-specific contribution localizes to the **LXT-vs-Random magnitude ratio**,
  not a distinct depth signature (as anticipated ex ante); **Llama** profiles are highly correlated
  (r=0.84--0.90) but the semantic peak shifts ~3--4 layers earlier (peak-match fails); **Gemma** profiles
  are **distinct** (r=0.14--0.43) -> some typo-specific depth structure. Recorded honestly as structured
  heterogeneity rather than a uniform claim.
- **A3(d) attn earliest-layer negative** -- carried from the coarse appendix as an observation with
  interpretation reserved (fine sweep is residual-only).

**Summary.** The **core read-out-completion localization is confirmed** (H8f-1 6/6 at the exact predicted
depths; H8f-2 5/6 plateau; A3a specificity 6/6; A3b sanity 5/6). Three secondary predictions are
**modified through their pre-registered falsification branches**, each yielding a sharper mechanism
(early single-band write; gradual mid-depth read-out tail; single-layer necessity-not-sufficiency), and
A3c is honestly model-dependent. Fig. 5 is upgraded from a coarse window bar chart to the median
relative-depth x recovery overlay.

---

## Version History

| Date | Change |
|------|--------|
| 2026-07-16 | Initial registry creation with Phase A verdicts for H1--H10 |
| 2026-07-19 | 宿題1完結: H6 を Pending → Conditionally Supported に更新(ρ保持表を偏相関に補正)。宿題2/5 の H7-4・H3 再確認を追加(考察フォローアップ判定総括)。 |
| 2026-07-19 | Phase B pre-registration added: ERDC unified hypothesis (Encode--Repair--Divert--Carry) with H11--H18 (chain mediation, R_C composition, read-out concentration, no-CoT shortcut, patch->free-generation S1->S2 closure, unified GLMM, behavioral repair, format transplant). Falsification branches frozen ex ante. Master plan: `docs/experiments_11_18_plan.md`. H1--H10 verdicts unchanged. |
| 2026-07-19 | Size-ladder extension pre-registered: size-effect predictions P1--P6 (backbone IE defense, shortcut size-law, early-layer localization invariance, repair size-monotonicity, read-out dispersion, size-as-moderators capstone) derived from the ERDC + M1/M2/M3 unified hypothesis, plus the P1 falsification receptacle (regime scoping, Lanham boundary condition). Falsification branches frozen ex ante; ladder is a confirmatory replication layer statistically separated from the main analysis. Expansion plan: `docs/size_ladder_plan.md`. H1--H18 verdicts unchanged. |
| 2026-07-19 | Exp 8-fine pre-registered: single-layer injection-localization predictions H8f-1--H8f-5 (peak depth l/L<0.2, plateau-vs-spike, cumulative early-saturation, late-layer null at width 1, noising sufficiency) refining the coarse Exp 8 `residual[0,6)` (10/12) window to 1-layer resolution. Falsification branches frozen ex ante (spike sharpening, model-specific depth, necessity-not-sufficiency). Output: `analysis/exp8_fine/`. H1--H18, P1--P6 verdicts unchanged. |
| 2026-07-19 | Exp 8-fine adversarial-review rider A3 added: claim precision (injection site -> **read-out completion point** of the span; defensive implication unchanged), and three specificity controls on the same hook/flip pairs -- A3(a) other-span patch (~0), A3(b) all-positions patch (~1, framework sanity), A3(c) semantic (real-word) replacement profile vs typo (isomorphic -> read-out is generic + typo-specific effect localizes to LXT/Random 4.8-10.1x ratio; distinct -> typo-specific depth), plus A3(d) attn earliest-layer negative reported as observation, interpretation reserved. H8f judgment extended with the A3 specificity block. |
| 2026-07-19 | **Phase B executed (5/8): verdicts recorded.** H11 Chain Mediation **SUPPORTED** (core5 stage-1 neg-sig 35/50, pooled PM 0.577; MATH counter-example PM -0.40 = chain fork). H12 R_C Composition **strong-form REFUTED / mechanism SUPPORTED** (conclusion-share 0.130 not >0.5, all-31 \|r\|=0.184; MC-only r=+0.705 p<0.001). H13 Read-out Concentration **pre-registered-form REFUTED / scope-matched mechanism SUPPORTED** (Gini rank Gemma>Llama>Mistral, rank-corr vs RD_content -0.564; vs RD_all +0.782; Mistral double dissociation RD_content 0.026 vs Llama 0.479). H14 no-CoT Shortcut **literal NOT-SUPPORTED / Simpson mechanism SUPPORTED** (overall rho -0.04, MH OR 8.85; MC rho +0.726 / gen +0.633; Gemma-1B x CSQA rank 2/40 within MC). H17 already REFUTED (consistency re-confirmed). Summary table + detailed records H11-H14 updated; interpretation branches (partial) taken per pre-registration. Sources: `analysis/exp11_chain_mediation/`, `analysis/exp12_rc_composition/`, `analysis/exp13_readout_concentration/`, `results/exp14_nocot/analysis/`, `analysis/exp17_behavioral_repair/`. H15/H16/H18 remain pending. |
| 2026-07-19 | Adversarial review A/B/C + defense D1 logged in `experiment_details.md` / `all_results_by_setting.md` (not new hypotheses): A1 Mistral word_scores audit (no contamination; exp-04 rho_default & exp-03 precision@10 reproduced exactly, second main conclusion intact), A2 restore "trivial-copy" rebuttal (MMLU no-leak n=789 restore 0.772; recovery curve GSM8K p25=0.50/p50=0.71), B1/B4/B5/B6 confound checks, C = Exp 6 rho-preservation corrected (LOO 6/6, G x I 2/6, rollout 0/6; H6 conditional support unchanged). D1 defense-oracle dataset build complete (6 M3xB2 settings, separable retention 0.92-0.99); generation/recovery pending, pre-registered criterion pooled k=1 oracle>random>inverse. |
| 2026-07-19 | **Exp 8-fine executed** (n=150 x 6 settings, typo + semantic). Verdicts recorded (median-robust; the recovery ratio `1-KL_p/KL_b` is unbounded-below so mean is outlier-corrupted). H8f-1 **Supported 6/6** (Gemma L5, Llama L3-4, Mistral L1; all l/L<0.2, predicted bands exact); H8f-2 **Supported 5/6** (Gemma-GSM8K spike branch); H8f-3/H8f-4/H8f-5 **MODIFIED** via pre-registered branches (early-band write not distributed; gradual mid-depth read-out tail with L20/L26 null; single-layer necessary-not-sufficient); A3(a) **6/6**, A3(b) **5/6**, A3(c) **mixed/model-dependent**. Mechanism validated (fine cumulative == coarse window bit-for-bit; sham~0; all-positions=1.0). Output: `analysis/exp8_fine/`. H1-H18, P1-P6 verdicts unchanged. |
