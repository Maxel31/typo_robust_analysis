# ARR August 2026 Resubmission: 章立て確定案

Phase A 再分析結果に基づく修正C（本文4本柱）の章立て。

---

## 全体構成

| # | Section | 紙幅目安 |
|---|---------|---------|
| - | Title | - |
| - | Abstract | 0.3p |
| 1 | Introduction | 1.0p |
| 2 | Framework | 1.0p |
| 3 | Experimental Setup | 0.8p |
| 4 | Results (4 Pillars) | 2.5p |
| 5 | Supporting Evidence | 1.5p |
| 6 | Discussion | 1.0p |
| 7 | Limitations | 0.5p |
| - | References | 0.5p |
| - | Appendices A--F | 3--4p |
| | **Total (main body)** | **~8.0p** (long paper 8p 制限) |

---

## Title

**How Typos Derail Chain-of-Thought: A Causal Mediation Analysis Across Surface, Attribution, and Representation Layers**

代替案: "Typos Break Reasoning Through the Chain: Causal Evidence from Three Analytical Layers"

---

## Abstract

- **この節の主張**: Typo が CoT 推論を壊すメカニズムを三層フレームワーク (Surface / Attribution / Representation) で解明する。主推定量は反事実的 **restore 率**（typo で反転した答えのうち clean CoT を移植 do(CoT:=clean) すると元の答えに復帰する割合; pooled 76% / GSM8K 96%）であり、答えに重要なトークンの数値語集中、入力校正器のニューラル > LLM という意外な優位を報告する。TE/DE/IE のセル別リスク分解では間接経路が支配的（記述的比率 IE/TE≈0.83）だが、これは非加法（DE+IE≠TE）ゆえ「媒介割合」ではなく記述指標として提示する。
- **紙幅目安**: ~250 words

構成:
1. 問題設定 (typos in questions degrade CoT reasoning -- but how?)
2. 手法の骨格 (three-layer framework, M5 x B5, 10 experiments)
3. 主要発見 (do(CoT:=clean) restores 76% of flipped answers under LXT-4 (96% on GSM8K); cell-level risks TE/DE/IE support an indirect-path-dominant decomposition, descriptive IE/TE=0.83; numeric token dominance in R_C; neural corrector restores 87% of tokens vs LLM's 71%)
4. 含意 (defense should target token restoration, not full rewriting)

---

## 各節の詳細

---

### S1. Introduction (~1.0p)

- **この節の主張**: LLM の Chain-of-Thought 推論がノイジーな入力（typo）に脆弱であることは知られているが、「なぜ・どこで CoT が壊れるのか」の因果的理解は欠如している。本研究は三層フレームワークと 10 実験で因果分解・帰属・内部表現の各水準からこの問いに答える。
- **使用する図表**: なし（テキストのみ）
- **依存する実験番号**: 全体の動機づけ（直接の数値引用は避ける）
- **紙幅目安**: 1.0p

#### 構成:

**Paragraph 1 -- Motivation**:
- LLM reasoning (CoT prompting, reasoning models) is now standard
- Real-world inputs contain typos, misspellings, OCR errors
- Prior work (Gan+ 2024, Gao+ 2024) showed accuracy drops, but treated the CoT as a black box
- Gap: no causal understanding of *how* typos propagate through the reasoning chain

**Paragraph 2 -- Why CoT** (verbatim from experiment_plan.md S6.5):

> We deliberately center our analysis on the CoT, for three reasons. First, in
> autoregressive generation the CoT is not a post-hoc report but a node in the
> computational graph: generated CoT tokens are re-consumed as context by the
> answer step, which is precisely what makes do(CoT) interventions (Exp. 1-2)
> well-defined. Second, the CoT is the only intermediate variable that is
> comparable across architectures and tokenizers, human-readable for diagnosis,
> and directly actionable by text-level defenses. Third, rather than assuming
> that the CoT faithfully captures the computation, our design measures the
> residual: the direct effect that bypasses the CoT text (Exp. 1) is quantified
> and then localized at the representation level (Exp. 8). CoT-centrality is
> thus a tested property of the pathway, not a premise of the framework.

**Paragraph 3 -- Contributions** (4 pillars):
1. **Causal decomposition** (Exp 1): a CoT-transplant 2x2 design with a counterfactual primary estimand -- the *restore rate*, i.e. the fraction of typo-flipped answers that return to the clean answer under do(CoT:=clean) -- shows that transplanting a clean CoT restores 76% of flipped answers (96% on GSM8K). The cell-level risks (TE/DE/IE) support an indirect-path-dominant decomposition (descriptive IE/TE=0.83; we do *not* interpret this ratio as a "proportion mediated", since the effects are non-additive, DE+IE!=TE). Following Vig et al. (2020), who introduced causal mediation analysis at the neuron level, we carry the direct/indirect decomposition to the *text-artifact* level (the CoT text) and connect it to defense design.
2. **Targeted deletion** (Exp 2): Top-R_C tokens cause 67% flip vs 4% for controls (x15 ratio); the critical tokens are overwhelmingly numeric/arithmetic, not content words.
3. **Fixed-target diagnostic** (Exp 4): Endogeneity-corrected attribution reveals family x format interaction (eta^2=0.066, p=8.9e-5) -- MC formats inflate correlations for Gemma/Llama but not Mistral.
4. **Corrector ladder, read as defense prioritization** (C4; Exp 7, synthesized with Exp 8/15): the neural corrector (T5) restores 87% of tokens and nearly recovers clean accuracy, outperforming LLM correction (71% restoration), and conservative (token-level) restoration beats aggressive (sentence-level) rewriting. We read this not as a defense we propose but as a *mechanistic basis for prioritizing defense investment*: the rate-limiting factor is the restoration precision of answer-relevant words -- the residual corrector gap concentrates (weakly) on high-R_Q tokens (γ; Exp 7, Holm-significant in 17/25 pooled settings; oracle ceiling to be set by D1) -- and, when an internal fix is warranted, its target localizes to an early-layer band (Exp 8, 12/12 conditions; Exp 15's early-window patch redirects free generation). Section 6 states the resulting four-point intervention map as *falsifiable predictions*, keeping the paper an analysis rather than a method.

---

### S2. Framework (~1.0p)

- **この節の主張**: Surface / Attribution / Representation の三層からなる分析フレームワークを定式化し、各層の主要概念（R_Q, R_C, fixed-target, flip）を定義する。
- **使用する図表**: **Fig. 1** (三層フレームワーク概念図 -- Surface/Attribution/Representation の関係を示す図)
- **依存する実験番号**: 定義のみ（実験0の凍結条件を参照）
- **紙幅目安**: 1.0p

#### 構成:

**2.0 Relation to prior work (four islands, one bridge).**
本フレームワークは四つの確立した研究系譜の上に立つが、我々の新規性は個々の「島」の中身ではなく、島と島を架ける「橋」にある。(i) *摂動下の CoT faithfulness*: Lanham et al. (2023) は CoT を切断・破損 (truncation/corruption) すると答えが変わることを示し、忠実性を挙動レベルで診断した。(ii) *ニューラル計算の因果媒介分析*: Vig et al. (2020) はモデル挙動を直接効果・間接効果に分解する causal mediation analysis を **ニューロンレベル** で導入した。(iii) *inner lexicon 仮説*: Kaplan et al. (2025) は初期層がサブワード断片から正規の語形を再構成すると主張する。(iv) *typo ニューロン / 表現デノイズ*: Tsuji et al. (2025) は typo 処理を特定ユニットに局在させる。本研究は、これらを架橋する: **テキスト人工物 (CoT テキスト) レベルの因果媒介分解** (島 ii をニューロンではなく CoT テキストへ移設) を、表層復元の防御的読み (島 iii/iv) と挙動的忠実性 (島 i) に接続する。とりわけ実験1は Lanham et al. の摂動版 **ではない**: (a) typo は CoT 編集ではなく**質問側から入力**される (入力摂動起点)、(b) 主推定量は do(CoT:=clean) 下の反事実的 **restore 率**、(c) 測定はアーカイブ実行との TE 再現照合で**監査済み**、(d) 単一プローブでなく **50 設定 (model x benchmark x perturbation)** 規模で実施する。この4点が Lanham 系との差分である。

**2.1 Three Analytical Layers**
- **Surface layer**: CoT テキストの変化を測定。ROUGE-L (F1, clean vs typo CoT), Jaccard@k (clean CoT と typo CoT の top-k 語彙重複)
- **Attribution layer**: 答えに影響する CoT トークンの特定。R_Q (question-side attribution), R_C (CoT-side attribution) を AttnLRP で計算
- **Representation layer**: 隠れ状態レベルでの表現変化。activation patching, cosine repair score

**2.2 Key Definitions**

- **R_Q (question-side relevance)**:
  $$R_Q(q_i) = \text{AttnLRP}(\text{logit}(a) \mid q_i)$$
  質問中の各トークンの答えへの寄与度

- **R_C (CoT-side relevance)**:
  $$R_C(c_j) = \text{AttnLRP}(\text{logit}(a) \mid c_j)$$
  CoT 中の各トークンの答えへの寄与度

- **Fixed-target attribution**:
  $$R_C^{\text{fixed}}(c_j) = \text{AttnLRP}(\text{logit}(a^{\text{clean}}) \mid \text{teacher-force}(Q_p, \text{CoT}_p, \text{trigger}))$$
  帰属先を元の答え (clean answer) に固定した R_C。内生性 (答えが変わると帰属先も変わる) を除去。

- **Flip (primary estimand)**:
  Clean 条件で正解だったサンプルが typo 条件で不正解になること (correct -> incorrect)。全実験の主推定量。

- **Delta-rho**:
  $$\Delta\rho = \rho_{\text{fixed}} - \rho_{\text{default}}$$
  Fixed-target と default の偏相関の差。正値 = default が内生性で膨張していた。

**2.3 Causal Graph** (Fig. 1 の一部として)
- Q_typo -> CoT_typo -> Answer (indirect path)
- Q_typo -> Answer (direct path, bypassing CoT)
- Exp 1 decomposes TE = IE + DE along this graph

---

### S3. Experimental Setup (~0.8p)

- **この節の主張**: 5 モデル x 5 ベンチマークの 25 設定で統一された実験条件を設定し、2 種の摂動条件（LXT-4, Random-4）で体系的評価を行う。
- **使用する図表**: **Table 1** (M5 x B5 coverage matrix with adoption criteria)
- **依存する実験番号**: Step 0 (凍結条件)
- **紙幅目安**: 0.8p

#### 構成:

**3.1 Models (M5)**

| Model | Family | Params | Layers |
|-------|--------|--------|--------|
| Llama-3.2-1B | Llama | 1.2B | 16 |
| Llama-3.2-3B | Llama | 3.2B | 28 |
| Mistral-7B-Instruct-v0.3 | Mistral | 7.2B | 32 |
| Gemma-3-1B-it | Gemma | 1.0B | 26 |
| Gemma-3-4B-it | Gemma | 3.9B | 34 |

全モデル instruction-tuned 版を使用。M3 (Representation 層分析用) = Gemma-3-4B, Llama-3.2-3B, Mistral-7B。

**3.2 Benchmarks (B5)**

| Benchmark | Task Type | Answer Format | # Samples |
|-----------|-----------|--------------|-----------|
| GSM8K | Math reasoning | Free-form numeric | 1,319 |
| MMLU | Knowledge QA | 4-choice MC | 2,850 |
| MMLU-Pro | Hard knowledge QA | 10-choice MC | 1,400 |
| ARC-Challenge | Science reasoning | 4-choice MC | ~1,170 |
| CommonsenseQA | Commonsense | 5-choice MC | ~1,220 |

**3.3 Common Conventions**
- Decoding: greedy (do_sample=False, temperature=0.0)
- Seed: 42
- Max new tokens: 512
- CoT prompting: "Let's think step by step" suffix

**3.4 Perturbation Conditions**
- **LXT-4**: AttnLRP 重要度 top-4 語に typo 注入（重要語標的）
- **Random-4**: ランダムに選んだ 4 語に typo 注入（非標的ベースライン）
- Typo 操作: 文字の入替・削除・挿入（先行研究準拠）

**3.5 Adoption Criteria**
- chance + 10pt 以上の clean accuracy を持つ設定のみを主分析対象に採用
- Span extraction 失敗率の明示（全体 13.19%、最大 38.59% = Mistral x GSM8K）

---

### S4. Results: Four Pillars (~2.5p)

---

#### S4.1 Causal Decomposition: A Clean CoT Restores Most Flipped Answers (Exp 1) (~0.7p)

- **この節の主張**: CoT 移植 2x2 設計により、主推定量である反事実的 **restore 率** (typo で反転した答えのうち do(CoT:=clean) で元の答えに復帰する割合) が pooled 76% (GSM8K 96%) に達することを示す。TE/DE/IE はセル別リスクとして提示し、間接経路が支配的な分解を **支持** する (記述的比率 IE/TE≈0.83)。**本論文の最重要結果。**
- **使用する図表**: **Fig. 2** (Headline figure: restore 率の棒グラフ + TE/DE/IE のセル別リスク、M5 x B5 ヒートマップ), **Table 2** (Pooled restore / TE/DE/IE / descriptive IE/TE / sensitivity, LXT-4 vs Random-4)
- **依存する実験番号**: Exp 1
- **紙幅目安**: 0.7p

**用語の凍結 (査読対策)**: 主推定量は **restore 率** = 「TE で反転した事例のうち clean CoT を強制 (セルC, do(CoT:=clean)) すると元の答えに復帰した割合」(反事実的に一意)。TE/DE/IE は 4 セル (A/B/C/D) の **flip リスク** であって効果の加法分解ではない (GLMM の q×cot 交互作用は負, -5.9 前後 = サブ加法的で DE+IE!=TE)。したがって **IE/TE は記述的比率**であり「媒介割合 (proportion mediated)」とは呼ばない。因果媒介分解の枠組みは Vig et al. (2020) のニューロンレベル分析を **CoT テキスト (テキスト人工物) レベル** へ移し、防御設計に接続する点で位置づけられる (Pearl/VanderWeele 型の proportion-mediated は非加法下で定義されないため主張しない)。

本文の数値 (Phase A 確定, `results/exp01_03/*/outcomes.json` から再集計):

| Condition | restore (primary) | TE | DE | IE | IE/TE (descriptive) | sensitivity TE/DE/IE (除外込み) | IE\|ROUGE<1 |
|-----------|-----|-----|-----|------|-------|-------|-------|
| Pooled LXT-4 | 76.2% (GSM8K 95.8% / MC 73.0%) | 23.9% | 7.4% | 19.7% | 0.83 | 26.0 / 8.1 / 21.8% | 20.5% |
| Pooled Random-4 | 72.2% (GSM8K 95.4% / MC 67.5%) | 17.0% | 6.1% | 13.6% | 0.80 | 18.9 / 6.5 / 15.4% | 15.2% |

強調ポイント:
- **restore 優位 (間接経路支配) が両摂動条件 (LXT-4, Random-4) で 48/50 設定に成立** -> 修正A の見出し論理成立。唯一の例外は Gemma-3-1B x CommonsenseQA の両条件 (DE > IE の 1B x 多肢選択)。
- **選択バイアスへの頑健性 (事前登録の感度分析)**: 構造的除外 (trigger 検出失敗等) を含めた「除外込み感度分析」でも記述的 IE/TE はほぼ不変 (LXT 0.83->0.84, Random 0.80->0.82) で、restore 優位の見出しは除外設計のアーティファクトではない。CoT が実際に変化した事例に条件付けた IE (IE|ROUGE<1) も併記 (LXT 20.5% / Random 15.2%)。
- **within-run ノイズは 0**: 4 セル (A/B/C/D) は同一ラン・同一バッチ (greedy) で生成されるため再現性ノイズを含まない (実験7 の within-run 検証で byte-identical->flip 0/45,641)。付録で報告するクロスラン flip 9.56% はアーカイブ比較固有のノイズフロアであり、本 4 セル設計の DE (GSM8K で ~1.3%) には適用されない。
- Gemma-4B x GSM8K で restore 93% (IE/TE=0.98 LXT / 1.02 Random) -- CoT がほぼ全経路を担う設定。
- DE は非ゼロだが小 (7.4%) -> Exp 8 の activation patching で局在化の伏線。
- GLMM: cot_typo 効果が支配的 (Gemma-4B x GSM8K LXT: coef=4.998 vs q_typo=1.352)。

ナラティブ: "When we transplant a clean CoT into a perturbed context (cell C), most flipped answers return to the clean answer (restore rate 76%, and 96% on GSM8K) -- the damage travels *through* the reasoning text rather than around it."

---

#### S4.2 Targeted Deletion: Numeric Tokens Are the Critical Bottleneck (Exp 2) (~0.6p)

- **この節の主張**: CoT 中の R_C 上位トークンの削除は flip 率を劇的に上昇させる（top-4 で 66.7% vs 統制 4.3%、x15 倍）。さらに、その重要トークンは圧倒的に数値・演算語であり、内容語のみの削除は flip をほぼ引き起こさない。**新発見: numeric vs content の二分性。**
- **使用する図表**: **Fig. 3** (deletion effect by token stratum: numeric vs content bar chart), **Table 3** (flip rates by target x dose)
- **依存する実験番号**: Exp 2
- **紙幅目安**: 0.6p

本文の数値 (Phase A 確定):

**Gemma-4B x GSM8K (smoke, n=24)**:
- Top-R_C unrestricted delete k=4: **66.7%** flip vs stratum_matched_random: **4.3%** (x15 ratio)
- Numeric top k=4: **91.7%** flip
- Content-only deletion: **~0%** flip
- Dose-response: unrestricted top は単調 (slope 0.214, p=5e-4)、統制はフラット

**MMLU settings (本番)**:
- Gemma-4B x MMLU: top k=4 = **20.7%** vs control **2.1%** (RD=0.172, CI [0.135, 0.209], p=4e-19)
- Gemma-4B x MMLU-Pro: top k=4 = **25.5%** vs control **2.4%** (RD=0.209, p=3e-26)
- Gemma-1B x GSM8K: top k=4 = **85.0%** vs control **4.2%** (RD=0.803, p=8e-111)

強調ポイント:
- GSM8K の R_C 上位はほぼ数値・演算語 -> CoT の算術ステップが最重要
- MMLU では content 層でも top > matched (k=4: 9.5% vs 1.7%, p<0.001) -> 形式依存
- Dose-response の単調性 -> 因果的解釈を支持

ナラティブ: "Deleting four top-R_C tokens from the CoT causes 67% of answers to flip -- but this effect is almost entirely carried by numeric tokens. Content words, even highly attributed ones, contribute negligibly."

---

#### S4.3 Fixed-Target Diagnostic: Family x Format Interaction (Exp 4) (~0.6p)

- **この節の主張**: Fixed-target 分析を診断ツールとして再定義する。内生性を除去した測定により、delta_rho に強い家族 x 答え形式の交互作用が現れる。Gemma/Llama は多肢選択で大幅減衰（平均 delta_rho = +0.371/+0.575）、自由記述でほぼ不変。一方 Mistral は全設定で負の delta_rho（相関強化）を示す。
- **使用する図表**: **Fig. 4** (family x format heatmap of delta_rho -- 主図の一つ), **Table 4** (25-setting delta_rho summary, 付録 B に full table)
- **依存する実験番号**: Exp 4
- **紙幅目安**: 0.6p

本文の数値 (Phase A 確定):

**Two-way ANOVA** (family x format on delta_rho):
- Family main effect: eta^2 = 0.397, p = 1.2e-10
- Format main effect: eta^2 = 0.085, p = 3.9e-6
- Interaction: eta^2 = 0.066, p = 8.9e-5

**Family x Format means**:

| | MC (delta_rho) | Free-form (delta_rho) |
|---|---|---|
| Gemma | +0.371 (large attenuation) | -0.006 (near zero) |
| Llama | +0.575 (large attenuation) | -0.089 (slight strengthening) |
| Mistral | negative (pooled) | negative (pooled) |

- **Mistral 全設定**: 負の delta_rho (pooled mean = -0.077, CI [-0.127, -0.028])
- **Gemma/Llama only (format effect)**: p = 5.4e-7, MC mean = +0.473, FF mean = -0.047

強調ポイント:
- Fixed-target は相関の強弱を測るツールではなく、内生性の構造を診断するツール
- Gemma/Llama の MC 減衰 = 答え選択肢への帰属が flip 時に再配分される証拠
- Mistral の負 delta_rho = 固有の帰属構造（Discussion S6.1 で深掘り）
- 25 設定中 22 で rho_fixed は Holm 有意残存 -> 帰属の基本関係は堅牢

ナラティブ: "Fixed-target attribution serves as a diagnostic: when the endogeneity correction *changes* the correlation, the original measurement was partly an artifact of answer-dependent re-attribution."

---

#### S4.4 Corrector Ladder: Neural > LLM, Conservative Correction Advantage (Exp 7) (~0.6p)

- **この節の主張**: 3段の入力校正器（pyspellchecker / T5-large-spell / Qwen-7B）で typo を修復し、ニューラル校正器 (T5) が LLM 校正器 (Qwen) を上回ることを示す。byte-identical に復元されたサンプルは flip 率 0% であり、token-level の保守的修復が sentence-level の積極的書き換えより安全である。
- **使用する図表**: **Fig. 5** (3-panel: corrector strength x {restoration rate, accuracy recovery, R_Q failure concentration}), **Table 5** (accuracy by corrector x model x benchmark)
- **依存する実験番号**: Exp 7
- **紙幅目安**: 0.6p

本文の数値 (Phase A 確定、一部進行中):

**Token restoration rate**:
- Neural (T5): **0.871**
- LLM (Qwen): **0.710**

**Byte-identical -> flip confirmation**:
- byte-identical サンプルは全件 flip = 0% (T5: 0/9, Qwen: 0/5)

**Accuracy recovery** (代表設定):

| Setting | Clean | LXT-4 | Spellfix | Neuralfix | LLMfix |
|---------|-------|-------|----------|-----------|--------|
| Llama-3B x GSM8K | 0.705 | 0.640 | 0.627 | 0.695 | 0.688 |
| Llama-1B x GSM8K | 0.361 | 0.335 | 0.334 | 0.360 | 0.369 |
| Gemma-1B x GSM8K | 0.404 | 0.330 | 0.356 | 0.403 | - |

強調ポイント:
- Neuralfix (T5) は多くの設定で clean accuracy にほぼ到達（Llama-3B: 0.695 vs clean 0.705）
- Spellfix (pyspellchecker) は LXT-4 とほぼ同等 -> 非文脈型校正は不十分
- 「保守的修復の優位 (conservative correction advantage)」: byte-identical 復元 = flip 0% は、完全な字面復元が十分条件であることを示す
- LLM 校正の劣位は、生成的書き換えが意味変更のリスクを伴うことを示唆
- 壁打ち事前予測「LLM で 90% 超」は裏切られた -> 実験的発見

ナラティブ: "The neural corrector restores 87% of perturbed tokens and nearly recovers clean accuracy, while the LLM corrector -- despite being a stronger language model -- restores only 71%. Perfect byte-level restoration guarantees zero flips, establishing a *conservative correction advantage*."

---

### S5. Supporting Evidence (~1.5p)

---

#### S5.1 KL-R_C Complementarity (Exp 3) (~0.3p)

- **この節の主張**: KL 発散上位位置（摂動が推論を逸らす地点）と R_C 上位位置（答えに重要なトークン）は空間的に相補的（complementary）な分布を示し、前者は CoT の前半、後者は後半に集中する。これは「摂動がまず文脈を汚染し、その影響が下流の重要トークンに伝播する」という因果連鎖を支持する。
- **使用する図表**: **Fig. 6** (KL vs R_C positional density plot), 付録 D に全プロファイル
- **依存する実験番号**: Exp 3
- **紙幅目安**: 0.3p

注意: 旧版の "agreement" から "complementarity" にリフレームする。KL 上位と R_C 上位の重なりが低い (precision@k = 0.151 vs null 0.334) ことは矛盾ではなく、異なる段階を捉えている証拠。

本文の数値:
- 60 settings 分析
- Global KL mean position: **0.359** (前半集中), R_C mean position: **0.541** (後半集中)
- Global Wasserstein distance: **0.182**, position separation p = 0.0
- KL front fraction ~73--87%, R_C front fraction はモデル依存
- GSM8K で最強の分離 (Wasserstein 0.40--0.47)
- Importance 条件 (LXT-4) での分離がランダム摂動より強い

---

#### S5.2 Matched Controls (Exp 5) (~0.2p)

- **この節の主張**: 5 変数の層化マッチング後も、25 設定中 22 で LXT 標的語の flip 率が matched random を有意に上回る。非有意は低精度モデル x GSM8K 系のみ。
- **使用する図表**: **Table 6** (summary row: 22/25 significant, 付録 C に full balance tables)
- **依存する実験番号**: Exp 5
- **紙幅目安**: 0.2p

本文の数値:
- 22/25 settings significant (cond p < 0.05)
- Non-significant: 低精度 model x GSM8K systems (Llama-1B x GSM8K p=0.41, Gemma-1B x GSM8K p=0.092, Gemma-1B x MMLU-Pro p=0.21)
- Risk difference (LXT の追加低下) > 0: 24/25 settings

---

#### S5.3 Attribution Method Comparison (Exp 6) (~0.2p)

- **この節の主張**: LOO (Leave-One-Out) による帰属なし再構成でも R_C との相関が保持され、AttnLRP 依存性が排除される。
- **使用する図表**: 本文は 1--2 文 + 付録参照
- **依存する実験番号**: Exp 6
- **紙幅目安**: 0.2p

本文の数値:
- LOO vs R_C Jaccard@10: 0.460 (案B, Gemma-4B x GSM8K)
- 案B vs 案A Top-10 Jaccard: mean 0.755 (ランキング概ね保持)

---

#### S5.4 Activation Patching (Exp 8) (~0.3p)

- **この節の主張**: DE (CoT を迂回する直接効果) が中間層帯の質問スパン表現に局在することを activation patching で確認。Representation 層の実証。
- **使用する図表**: **Fig. 7** (layer x site recovery heatmap)
- **依存する実験番号**: Exp 8
- **紙幅目安**: 0.3p

本文の数値:
- M3 x B2 (3 models x 2 benchmarks) でスモーク完了
- Gemma-3-4B x GSM8K: 11/16 ペア完了
- 計画比で大幅に低コスト (~0.4 GPU日 vs 計画 4--6 GPU日)

注意: 本番結果待ち。スモーク結果のみの場合は概要のみ記載し、詳細は付録送り。

---

#### S5.5 Internal Repair (Exp 9) (~0.5p)

- **この節の主張**: Llama/Mistral では repair score (typo 語の hidden state が clean 語に回復する速度) が flip の強い負予測子（修復が速い語ほど flip しにくい）。Gemma は cos 類似度が 0.99 近傍に飽和し、repair score の分散がほぼゼロになる（飽和現象）。
- **使用する図表**: **Fig. 8** (repair score vs flip, family 別), Gemma 飽和の詳細は S6.1 または付録 E に分離
- **依存する実験番号**: Exp 9
- **紙幅目安**: 0.5p

本文の数値 (Phase A 確定):

**Llama/Mistral pooled regression**:
- Llama+Mistral pooled: repair_coef = **-0.977**, p = 3.7e-86
- All 5 models pooled: repair_coef = **-0.707**, p = 1.5e-97
- Cohen's d: Llama = 0.124, Mistral = 0.149

**「最強の負予測子」の範囲 (Track B 確定, C5)**: repair_score が「最強の負予測子」であるのは **集計 (pooled / 条件別) 水準に限定** される主張である。4 共変量 (repair/split_inc/zipf/r_q) を z 標準化した同一 GLM で、pooled では 14/14 の集計で repair が最大 |係数| だが、**個別設定で repair が単独最大になるのは 27/72 (38%)** に留まり、残りは zipf/split_increment/r_q が上回る。したがって本文は「(集計水準で) 最強の負予測子。設定単位では最強とは限らない」と明記し、無条件の「最強」主張は撤回する。

**Gemma saturation**:
- Gemma repair speed (relative): mean **17.1%** of layers (cos > 0.95 by layer 5.3 out of ~31)
- **100%** of Gemma words reach cos > 0.95 vs only **6.9%** for Llama and **17.9%** for Mistral
- Gemma pooled: repair_coef = **-77.0**, p = 0.0 (extreme due to near-zero variance)
- Cohen's d: Gemma = 0.457 (paradoxically largest despite saturation)

**Gemma の flip 弁別子**:
- Best Gemma predictor: repair_speed_95_rel (AUC = 0.615, d = 0.350)
- Gemma seg_middle (中間層 cos): flip = 0.991, noflip = 0.994, d = **-0.475** (最強弁別子)
- Llama/Mistral seg_middle: d = -0.134 / -0.134
- Gemma final layer drop: flip = 0.291, noflip = 0.267, d = 0.130, p = 1e-41

ナラティブ (Llama/Mistral): "Faster internal repair of the perturbed token predicts answer survival."
ナラティブ (Gemma): "Gemma repairs so quickly that conventional repair metrics saturate -- the discriminative signal shifts to subtle mid-layer dynamics."

---

### S6. Discussion (~1.0p)

---

#### S6.1 Model Family Effects (~0.5p) -- NEW section

- **この節の主張**: 三つの独立した分析（Mistral の delta_rho 反転 / Gemma の cos 飽和 / KL-R_C 空間分離の家族差）が収束し、typo 頑健性のメカニズムがモデル家族によって異なりうることを示唆する。**分母は 5 モデル・3 家族なので、これらは「相関」でなく「対比事例 (contrast cases)」として記述し、5 点回帰 (生態学的誤謬) は行わない**。断定形は避け、実験12・13 がサンプル・設定レベルの分布を与えるまでは「収束する対比事例に動機づけられた仮説」として扱う (discussion_family_effects.md の scope note と整合)。
- **使用する図表**: (S4.3, S5.1, S5.5 の図表を統合参照)
- **依存する実験番号**: Exp 4, Exp 3, Exp 9
- **紙幅目安**: 0.5p

構成:
1. **Mistral reversal**: Mistral は全25設定で delta_rho < 0 (pooled mean = -0.077)。Fixed-target で相関が強化される = 帰属構造が他家族と質的に異なる。仮説: Mistral は答えへの帰属を広く分散させ、答え変更時の再配分が小さい。
2. **Gemma saturation**: Gemma は層5 (全体の17%) までに cos > 0.95 に到達（100% の語）。修復が高速すぎるため token-level repair score は弁別力を失い、中間層の微細な dynamics (seg_middle, d=-0.475) のみが flip/noflip を分ける。
3. **KL-R_C spatial separation**: 家族間で分離強度に差。GSM8K で最強 (Wasserstein 0.40--0.47)。Importance 条件 > random perturbation。

統合的解釈: "Model families differ not just in *how much* they are affected by typos, but in *how* the perturbation propagates."

---

#### S6.2 Design Implications: A Four-Point Intervention Map (~0.4p)

- **この節の主張**: 統一 ERDC 連鎖(Encode–Repair–Divert–Carry と 3 モデレーター M1–M3; §6 前段で提示する統一モデル)は、防御を「手法の提案」ではなく「連鎖のどの段の漏れを塞ぐか」という**介入点の選択問題**へ再定式化する。我々は各段の漏れ量を初めて定量した(IE≈0.8 TE / DE≈0.2 TE / 修復ゲート通過率 M1)。その会計の上に、4つの介入点 T1–T4 に対し**本地図が予測する処方と限界**を導出する。**手法は提案せず、因果地図が示唆する検証可能な予測 (falsifiable predictions) として提示する**(手法論文の採点入口を作らない)。
- **使用する図表**: **Table 7**(4介入点マップ: 介入点 × 処方 × 根拠 × 予測される限界)。新規図は起こさず、既存 Fig. 2 (IE/DE)・Fig. 3 (削除)・Fig. 5 (校正器)・Fig. 7 (早期層 patch) を参照する(修正C: 1主張=1介入=1図を守る)。
- **依存する実験番号**: Exp 1 (IE/DE), Exp 2 (削除), Exp 3 (分岐点), Exp 7 (校正・γ), Exp 8/15 (早期層), Exp 14 (DE 正体, 事前登録), D1 (オラクル, 進行中)
- **紙幅目安**: 0.4p

**散文 (paste-ready, 6 文)**:

> Read through the ERDC chain, robustness is not a single technique but a choice of *which stage's leak to seal*, and our decomposition is the first to quantify those leaks: the carry path accounts for roughly 80% of the total effect (IE ≈ 0.8 TE), the read-out shortcut for the remaining ~20% (DE ≈ 0.2 TE), and the repair-gate pass rate (moderator M1) governs how much damage reaches the CoT at all. At the **input (T1)**, the map favors precise restoration over aggressive rewriting -- perfect byte-level restoration yields zero flips (0/45,641; Exp 7 within-run) and the conservative neural corrector beats the stronger LLM rewriter -- and it predicts that *detecting and prioritizing answer-relevant words*, whose restoration failures carry the residual gap (γ; Exp 7, Holm-significant in 17/25 pooled settings), will beat uniform correction, an upper bound Exp D1 is measuring. At **encoding (T2)**, because the damage localizes to an early-layer band (Exp 8, 12/12 conditions) that Exp 15 shows is causally sufficient to redirect free generation, any representation-level fix -- character-perturbation augmentation, a character-level auxiliary objective, or early-layer-only fine-tuning -- can be budgeted to that band rather than the full depth, at the predicted risk of over-fitting a known typo distribution (which the held-out *natural-typo* set is designed to expose). At **generation (T3)**, the CoT is diverted at only a few branch points (Exp 3), so branch-point self-verification, re-reading, or self-consistency should suppress the indirect path -- but by construction it leaves the read-out shortcut untouched (it lowers IE, not DE), and because verbalized self-correction marks *difficulty rather than successful repair* (H17 refuted: correction cues co-occur with flips, OR 2–3), such prompts must be validated behaviorally, not assumed. At **read-out (T4)**, the residual DE concentrates in MC formats and small models and carries a systematic first-option anchoring bias (Exp 8 answer-position patch; DE-flips over-select option A, Cramér's V ≈ 0.12–0.16, pooled p ≪ 0.001), so forcing answers to *reference the CoT* rather than matching options directly should curb it -- contingent on Exp 14 confirming DE's CoT-bypass-shortcut identity. The contribution is thus a *leak-accounting* of the chain plus a *map* of where to intervene: a set of predictions the map makes, not a defense we advocate.

**Table 7 — 4介入点マップ**:

| 介入点 (ERDC 段) | 処方 (本地図が予測する方向) | 本研究の根拠 | 予測される限界 |
|---|---|---|---|
| **T1** 入力 / pre-encode | 高精度校正、特に**答え関連語を優先復元** | byte 復元 → flip 0/45,641・neural > LLM(強さでなく正確さ)・校正器の相補性 | 失敗が高 R_Q 語に集中(γ: Holm有意 17/25, ただし AUC≈0.54 と効果小)なら重要語検出→重点復元が正解(D1 のオラクル上限で検証) |
| **T2** 符号化 / S1+G | typo 不変な語彙表現(文字摂動拡張・文字補助目的・**早期層限定微調整**) | 実験8 早期層局在 **12/12**(全層不要=予算削減根拠)・実験9 修復能力のモデル差・実験15 patch→自由生成で早期層の**因果十分性** | 拡張は既知 typo 分布への過適合リスク(自然 typo held-out が試金石) |
| **T3** 生成 / S2+S3 | 分岐点自己検証・再読・self-consistency | 実験3 逸脱の**少数位置集中**・R1 の高重要度攻撃減衰(表現的修復 M1) | **IE は減らせても DE には効かない**;明示的自己訂正は困難の徴候で修復ではない(H17 反証)ため行動的検証が必須 |
| **T4** 読み出し / 残余 DE | MC で選択肢-質問の直接マッチ抑制、**CoT 参照強制** | DE が **MC × 小型に偏在**・実験8 答え位置 patch・B6 第1選択肢アンカリング(V≈0.12–0.16, 11/16 セルで A 過選択) | 実験14 で DE の**ショートカット正体確定**が前提 |

**検証可能な予測 (future work; falsifiable)** — 訓練系・検出系の頑健化手法そのものは提案せず、本地図が予測する有望方向として登録する。各予測は本文と同一の推定量 (flip / IE / restore) で反証可能:

> - **P(T2-train)**: early-layer-only perturbation fine-tuning attains the robustness of full-depth fine-tuning with *less catastrophic forgetting* (grounded in the early-layer localization of Exp 8 and its causal sufficiency in Exp 15).
> - **P(T1-detect)**: answer-relevant-word detection followed by targeted restoration attains *equal-or-greater flip suppression with fewer edits* than uniform correction (grounded in the R_Q concentration of the residual gap, γ; oracle ceiling to be set by Exp D1).
> - **P(T3-branch)**: self-verification restricted to the *few early branch points* identified in Exp 3 attains the IE suppression of full-trace verification (grounded in the positional concentration of divergence).
>
> These are predictions of the causal map, not results; we state them so that a later study can refute them.

**D1 依存の注記**: T1 の「答え関連語優先復元が汎用校正を上回る」の実証根拠は D1(オラクル復元実験、進行中)の数値で確定する。D1 完了までは γ(H7-4: R_Q 偏在、Holm有意 17/25、効果小)を根拠として「高 R_Q 語に残余が偏る確率的傾向」までを主張し、オラクル上限の数値は **[D1 で検証予定]** のプレースホルダとする。D1 が出れば §5/T1 の根拠は「予測」から「実証」に格上げされる。

---

#### S6.3 Why Some Typos Succeed: The Repair Mechanism (~0.2p)

- **この節の主張**: 内部修復メカニズム（Exp 9）が typo の成否を説明する。修復が速い語は flip しにくく、修復が不完全な語が CoT を経由して答えを破壊する。
- **使用する図表**: (S5.5 の図表を参照)
- **依存する実験番号**: Exp 9, Exp 1
- **紙幅目安**: 0.2p

構成:
- Repair coefficient (Llama+Mistral pooled): -0.977 -- a one-SD increase in repair speed reduces flip odds by ~62%
- Connection to Exp 1: IE dominance means the repair failure propagates *through the CoT*
- Gemma: despite rapid repair, flip still occurs -> the mechanism is more nuanced (final-layer perturbation dynamics)

---

### S7. Limitations (~0.5p)

- **この節の主張**: 本研究の射程と制約を明示し、残余の脅威を honest に記述する。
- **使用する図表**: なし
- **依存する実験番号**: 全体
- **紙幅目安**: 0.5p

#### 箇条書き:

1. **Gemma cos saturation and ceiling effect**: Gemma の修復スコアは 0.99 近傍に飽和し、token-level の repair score が flip 弁別力を失う。中間層 dynamics (seg_middle) で代替しているが、saturation の原因自体は未解明。

2. **R1-distillation greedy vs official recommendation gap**: Reasoning model の評価を greedy decoding (temperature=0) で実施。一部のモデルの公式推奨は temperature > 0 であり、greedy 設定での性能は過小推定の可能性がある。

3. **Span extraction failure rate**: Union 除外率が全体 13.19%（最大 38.59% = Mistral x GSM8K）。分析対象が clean・typo 双方で答えスパン抽出に成功したサンプルに限定されるため、抽出困難な長文回答での typo 影響は過小評価の可能性。

4. **Faithfulness residual (Turpin-type)**: CoT に明示されない要因が答えを曲げる不忠実さ問題。実験 1 (DE の定量化) と実験 3 (KL-R_C の行動的分岐点との一致) で大幅に狭まるが、完全には排除できない。「本研究が示すのは答え段の CoT 依存性であり、CoT が計算過程を網羅する保証ではない」。

5. **Representation analysis limited to M3 x B2**: Activation patching (Exp 8) と内部修復分析 (Exp 9) は計算コストにより M3 (3 models) x B2 (2 benchmarks) に限定。M5 x B5 の全カバレッジではない。

6. **単一 seed・greedy・単一テンプレート**: 全実験は凍結条件 (seed=42, greedy decoding (temperature=0.0), 単一の "Let's think step by step" テンプレート) の下で実施した。したがって主結果はサンプリング分散・プロンプト言い換えに対する感度を評価していない。実験1の見出し (restore 率・IE 優位) については temperature=0.7 x 3 seed での再現を別途スポット実施予定であり、本文の結論は greedy・単一 seed 条件に限定して読まれるべきである。

7. **英語・文字レベル摂動のみ**: 摂動は英語テキストへの文字レベル操作 (入替・削除・挿入; LXT-4/Random-4) に限定する。語レベルの誤り (実在語への置換・語順・文法誤り)、および多言語 (非英語・非ラテン文字) はスコープ外であり、本研究の結論をこれらに外挿しない。

---

## Appendices

### Appendix A: Hypothesis Registry (H1--Hn)

- 各実験の事前仮説と事後的検証結果の対照表
- 事前予測が裏切られたケースの明示（特に Exp 4 の Mistral 反転、Exp 7 の LLM 劣位）

### Appendix B: Full delta_rho Table (25 Settings)

- Exp 4 の全 25 設定 x {rho_default, rho_fixed, delta_rho, CI, Holm p}
- Fig. 4 (本文 heatmap) の元データ

### Appendix C: Matched Control Balance Tables

- Exp 5 の全 25 設定の層化マッチバランス表
- 5 変数 (class, char_len, zipf, split_increment, centrality) の SMD

### Appendix D: Divergence Profiles

- Exp 3 の全設定 KL 発散プロファイル
- Position-wise KL density, onset detection, precision@k 詳細

### Appendix E: Corrector Restoration Details

- Exp 7 の校正器別・設定別の {token restoration rate, byte-identical rate, miscorrection rate, R_Q failure concentration}
- Mann-Whitney 検定結果

### Appendix F: Case Studies

- 代表的な flip 事例の walkthrough
- typo -> CoT 変化 -> 答え反転の具体例（GSM8K の数値伝播、MMLU の選択肢シフト）

---

## 図表リスト（確定）

| ID | 内容 | 節 | 実験 |
|----|------|-----|------|
| Fig. 1 | 三層フレームワーク概念図 + 因果グラフ | S2 | - |
| Fig. 2 | **Headline**: restore 率 (primary) 棒グラフ + TE/DE/IE セル別リスク ヒートマップ | S4.1 | Exp 1 |
| Fig. 3 | Deletion effect by token stratum (numeric vs content) | S4.2 | Exp 2 |
| Fig. 4 | **Family x format heatmap** of delta_rho | S4.3 | Exp 4 |
| Fig. 5 | 3-panel corrector comparison | S4.4 | Exp 7 |
| Fig. 6 | KL vs R_C positional density plot | S5.1 | Exp 3 |
| Fig. 7 | Layer x site recovery heatmap (activation patching) | S5.4 | Exp 8 |
| Fig. 8 | Repair score vs flip, by family | S5.5 | Exp 9 |
| Table 1 | M5 x B5 coverage matrix | S3 | Step 0 |
| Table 2 | Pooled restore (primary) + TE/DE/IE cell-risks + descriptive IE/TE + sensitivity | S4.1 | Exp 1 |
| Table 3 | Deletion flip rates by target x dose | S4.2 | Exp 2 |
| Table 4 | delta_rho summary (25 settings) | S4.3 | Exp 4 |
| Table 5 | Accuracy by corrector x model x benchmark | S4.4 | Exp 7 |
| Table 6 | Matched control summary (22/25 significant) | S5.2 | Exp 5 |
| Table 7 | Four-point intervention map (T1–T4: prescription × basis × predicted limit) | S6.2 | Exp 1/2/3/7/8/15 |

---

## 実験と節の対応マップ

| 実験 | 本文での位置 | 役割 | 完了状態 |
|---|---|---|---|
| Step 0 | S3 | 基盤 | 完了 |
| Exp 1 | S4.1 (主柱) | 因果分解 | 完了 |
| Exp 2 | S4.2 (主柱) | 標的削除 | 本番実行中 |
| Exp 3 | S5.1 (支持) | KL-R_C 相補性 | 完了 |
| Exp 4 | S4.3 (主柱) | fixed-target 診断 | 完了 |
| Exp 5 | S5.2 (支持) | マッチド統制 | 完了 |
| Exp 6 | S5.3 (支持) | 帰属手法比較 | LOO スモーク完了 |
| Exp 7 | S4.4 (主柱) | 校正器ラダー | 本番実行中 |
| Exp 8 | S5.4 (支持) | activation patching | スモーク完了 |
| Exp 9 | S5.5 (支持) + S6.1 | 内部修復 | 完了 |

---

## 執筆上の注意

1. **本文 4 本柱 (S4) の順序**: 因果分解 (最強) -> 削除介入 (新発見) -> fixed-target (診断) -> 校正器 (実用) の順。因果的証拠の強い順に並べる。
2. **S5 supporting evidence の扱い**: 各 0.2--0.3p で簡潔に。数値は 1--2 個に絞り、詳細は付録送り。
3. **Discussion S6.1 (Model Family Effects)**: 完全に新設の節。S4.3 (Mistral reversal), S5.1 (KL-R_C separation), S5.5 (Gemma saturation) の三つの独立した証拠を統合する。
4. **Faithfulness の書き分け**: Introduction の Why-CoT 段落は「因果的入力としての CoT」の積極的正当化。Limitations は Turpin 型不忠実さの残余の明示。役割を分けて重複させない。
5. **数値の引用**: Phase A 確定数値のみ本文に記載。進行中の実験 (Exp 2 本番、Exp 7 全設定、Exp 8 本番) はスモーク値を使い、本番完了後に差し替え。
