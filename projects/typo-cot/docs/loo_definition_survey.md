# Leave-one-out(削除ベース)単語重要度の定義:文献調査メモ

作成日: 2026-07-14
目的: CoT テキスト中の語の LOO 重要度(削除→答えトークン列の teacher-forcing log-prob 低下)における「削除単位=語タイプ定義」の選択を、先行研究の慣行に照らして整理する。

調査方法: WebSearch + 原論文(arXiv/ar5iv/ACL Anthology)の本文確認。本文まで確認できた項目と、検索結果・アブストラクトのみで確認した項目を区別して記す。

---

## 1. 主要先行研究の一覧(削除単位・方法・全出現 vs 単一出現・評価指標)

### 1.1 削除/遮蔽ベースの重要度・忠実性評価(NLP)

| 論文(年・会議) | 削除単位 | 削除方法 | 全出現 vs 単一出現 | 指標 |
|---|---|---|---|---|
| Li, Monroe & Jurafsky, "Understanding Neural Networks through Representation Erasure" (2016, arXiv:1612.08220、**査読会議への採録は確認できず**) | 入力語(および語ベクトル次元、中間ユニット、句) | **語は系列から削除**(次元はゼロ化) | **単一出現ごとに消去**。語彙(タイプ)レベルの重要度は、正解ラベル対数尤度の相対差 (S(e,c) − S(e,c,¬d))/S(e,c) を**コーパス中の全出現で平均**して定義 | 対数尤度の相対低下、決定フリップ(RL で最小消去集合) |
| Feng et al., "Pathologies of Neural Models Make Interpretations Difficult" (EMNLP 2018) | 語 | 削除(input reduction: 重要度最小の語を逐次除去) | 単一出現(位置ごと) | 予測確信度の低下量 |
| Jain & Wallace, "Attention is not Explanation" (NAACL 2019) | トークン(単一要素) | 除去(leave-one-out) | 単一出現 | 出力分布の変化(TVD)、ランキング間の Kendall τ 相関 |
| Serrano & Smith, "Is Attention Interpretable?" (ACL 2019) | **中間表現**(attention 重み1個または集合) | attention 重みのゼロ化+再正規化(テキスト編集ではない) | 単一位置(および集合) | 決定フリップまでの消去数 |
| DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models" (ACL 2020) | **トークン**(rationale スパンを構成するトークン列。連続スコアは top-k_d トークンで離散化) | **入力からの削除**(comprehensiveness = m(x_i)_j − m(x_i\r_i)_j。x_i\r_i は「rationale を取り除いた入力」)。sufficiency は rationale のみ残す | 位置ベース(rationale に含まれる位置を削除。タイプ一括の概念なし) | comprehensiveness / sufficiency(予測確率差)、AOPC(top 1/5/10/20/50% の5ビン平均) |
| Atanasova et al., "A Diagnostic Study of Explainability Techniques for Text Classification" (EMNLP 2020) | **語**(サブワード・埋め込み次元のスコアを L2 ノルムまたは平均で語へ集約してからランク付け) | **マスク**(saliency 降順に 0–100% を 10% 刻みでマスク) | 位置ベース(スコア上位から k% を対象) | AUC-TP(閾値-性能曲線の面積) |
| Kim et al., "Interpretation of NLP models through input marginalization" (EMNLP 2020) | トークン | **置換**: ゼロ埋めや削除は OOD になると批判し、BERT の [MASK] 予測分布で**周辺化**(marginalization) | 単一出現 | 予測確率の変化(周辺化された削除効果) |
| Hase, Xie & Bansal, "The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations" (NeurIPS 2021) | 特徴(トークン)集合 | 削除は OOD を生むため、**Attention Mask または MASK トークン置換を推奨**(Counterfactual Training なしの場合) | 位置ベース | 予測確信度変化ほか |
| Hooker et al., "A Benchmark for Interpretability Methods in Deep Neural Networks"(ROAR)(NeurIPS 2019) | 画像ピクセル/特徴(**視覚領域**) | **無情報値(平均値)への置換**+**再訓練** | 位置ベース(top-k%) | 再訓練後のテスト精度低下 |
| Krishna et al., "The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective" (2022, arXiv:2202.01602) | (削除手法ではない)説明間一致の指標定義 | — | — | **feature agreement@top-k**(2説明の top-k 特徴集合の共通割合)、rank agreement 等 → 我々の Jaccard@10 と同型の慣行 |
| Achtibat et al., "AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers" (ICML 2024) | トークン(および潜在表現)への帰属 | (帰属手法。削除ではない) | 出現(位置)ごとにスコア | 帰属の忠実性(perturbation 評価) |
| "When LRP Diverges from Leave-One-Out in Transformers" (BlackboxNLP @ EMNLP 2025, arXiv:2510.18810) | LOO を帰属の参照(ground truth 的基準)として使用 | (本文の実装詳細は今回未確認) | — | LRP と LOO の整合性。**AttnLRP の bilinear 伝播が implementation invariance 公理を破ると指摘** — AttnLRP と LOO の突合という我々の設定に直接関連 |

注: Zeiler & Fergus (ECCV 2014) の occlusion(灰色パッチ置換のスライディングウィンドウ)、Samek et al. (IEEE TNNLS 2017) の region perturbation / AOPC(MoRF 順に領域を置換)は視覚領域の定番だが、今回セッションでは本文未確認のため概要のみ記載。

### 1.2 CoT の一部を削除・切断・摂動する研究(単位に注目)

| 論文(年・会議) | 摂動単位 | 摂動方法 | 指標 |
|---|---|---|---|
| Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning" (2023, arXiv:2307.13702。Anthropic テクニカルレポート、**査読会議採録は確認できず**) | **文**(NLTK punkt で CoT を文分割し、各文境界で切断) | Early answering: CoT を途中で**切断**(truncation)。Adding mistakes: 1つの文(ステップ)を誤り版に**置換**し以降を再生成 | 完全 CoT と同じ結論に至る頻度(答えの変化率) |
| Turpin et al., "Language Models Don't Always Say What They Think" (NeurIPS 2023) | (CoT ではなく**入力プロンプト**にバイアス特徴を付加。CoT 削除はしない) | 入力摂動(選択肢並べ替え等) | 予測変化・説明中でのバイアス言及の有無 |
| Bao et al., "How Likely Do LLMs with CoT Mimic Human Reasoning?" (COLING 2025, arXiv:2402.16048) | **CoT 全体**(SCM の1変数として扱う) | CoT 全体を golden CoT / random CoT(数値ランダム置換、論理式否定化)に**置換** | 答えへの平均処置効果(McNemar 検定) |
| Bogdan et al., "Thought Anchors: Which LLM Reasoning Steps Matter?" (2025, arXiv:2506.19143) | **文**(reasoning trace を文単位で扱う) | ①その文から**リサンプリング**(意味の異なる代替文で置換し以降を再生成)②attention 集約 ③特定文への attention 抑制(suppression) | 最終答え分布への反実仮想的影響 |
| "Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps"(FUR)(EMNLP 2025) | **推論ステップ** | ステップの情報を**パラメータから unlearning**(テキスト編集ではなくパラメトリック消去) | 予測への影響(parametric faithfulness) |

**要点**: CoT 忠実性研究の摂動単位は「文・ステップ・CoT 全体」が支配的で、**語・トークン単位で CoT を削除して重要度を測る確立した先行例は今回の調査では確認できず**(gradient 系帰属を CoT に適用した研究はあるが、削除ベースの語単位 LOO は見当たらなかった)。→ 我々の実験は「古典的な語単位 erasure(Li et al. 2016 系)を CoT に適用する」位置づけになる。

### 1.3 削除で文法・分布が壊れる問題への対処(削除 vs マスク vs 置換)

- **削除(token dropping)**: ERASER comprehensiveness、Li et al. 2016、Feng et al. 2018、Jain & Wallace 2019。もっとも単純で慣行として強いが、OOD 入力を生む。
- **マスク/置換**: Atanasova et al. 2020(マスクトークン)、Hase et al. 2021(**Attention Mask か MASK トークン置換を推奨**)、Kim et al. 2020(BERT による周辺化)、ROAR(平均値置換+再訓練)。
- ただし MASK 系は **[MASK] を持つ encoder(BERT 系)前提**の対処であり、decoder-only LLM には [MASK] 相当がない。decoder-only での対応は (a) 削除のまま(OOD を許容し、全変種が同じバイアスを受けると仮定)、(b) attention マスクで該当位置を遮蔽、(c) 中立的なプレースホルダ文字列への置換、のいずれかになる。CoT 忠実性研究(Lanham、Thought Anchors)は文単位のため削除・置換でも文法が壊れにくく、この問題を単位の粗さで回避している。

### 1.4 「全出現一括 vs 単一出現」について確認できたこと

- **単一出現(位置ごと)の削除が標準**: Li et al. 2016、Jain & Wallace 2019、ERASER、Atanasova et al. 2020 はすべて位置ベース。
- ただし Li et al. 2016 は**語タイプの重要度を「出現ごとの消去効果の平均」として定義**しており、「タイプレベルの重要度 = 出現レベル消去の集約」という先例がある(ただし集約はコーパス横断であり、1テキスト内の全出現一括削除ではない)。
- **1テキスト内で同一タイプの全出現を同時に削除して1変種とする LOO の直接の先例は今回の調査では確認できず**。近い慣行としては、数理推論の摂動ベンチマーク(例: GSM-Symbolic, Mirzadeh et al. 2024, arXiv:2410.05229)が変数(数値・名前)を**テンプレート経由で全出現一貫して置換**する、対実データ作成で実体の全言及を一貫置換する、といった「介入の一貫性(consistency)」の考え方がある(これらは重要度測定ではなく頑健性評価/データ作成の文脈)。
- 単一出現削除には**冗長性バイアス**がある: CoT 内で同じ数値が複数回言及される場合、1出現を消しても他の出現から情報が回復できるため、LOO 低下がほぼ 0 になり「重要でない」と誤判定しうる。Thought Anchors が単純削除でなく「リサンプリング+以降の再生成」を使う動機も、後続に情報が再出現する問題への対処である。全出現一括削除はこの冗長性を遮断でき、「その値が CoT 中のどこからも得られない」という反実仮想に対応する。

---

## 2. 我々の制約に照らした選択肢

制約: (i) AttnLRP 側 R_C(サブワード→語マージした word_scores)との Jaccard@10 互換、(ii) teacher-forcing での答えトークン列 log-prob 差、(iii) テキスト編集ベース(サブワード削除は定義不能)。

### 案 A(現行): 空白区切り語タイプ・全出現一括削除
- 内容: 端の句読点を剥がした空白区切り語タイプごとに、全出現を一括削除して1変種。タイプをキーに Jaccard@10。
- 利点:
  - 変種数 = タイプ数で最少(計算コスト最小)。
  - タイプキーのランキングが R_C(語ランキング)とそのまま突合可能。
  - 冗長に再言及される数値・演算対象について「その情報を CoT から完全に奪う」介入になり、数理 CoT では意味的に自然(GSM-Symbolic 的な一貫介入と同型)。
  - 削除という操作自体は ERASER comprehensiveness・Li et al. の慣行に一致。
- 欠点:
  - **1テキスト内全出現一括削除の直接の先例が確認できず**、査読で「非標準」と突かれうる。
  - 高頻度タイプほど介入が大きく(削除トークン数が多く)、log-prob 低下が頻度と交絡する。
  - 削除による文法崩壊・OOD(Hase et al. 2021 の批判)が最大。

### 案 B: 出現(位置)ごとの削除 → 語タイプへ集約(平均または最大)
- 内容: 各出現を1つずつ削除して log-prob 低下を測り、タイプの重要度 = 出現スコアの平均(Li et al. 2016 準拠)または最大。タイプランキングで Jaccard@10。
- 利点:
  - **先行研究の慣行に最も忠実**(位置単位削除: ERASER/Jain & Wallace/Feng et al.、タイプ集約 = 出現平均: Li et al. 2016)。
  - 介入サイズが常に1語で頻度交絡がない。OOD の程度も最小。
  - 「サブワード→語」マージ(R_C 側)と「出現→タイプ」集約(LOO 側)が対称的な設計として説明しやすい。
- 欠点:
  - 変種数が出現数に増える(ただし CoT では大半のタイプが1出現なのでコスト増は限定的なはず。要実測)。
  - **冗長性バイアス**: 反復言及される数値の重要度を系統的に過小評価。
  - 集約関数(mean/max)という自由度が増える。R_C 側が同一タイプ複数出現をどう1スコアに畳むかと**揃える必要**がある(揃っていないと Jaccard@10 が単位の不一致を測ってしまう)。

### 案 C: 削除でなく遮蔽・置換(attention マスク or プレースホルダ)
- 内容: 語のトークン位置を teacher-forcing 時に attention マスクで遮蔽する、または中立プレースホルダへ置換。
- 利点:
  - OOD・文法崩壊への対処として Hase et al. 2021 が明示的に推奨する形(Attention Mask / MASK 置換)。
  - attention マスク版はトークン整列が完全に保たれ、原理的には**サブワード(トークン)レベルで AttnLRP と完全整合**した比較すら可能になる(「テキスト編集では揃えられない」制約自体を解消)。
- 欠点:
  - もはや「テキスト編集ベースの LOO」ではなく実装が別物(位置 ID・KV キャッシュとの相互作用も検証要)。「編集ベースの単純さ・帰属フリー」という売りが弱まる。
  - decoder-only に [MASK] はなく、プレースホルダ文字列は独自の意味を持ち込む。
  - 実装変更の波及が大きい。

### 案 D: 文・ステップ単位 LOO(補助分析)
- 内容: Lanham(文単位切断)、Thought Anchors(文単位リサンプリング)、FUR(ステップ単位)に合わせ、文単位の削除重要度も併記。
- 利点: CoT 忠実性文献の単位慣行と最も整合し、文法崩壊もない。
- 欠点: **語ランキング R_C との Jaccard@10 が定義できない**ため主実験の代替にはならない。あくまで補助。

---

## 3. 推奨案

**推奨: 案 B(出現ごと削除 → タイプへ集約)を主定義とし、案 A(全出現一括)を「タイプレベル消去」の感度分析として併記する。ただし計算資源・スケジュール制約が強い場合は、案 A を主のまま以下の3点をパラグラフで明示すれば防御可能。**

根拠(引用付き):
1. **削除という操作**は erasure 系の標準そのもの: ERASER の comprehensiveness は「予測 rationale を入力から取り除いた x_i\r_i」との予測差で定義され(DeYoung et al., ACL 2020)、語の消去による重要度定義の原型は Li, Monroe & Jurafsky (2016, arXiv:1612.08220) の「正解ラベル対数尤度の相対低下」。我々の「答えトークン列 log-prob 低下」はこの系譜に正確に乗る。
2. **単位は位置(単一出現)が標準**であり、**語タイプの重要度は出現レベル消去の集約として定義するのが Li et al. (2016) の先例**(語彙レベル重要度 = 出現ごとの対数尤度相対差の平均)。1テキスト内での全出現一括削除を1変種とする先例は今回の調査では確認できなかったため、案 A を主とするなら「type-level erasure(冗長な再言及を遮断する反実仮想)」として明示的に定義・正当化する必要がある(冗長性バイアスの議論は Thought Anchors のリサンプリング動機と同根で説明できる)。
3. **語単位(サブワードから語への集約)**は Atanasova et al. (EMNLP 2020) がサブワード・埋め込みスコアを L2/平均で語に集約してから摂動評価しており、慣行として確立。**top-k 集合の重なりでランキングを比較する**のも Krishna et al. (2022) の feature agreement@top-k と同型で標準的。よって「空白区切り語タイプ + Jaccard@10」という枠組み自体は先行慣行と整合しており、争点は全出現一括か出現集約かのみ。
4. 削除の OOD 批判(Hase et al., NeurIPS 2021; Kim et al., EMNLP 2020)には、(a) 本研究は絶対的重要度でなく**2手法のランキング一致**を見るため OOD バイアスは両者に共通にかからない(LOO 側のみの系統誤差は1語削除で最小化される=案 B の利点)、(b) decoder-only LLM に [MASK] 相当がない、の2点で応答できる。attention マスク版(案 C)は「トークンレベルで完全整合な比較が原理的に可能」という将来課題として脚注に置く価値がある。
5. 実務上の注記: 案 B を採る場合、R_C 側で同一タイプ複数出現のスコアをどう畳むか(sum/mean/max)と LOO 側の集約を**同じ関数に揃える**こと。また R_C の word_scores が改行をまたいで語を結合する既知の癖("dollars.\nThe")は、Jaccard@10 が単位不一致を「不一致」として数えてしまうため、比較の公平性の観点から先に修正(または LOO 側と同一のセグメンタを共有)すべき。

関連して要注意の新しい先行研究: "When LRP Diverges from Leave-One-Out in Transformers" (BlackboxNLP @ EMNLP 2025, arXiv:2510.18810) は AttnLRP の bilinear 伝播が implementation invariance を破り LOO と乖離しうると指摘している。我々の「AttnLRP ランキングと LOO ランキングの Jaccard@10」はまさにこの整合性を測る実験なので、関連研究として引用でき、一致度が中程度に留まる場合の解釈(帰属手法側の公理的限界)にも使える。

---

## 4. 確認できなかったこと(明示)

- 1テキスト内の「同一語タイプ全出現の一括削除」を重要度の1変種とする論文: **確認できず**。
- CoT を「語・トークン単位で削除」して重要度ランキングを作る削除ベースの先行研究: **確認できず**(文・ステップ・CoT 全体単位が支配的)。
- Li et al. 2016 の査読会議採録: arXiv のみ確認(**会議採録は確認できず**)。Lanham et al. 2023 も arXiv(Anthropic レポート)のみ確認。
- "When LRP Diverges from LOO"(2510.18810)の LOO 実装詳細(削除かマスクか): アブストラクトからは**未確認**。引用前に本文確認を推奨。
- Zeiler & Fergus 2014 / Samek et al. 2017 の記述は一般に知られた内容であり、今回セッションでは原文未確認。

## 5. 参照 URL

- ERASER: https://aclanthology.org/2020.acl-main.408/ / https://arxiv.org/abs/1911.03429
- Li et al. 2016: https://arxiv.org/abs/1612.08220
- Feng et al. 2018: https://aclanthology.org/D18-1407/
- Jain & Wallace 2019: https://aclanthology.org/N19-1357/
- Serrano & Smith 2019: https://arxiv.org/abs/1906.03731
- Hooker et al. 2019 (ROAR): https://arxiv.org/abs/1806.10758
- Atanasova et al. 2020: https://aclanthology.org/2020.emnlp-main.263/ / https://arxiv.org/abs/2009.13295
- Kim et al. 2020: https://aclanthology.org/2020.emnlp-main.255/
- Hase et al. 2021: https://arxiv.org/abs/2106.00786
- Krishna et al. 2022: https://arxiv.org/abs/2202.01602
- Lanham et al. 2023: https://arxiv.org/abs/2307.13702
- Turpin et al. 2023: https://arxiv.org/abs/2305.04388
- Bao et al. (COLING 2025): https://aclanthology.org/2025.coling-main.524/ / https://arxiv.org/abs/2402.16048
- Thought Anchors 2025: https://arxiv.org/abs/2506.19143
- FUR (EMNLP 2025): https://arxiv.org/abs/2502.14829 / https://aclanthology.org/2025.emnlp-main.504/
- When LRP Diverges from LOO (BlackboxNLP 2025): https://arxiv.org/abs/2510.18810
- AttnLRP (ICML 2024): https://proceedings.mlr.press/v235/achtibat24a.html
- GSM-Symbolic: https://arxiv.org/abs/2410.05229
