rebuttal本文の構成と追加実験を、スコアを動かす期待値順に整理します。short paperのrebuttal期間(通常1週間)を想定し、実験は「回せる規模」に絞っています。

---

## 優先度1:Endogeneity反証実験(AxQH Weakness 2)→ AxQH 2.5→3の鍵

**指摘の核心**:CoT:Jaccard@kのR_Cはanswer spanへの帰属で定義されるため、答えがflipするとattributionのターゲット自体が変わる。よってJaccard@kは「flipの予測因子」なのか「flipの結果」なのか区別できない。

**追加実験:Fixed-Target Attribution**
- 設計:perturbed条件でも、**摂動前の正解answer span(の文字列)を固定ターゲット**としてR_Cを再計算。生成CoT中に同一answer phraseが存在しない場合は、モデルに正解フレーズを強制デコードさせた時点での帰属、またはターゲット語彙(例:"B"や"18")のlogitへの帰属で代替
- 規模:Gemma-3-4B + Llama-3.2-3B × MMLU + GSM8K、k=4のみで十分
- 検証:固定ターゲット版CoT:Jaccard@kでも答えflipとの負の偏相関が保持されるか
- 期待される主張:「ターゲットを固定してもInternal軸の関連は保持される → endogeneityは主結論を覆さない」

**結果別のrebuttal文言を2パターン用意**:
- 保持された場合:「新実験により、指摘のendogeneityは我々の結論に影響しないことを確認した」と正面反証
- 弱まった場合:「一部endogeneityの寄与を認め、本文でJaccard@kを『diagnostic signal(予測因子とは主張しない)』と再定義する」と誠実に後退。これでも「指摘を実験で検証した」事実がconf 4レビュアーには効きます

---

## 優先度2:Causal Languageの全面改稿提示(3名共通)→ コストゼロ・全員に効く

実験不要。rebuttal内に**before/after対応表**をそのまま貼るのが最重要ポイントです。約束ではなく現物を見せます。

| 現行 | 改稿後 |
|---|---|
| "pathways from typos to answer errors" | "axes associated with answer errors" |
| "interventions on internal representations **are necessary**" | "**motivate** future interventions on internal representations" |
| "superficial denoising alone **is insufficient**" | "results **suggest potential limitations** of relying only on surface-level denoising" (← N5Yqの提案文言をそのまま採用) |
| "causes degradation" | "is associated with degradation" |
| "Internal pathway" | "**attribution-based preservation of answer-relevant CoT tokens**" (← AxQHの提案文言をそのまま採用) |

タイトルの "Superficial and Internal **Pathways**" も "Two **Axes**" 系に変更する意思を明示。**レビュアー自身が提案した文言を採用する**のは、「あなたの指摘を正確に理解した」というシグナルとして最も強く、抵抗なくスコアを上げやすくします。

あわせてAxQHのComments対応:内部軸の定義明確化(hidden statesではなくAttnLRP帰属ベースであることをSection 2.3冒頭で明示)、Jaccard@kのターゲットが条件ごとに計算される旨の明記。

---

## 優先度3:Denoisingベースライン実験(N5Yq Weakness 1)→ N5Yq 3→3.5の鍵

**指摘の核心**:「surface-level denoisingでは不十分」と主張しながら、spell correction等のdenoisingベースラインを一切評価していない。

**追加実験:Spell-Correction Restoration**
- 設計:LXT-4のperturbed入力に既製のスペル訂正(pyspellchecker / SymSpell / neuspell等)を適用 → 復元後入力で再推論 → (a)入力の復元率、(b)残存するanswer flip率、(c)復元後CoTのCoT:Jaccard@k を測定
- 規模:1モデル(Gemma-3-4B)× 2ベンチマーク(MMLU, GSM8K)で十分
- 期待される結果と主張:
  - 復元が不完全(固有名詞や文脈依存語で訂正ミス)→「入力復元自体に限界」
  - 完全復元されたサブセットでもCoTが別framingで再生成されflipが残存 →「ケーススタディ(Sun→Sjn)の定量版」として最強の補強
- 万一「denoisingでほぼ全部直る」結果でも、「訂正器が及ばない実運用条件(多言語・OCR等)でInternal軸の診断が必要」と限定付きで再主張可能。ただしこの場合は結論軟化(優先度2)とセットで整合させる

これはN5Yqが着地点として示した「結論軟化」を超えて**証拠を足す**対応なので、+0.5を狙える唯一の材料です。

---

## 優先度4:AttnLRP交絡コントロール(AxQH Weakness 3 / R1 Comment 2)

**指摘の核心**:LXT vs Randomの差は、高relevanceトークンが「長い・低頻度・内容語」という交絡で説明できるかもしれない。AttnLRP自体の妥当性も未検証。

**追加実験(小規模):Matched-Random対照**
- 設計:Random条件を「LXT-4が選んだトークンと**語長・品詞(内容語/機能語)をマッチさせた**ランダム選択」に置換し、accuracy dropを再比較。1モデル×2ベンチマーク
- LXTの優位が残れば「交絡だけでは説明できない」と反証。差が縮んでも「AttnLRPは内容語選択+αの情報を持つ」と限定的に主張し直せる
- あわせてAppendix Dの定性比較(LXT-4 vs Anti-LXT-4)を参照し、選択トークンの意味役割の差を再掲

AttnLRPの限界(faithfulness問題)については反証不能なので、**Limitationsに「attribution手法依存性」の段落を追加し、leave-one-out等との比較を明示的にfuture workとする**ことを約束。R1のComment 2(AttnLRPの限界の明確化)と同時にカバーできます。

---

## 優先度5:R1「incremental / whyがない」への位置づけ反論 → 実験なし・印象管理

スコア変動は期待せず、**ACが読むこと**を意識して書きます。

- 「入力摂動とCoT変化の相関が弱い(|ρ|<0.3)」は事前には自明でない:素朴な仮説は「typoがCoTに文字通り伝播する」であり、それを棄却したのが本研究の発見。AxQHも "the main empirical observation ... is interesting" と評価している(←他レビュアーの言葉を引用して間接反論)
- 「why」の解明には介入実験(activation patching等)が必要で、それは本diagnostic研究が**分解した2軸があって初めて設計できる**。R1の提案(causal tracing)はまさに我々のfuture workであり、その前提整備が本論文の貢献
- 丁寧に、しかし卑屈にならず。conf 4なので論破は狙わない

---

## 優先度6:細部対応(コスト5分・落とすと印象減点)

- "We analyzes" → "We analyze"、"questioner side" → "question side"、Figure 3参照のフォーマット修正を「修正済み」と報告
- N5Yqのスコープ指摘(多言語・大規模モデル・self-consistency)→ Limitationsに1段落追加
- AxQHの詳細要求(プロンプト、デコード設定、シード、フィルタ率のモデル×データセット別報告)→ Appendixに追加すると明記。**フィルタ統計は実際に集計して数値をrebuttalに載せる**(手元のログから出せるはずで、誠実さの証明になる)

---

## 実行プランまとめ

| 実験 | 対象 | 規模 | 目安工数 |
|---|---|---|---|
| ①Fixed-Target Attribution | AxQH | 2モデル×2ベンチ | 2〜3日 |
| ③Spell-Correction復元 | N5Yq | 1モデル×2ベンチ | 1〜2日 |
| ④Matched-Random | AxQH/R1 | 1モデル×2ベンチ | 1日 |
| フィルタ統計集計 | AxQH | ログ集計のみ | 半日 |

GPU時間が足りない場合は**①>③>④の順で切り捨て**。①だけは死守してください(conf 4のborderline 2.5を動かせる唯一の材料なので)。

rebuttal文面は「共通指摘(causal language)への回答を冒頭に1回書き、各レビュアー宛はそこを参照する」構成が読みやすくACにも共通対応が伝わります。書き始める場合、まずAxQH宛からドラフトしましょうか?