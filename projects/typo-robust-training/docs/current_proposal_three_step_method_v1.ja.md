# 現行 typo 頑健化 proposal：3-step 手法仕様書

更新日: 2026-08-16

対象: Gemma-3-4B-IT / Cycle 3

手法名: **介入誘導型・局在状態蒸留**（intervention-guided localized state distillation）

## 1. Executive summary

### 1.1 何を検証する手法か

現行 proposal は、次の一つの仮説を検証する。

> Activation Patching により、clean state の移植だけで typo の影響を修復できる layer・token 座標を先に同定する。その座標へ限定した state 教師信号を output distribution matching に追加すると、output matching 単独より typo 頑健化を改善できるか。

処理は、互いにデータを分離した3 stepから成る。

1. **Localization**: 汎用テキスト上の因果介入で対象 layer window を一度だけ同定する。
2. **Training**: clean Teacher と clean/typo Student を整合し、追加の state loss だけを対象 window・編集語末へ局在させる。
3. **Evaluation**: 学習なし Base、output matching、proposal、random-window、all-layer を、凍結済みの同一 clean/typo pair で比較する。

Activation Patching は学習前の座標選択と学習後の監査にだけ使う。配備時の推論は Student 単体であり、clean 文、Teacher、patch hook を必要としない。

### 1.2 現在から判定までのdecision map

~~~text
現在
  Step 1: [0,6)を選択・独立検証済み
  Step 2: 10M seed42の5-arm比較完了
          64M output-onlyは3.92M-token checkpointからresume待機、proposalも待機
  Step 3: evaluation protocol v1.4を凍結済み

64M判定
  ├─ proposal > output-only
  │    └─ random-window / all-layer / 3 seedsへ拡張
  │         ├─ proposal > random かつ proposal ≥ all-layer
  │         │    └─ causal localizationの学習価値を主張可能
  │         └─ controlに勝たない
  │              └─ state追加または狭い教師範囲までに主張縮小
  ├─ behavior同等、proposalだけmechanistic audit改善
  │    └─ 実用優位なし。機構差だけを報告
  └─ behavior / auditとも固有差なし
       └─ raw residual cosine形式をkillし、別登録の後継案へ
~~~

### 1.3 現在の状態

| 要素 | 状態 | 凍結された内容 |
|---|---|---|
| Step 1: localization | **完了済み** | FineWeb-Edu、KL-only、全深度、幅6、選択窓 `[0,6)` |
| Step 2: 10M学習 | **完了済み** | seed 42、5 armのmatched-budget比較 |
| Step 2: 64M用量検証 | **output matchingは3.92Mでresume待機 / proposal待機** | GPU 0のSAE WP-2検収を先行。2 armとも設定は10Mから変更しない |
| Step 3: 評価protocol | **凍結済み** | protocol v1.4、paired clean非劣性 / typo優越性 |
| SAE | **現行proposal外** | 診断・後継仮説の事前因果検証だけに使用 |

状態ラベルは本書全体で次の意味を持つ。

| ラベル | 意味 |
|---|---|
| 凍結済み | 結果を見た後に変更しない手続き・設定 |
| 実装済み | repository上に実行可能な実装がある |
| 完了済み | runまたは評価が終了した |
| 実行段階 | 凍結設定で計算を進めている |
| 将来候補 | 現行proposalには含まれない未登録または未実行案 |

### 1.4 手法contract

| Contract項目 | 固定内容 |
|---|---|
| 固定Base / Teacher | 同一revisionのGemma-3-4B-IT、Teacherは完全freeze |
| 学習可能parameter | Student LoRAだけ |
| Localization | generic text、random 1 typo、joint fixed-width patch、KL-only |
| 主学習信号 | 非編集aligned tokenのforward KL |
| Proposal追加信号 | [0,6) × 編集語末のbounded residual cosine |
| Clean保持 | clean:noisy 1:1、clean self-distillation、T0 safety gate |
| Primary behavior判定 | Base比のclean非劣性とrandom-2 typo優越性 |
| 新規性判定 | output-only、random-window、all-layerとのmatched comparison |
| 推論時要件 | Studentのみ。clean oracle、Teacher、SAE、patchは不要 |

### 1.5 Arm一覧

| arm | state教師 | 役割 | 10M seed42 | 64M |
|---|---|---|---|---|
| 学習なしBase | なし | 絶対基準 | 評価済み | 共通参照 |
| Output matching | なし | 中心baseline | 完了 | 3.92M-token checkpointでresume待機 |
| Proposal | causal [0,6) | 主仮説 | 完了 | 待機 |
| Random-window | random [20,26) | 座標選択control | 完了 | 段階投入待ち |
| All-layer | block 0--33 | 局在範囲control | 完了 | 段階投入待ち |

### 1.6 現行proposalに含まれないもの

- SAE feature loss
- neuron/head単位のlocalizationまたはrelative MSE
- gold answer cross entropy
- typo文字列をtargetにした通常のnext-token LM loss
- 独立したclean KL loss
- dynamic loss weightingまたはrho grid search
- AttnLRPによるtypo位置選択
- `rho=0.20`

SAEと後継案は [SAE診断トラックと後継proposal候補](sae_and_successor_proposals_v1.ja.md) に分離する。

改善案は文書末尾の [現時点の改善候補](#241-現時点の改善候補) に、優先順位・開始条件・最小比較・kill条件付きで整理した。概要は次の通りである。

1. 現行手法を変えず、64M tokensで用量不足仮説を検証する。
2. SAEの因果kill testでG1/G2/G3が揃い、かつ結果計算前のregistry amendmentで経路を明示した場合だけ、pair-specific spurious-feature suppressionを別studyとして登録する。
3. Raw residual整合とSAE feature targetがともに不成立なら、patch後のoutput分布を教師にするCausal Patch Distillationを第一の別proposalとする。
4. SAE energyまたはpatch restorationによるdata selectionは、学習target変更と混ぜず、独立したデータ効率studyとして扱う。

これらは選択肢を結果に応じて継ぎ足すメニューではない。現行proposalへの無断追加を禁止し、それぞれ一つの独立仮説として事前登録・baseline/control・停止条件を揃えてから実行する。

### 1.7 Step横断のデータ分離matrix

| Pool | Step 1選択 | Step 1検証 | Step 2学習 | T0 monitor | tune | pre-PR | final | SAE train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FineWeb localization selection 200 | 使用 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 |
| FineWeb localization validation 200 | 禁止 | 使用 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 |
| FineWeb 64M train stream | 禁止 | 禁止 | 使用 | 禁止 | 禁止 | 禁止 | 禁止 | 10M用の先頭30,000 recordだけ除外。残り約51.74MはSAE corpusと重複し得る |
| FineWeb monitor 200 / natural 100 | 禁止 | 禁止 | 禁止 | 使用 | 診断のみ | 禁止 | 禁止 | 運用上禁止。fail-closed roleは未登録 |
| Task tune 500 | 禁止 | 禁止 | 禁止 | 禁止 | 使用 | 禁止 | 禁止 | 禁止 |
| Pre-PR task/corpus | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 1回 | 禁止 | 禁止 |
| Final task/corpus | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 禁止 | 1回 | 禁止 |

全poolについて、record ID、source/group ID、exact duplicate、near-duplicateを検査する。Natural typo辞書・repositoryも用途別に分離する。

ここで「Step 2学習」と「SAE train」は完全非重複ではない。SAE用のmachine-readable exclusionは、先行10M runを想定した先頭30,000 record（概算約12.26M source tokens。ただしdedup amendment分を含む逆算値）を保護する。その直後に得られる初期eligible pool約51.74M source tokensは64M adapter streamの残りと重複し得る。SAE corpus構築時の `--training-budget` は現在registryで一意に凍結されておらず、`minimum` と `preferred` の両方が正当である。`minimum` なら100M training + 10M statistics + 200文書×512 = 110,102,400 source tokensで、約58.37Mを64M stream外から補充する。`preferred` なら合計210,102,400 source tokensで、stream外補充は約158.37Mになる。いずれも最大重複部分は初期eligible約51.74Mであり、「SAE corpus全体が64M stream内」でも「両者が完全非重複」でもない。

Fail-closedに強制されるSAE除外roleは、localization selection / validation、tune、pre-PR、finalの5種である。従ってWP-3はlocalization validationとして保護される一方、monitorとWP-4/5診断itemの非重複は現registryだけでは保証されない。これらは実行前にID/hashをexclusion inventoryへ登録し、SAE train manifestとの非重複をrunnerで検証する。重複があれば結果を計算せず、分離済みreplacement cohortを事前登録する。この対応前に「全診断itemがSAE trainから分離済み」とは主張しない。凍結behavior評価の5 roleに対するfail-closed契約は維持される。

## 2. 研究仮説、非主張、成功条件

### 2.1 Patch可能性と学習可能性

Activation Patching が直接検証するのは、次の**patch可能性**である。

> 外部からclean stateを与えたとき、typoで変化した将来token分布がclean側へ戻るか。

学習が検証するのは、次の**学習可能性**である。

> clean stateを推論時に与えなくても、typo入力だけからStudentが修復に相当する内部状態または出力を生成できるか。

前者は後者を保証しない。曖昧なtypoからclean語を一意に復元できない場合、clean residual全体への一致には原理的な到達不能成分があり得る。この差を実験で裁定することが本研究の中心である。

### 2.2 主張するために必要な比較

Baseに対してcleanを保ちtypoを改善するだけでは、output matchingに対する新規性は立たない。因果的localizationの価値を主張するには、少なくとも次を満たす必要がある。

| 条件 | 検証するもの |
|---|---|
| proposal > output matching | state教師を追加する増分価値 |
| proposal > random window | Activation Patchingによる座標選択の情報量 |
| proposal ≥ all-layer state | 局在しても性能を失わない、または局在自体が有利 |
| proposalがBaseに対してclean非劣 | 頑健化がclean能力の犠牲でない |
| proposalがBaseに対してtypo優越 | 実用上の頑健化がある |

学習後の追加patch gain縮小は機構仮説を補強するが、behavior gateの代わりにはしない。

### 2.3 明示的な非主張

- `[0,6)` が全モデル共通の普遍座標だとは主張しない。移植するのは**モデルごとに再localizeする手続き**である。
- `round(L/6)` が理論的最適幅だとは主張しない。
- 同じ64M予算の実装内baselineを上回っても、データ・モデル・全実装が同一でない限り「Kojima et al.そのものを上回った」とは表現しない。
- State loss低下、state距離、patch gainだけから頑健化成功を主張しない。

## 3. 全体アーキテクチャと共通モデル

### 3.1 End-to-end flow

```text
Step 1: causal localization（学習・評価とID非重複）
  FineWeb-Edu clean 200 + typo 200
      -> 全候補windowをclean-to-typo joint patch
      -> 後続token 2--16のKL restorationを比較
      -> [0,6)を選択
      -> 独立FineWeb-Edu 200で検証し凍結

Step 2: matched training
  Frozen Base Teacher(clean)
      -> non-edited aligned tokenのoutput KL
      -> [0,6) × edited-word-finalのstate cosine（proposalだけ）
  Base + LoRA Student(clean / typo)
      -> LoRA parameterだけを更新

Step 3: frozen paired evaluation
  同じclean/typo itemを
      Base / output-only / proposal / random-window / all-layer
  へ入力
      -> clean非劣性、typo優越性、遷移、corpus保持、mechanistic audit
```

### 3.2 Model roles

| 項目 | 値 |
|---|---|
| Base model | `google/gemma-3-4b-it` |
| revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| decoder block数 | 34 |
| residual hidden dimension | 2,560 |
| 学習・推論dtype | bfloat16 |
| Base model parameter数 | 4,300,079,472 |
| 現行adapter modelの登録総parameter数 | 4,329,881,968 |
| 現行コードで登録・更新可能なLoRA parameter数 | 29,802,496（adapter model総数比で約0.688%） |
| scope修正前のCycle 3 adapter登録数 | 32,788,480（うち未使用vision-tower LoRA 2,985,984） |
| Teacher | 同じBase、完全freeze、clean入力 |
| Student | 同じBase + LoRA、cleanまたはtypo入力 |

同じrevisionをTeacher/Studentに使う理由は、typo頑健化と大Teacherからの能力移植を交絡させず、tokenizer、hidden dimension、word/token alignmentを一致させるためである。Base parameterはfreezeし、gradientはStudentのLoRAだけへ流す。

### 3.3 記号

| 記号 | 定義 |
|---|---|
| \(x_i^c,x_i^p\) | item \(i\) のclean入力、typo入力 |
| \(T,S_\theta\) | frozen Teacher、LoRA Student |
| \(u_i,v_i\) | clean側、typo側の編集語末token位置 |
| \(h^l_{T,u},h^l_{S,v}\) | block \(l\) 出力のcomplete residual state |
| \(A_i\) | 非編集targetのaligned causal-logit位置pair集合 |
| \(E_i\) | clean/typo編集語末位置pair集合 |
| \(W^*\) | localizationで選択・凍結したlayer window |

# Step 1: Activation Patchingによるcausal localization

## 4. 入力データと分離

| 用途 | source | n | 使用可否 |
|---|---|---:|---|
| window選択 | FineWeb-Edu | 200 | 選択に使用 |
| 独立validation | FineWeb-Edu | 200 | 選択後の検証だけ |
| GSM8K / MMLU / ARC等 | reasoning benchmarks | 0 | 選択に不使用 |

Datasetは `HuggingFaceFW/fineweb-edu` / `sample-10BT`、revision `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`、split `train`、seed 42である。最大系列長は512 tokens。Selection、validation、training、全評価tierはrecord IDと重複群で分離する。

Generic textだけで選択する理由は、behavior benchmarkに窓を直接適合させることを避けるためである。Reasoning taskは選択規則へ入れず、学習後の転移評価に残す。

## 5. Typo生成とtoken alignment

各文書に1 typoを加える。操作は次の3種だけで、balanced-deterministic規則によりほぼ均等に割り当てる。

- QWERTY keyboard-neighbor substitution
- deletion
- duplication

対象語は英字2文字以上で、編集後にも16 clean continuation tokensを確保できる候補から一様に選ぶ。AttnLRPは使わない。

文字spanを介してclean/typoの編集語を対応付け、それぞれのword-final token \(u_i,v_i\) を得る。Token数が増減しても、同じtoken indexを仮定しない。Alignment不成立は除外して件数を記録する。

## 6. 介入operatorと候補window

介入対象はattention headやMLP neuronではなく、各complete decoder blockを通過したresidual outputである。候補window \(W\) では、typo runの編集語末だけをclean stateで上書きする。

\[
h^{(l)}_{\mathrm{typo},v_i}
\leftarrow
h^{(l)}_{\mathrm{clean},u_i},
\qquad l\in W
\]

Window外layer、他token位置、model parameterは変更しない。

Decoder block数を \(L\) とし、結果を見る前に幅を固定する。

\[
w=\max\left(1,\left\lfloor \frac{L}{6}+0.5\right\rfloor\right)
\]

候補は全深度の連続windowである。

\[
W_s=[s,s+w),\qquad s=0,\ldots,L-w
\]

Gemmaでは \(L=34,w=6\) なので、開始layer 0--28の29候補をそれぞれ実際にjoint patchする。Single-layer peakからwindowを推測せず、学習で使うoperatorと同じjoint-window operatorで選ぶ。

## 7. 選択metric: multi-token KL restoration

編集語後のclean continuationをteacher-forceし、位置 \(t=2,\ldots,16\) の15 tokenを読む。

\[
D_i^{\mathrm{typo}}
=
\sum_{t=2}^{16}
KL\!\left(p_{i,t}^{\mathrm{clean}}\parallel p_{i,t}^{\mathrm{typo}}\right)
\]

\[
D_i^{\mathrm{patch}}(W)
=
\sum_{t=2}^{16}
KL\!\left(p_{i,t}^{\mathrm{clean}}\parallel p_{i,t}^{\mathrm{patch}(W)}\right)
\]

\[
R_i(W)=1-\frac{D_i^{\mathrm{patch}}(W)}{D_i^{\mathrm{typo}}}
\]

\(R=1\) は完全回復、\(R=0\) は効果なし、\(R<0\) は悪化を表す。比率そのものは15位置のsumでもmeanでも同じだが、実装のeligibility判定は

\[
\overline D_i^{\mathrm{typo}}
=\frac{1}{15}D_i^{\mathrm{typo}}
\]

を用いる。\(\overline D_i^{\mathrm{typo}}\le10^{-9}\) は理由と分母を記録して除外し、KL-eligibleが160件未満または80%未満ならfail closedとする。

Window scoreと選択規則は次で一意に定義する。

\[
S(W)=\operatorname{median}_i R_i(W),
\qquad
W^*=\underset{W}{\arg\max}\;S(W)
\]

完全同点は浅いwindowを選ぶ。Pair bootstrap 10,000回はCIと選択頻度の報告だけに使用し、selection ruleへフィードバックしない。Answer restorationとclean harmも選択には使わない。

## 8. Step 1の凍結出力

Selectionで得た \(W^*\) だけを、別のFineWeb-Edu 200文書で検証する。Validation 95% bootstrap CI下限が0以下なら、そのモデルのlocalized trainingを行わない。

| 出力 | 値 |
|---|---:|
| 選択window \(W^*\) | block 0--5、`[0,6)` |
| selection median restoration | 81.59% |
| selection 95% CI | [78.33%, 86.61%] |
| validation median restoration | 77.23% |
| validation 95% CI | [68.53%, 81.74%] |
| random-window control | block 20--25、`[20,26)` |

`[0,6)`はvalidation合格後にtask、seed、training cycle、behavior結果で再選択しない。Random windowも、非重複の同幅候補から固定seedとSHA-256規則で一つだけ抽選し凍結した。

ただし、現行のconfirmatory generatorが実装する抽選母集団は「\(W^*\)と非重複の同幅windowすべて」であり、training evidenceが要求するmiddle--late制約

\[
s_{\mathrm{random}}\ge \lceil 0.4L\rceil
\]

をgenerator自体は適用していない。Gemmaの凍結値は \(L=34\)、\(s_{\mathrm{random}}=20\ge14\) で制約を満たすため、現行の対照実験は有効である。一方、将来の再localizationで決定的抽選値がこの制約を満たさない場合は、有利な窓が出るまで再抽選せずfail closedとする。Generatorとconsumerの契約は、実験前の別の事前登録済み実装PRで一致させる。

# Step 2: 因果窓へstate教師信号を局在させる学習

## 9. Teacher / Studentとデータflow

### 9.1 学習データ

Cycle 3のstreamはFineWeb-Edu 100%である。

| 項目 | 値 |
|---|---|
| dataset | `HuggingFaceFW/fineweb-edu` / `sample-10BT` |
| revision | `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5` |
| 凍結training records | 157,032 |
| 凍結unique source tokens | 64,000,037 |
| long-run student-token budget | 64,000,000 |
| 最大系列長 | 512 |
| reasoning benchmark text | 0% |
| natural typo context | 0% |

`student tokens`はStudentへ入力されたnon-padding token数である。各rowでTeacher clean forwardとStudent forwardを行うので、forward計算量はStudent 64M tokensより大きく、概ねTeacher分を加えた約2倍相当になる。State条件はhidden-state保持とbackwardにも追加コストを持つ。

### 9.2 Clean/noisy 1:1

Student rowを文書単位で厳密に交互化する。

| row | Teacher入力 | Student入力 | output loss | state loss |
|---|---|---|---|---|
| clean | clean | 同じclean | あり | 0 |
| noisy | clean | 対応するtypo | あり | state armだけあり |

設定値 `explicit_clean_pair_probability=0` はclean row不在を意味しない。`exact-alternating-clean-noisy` schedulerが50:50を保証する。Clean rowのself-distillationがBaseからのdriftを抑えるanchorになる。

### 9.3 学習typo

Record ID、epoch、counterを鍵に決定的生成し、resume後と全armで同じ実現値を再現する。

| 編集語数 | 確率 |
|---:|---:|
| 1 | 0.50 |
| 2 | 0.30 |
| 3--4 | 0.20 |

| 操作 | 確率 |
|---|---:|
| keyboard-neighbor substitution | 0.2667 |
| deletion | 0.2667 |
| duplication | 0.2667 |
| train側natural typo統計に基づくsubstitution | 0.20 |

Natural substitutionは分離済みGitHub typo recordsから得た文字置換統計だけを使う。Natural context自体は学習しない。Adjacent transpositionは学習から除き、unseen-operation評価へ保持する。対象は英字2文字以上で、複数編集は異なる原語に適用する。AttnLRPは使わない。

## 10. Alignmentと教師mask

Typoによりtoken境界が変わるため、文字spanから次を構築する。

1. exact unchanged token pair
2. 非編集tokenを予測するcausal-logit位置pair \(A_i\)
3. clean/typoの編集語末位置pair \(E_i\)

編集語そのものを予測するtargetはoutput lossから除外する。誤綴り文字列自体を正解として学習することを防ぐためである。Alignmentを推測で補完せず、不成立pairを棄却する。Runtime alignment errorは学習停止条件である。

## 11. Loss 1: output distribution matching

Student入力 \(x_i^s\) はclean rowでは \(x_i^c\)、noisy rowでは \(x_i^p\) である。

\[
\mathcal L_{\mathrm{out}}
=
\frac{1}{\sum_i|A_i|}
\sum_i\sum_{(u,v)\in A_i}
KL\!\left(
p_T(\cdot\mid x_i^c,u)
\parallel
p_{S_\theta}(\cdot\mid x_i^s,v)
\right)
\]

- Forward KL、temperature 1.0
- Teacher logitsはdetach
- log-softmaxとKLはfp32
- gradient accumulation batch全体のaligned token総数で正規化
- sample平均の単純平均ではない

このlossの役割は**最終的な予測分布を保つこと**である。Clean rowではTeacher/Student入力が同一なので、独立clean KL lossなしでもclean anchorになる。

## 12. Loss 2: localized residual-state matching

Noisy rowの編集語末だけで、\(W^*=[0,6)\) のcomplete block residualを整合する。

\[
d_{\cos}(a,b)
=1-\frac{a^\top b}
{\max(\lVert a\rVert_2,10^{-8})\,\max(\lVert b\rVert_2,10^{-8})}
\]

\[
\mathcal L_{\mathrm{state}}
=
\frac{1}{|W^*|\sum_i|E_i|}
\sum_i\sum_{(u,v)\in E_i}\sum_{l\in W^*}
\operatorname{clip}_{[0,2]}
\left[
d_{\cos}\!\left(\operatorname{sg}(h^l_{T,u}),h^l_{S,v}\right)
\right]
\]

- Teacher stateはstop-gradient
- cosineはfp32、構造的に0--2へ有界
- clean rowでは0
- accumulation batch全体の編集座標総数で正規化

このlossの役割は**修復をどこで起こすかというinductive bias**を与えることであり、success metricではない。Cycle 1で使用したper-example teacher-energy正規化relative MSEはheavy tailを生じたため再使用しない。

## 13. 総lossと一回限りのgradient calibration

Proposalは、

\[
\mathcal L_{\mathrm{proposal}}
=\mathcal L_{\mathrm{out}}
+\lambda_{\mathrm{state}}\mathcal L_{\mathrm{state}}
\]

Output-only baselineは、

\[
\mathcal L_{\mathrm{output-only}}=\mathcal L_{\mathrm{out}}
\]

である。最初の8 noisy pairで各項を別々にbackwardし、LoRA gradient normの平均比から係数を一度だけ導出する。

\[
\lambda_{\mathrm{state}}
=
\rho
\frac{\frac18\sum_{i=1}^{8}\lVert\nabla_\theta\mathcal L_{\mathrm{out},i}\rVert_2}
     {\frac18\sum_{i=1}^{8}\lVert\nabla_\theta\mathcal L_{\mathrm{state},i}\rVert_2},
\qquad \rho=0.05
\]

導出後は学習中に変更しない。

| arm | state scope | seed 42の導出値 \(\lambda_{\mathrm{state}}\) |
|---|---|---:|
| proposal | block 0--5 | 0.2022819501 |
| random-window | block 20--25 | 0.7746756726 |
| all-layer | block 0--33 | 0.3938028647 |

Raw係数が違うのはscopeごとの未加重gradient normが異なるためである。`rho=0.10` pilotはstartup safety gateで停止した。`rho=0.20`は正式手法でも登録済みfallbackでもない。

重要な留保として、\(\rho=0.05\) は初期8例の**平均gradient norm同士の比**を一度だけ揃える。各example、各micro-batch、各optimizer stepでstate/output比を5%へ維持する規則ではない。このためnull結果は、targetの無効性と実効state用量を完全には分離しない。

## 14. LoRA、optimizer、更新範囲

| 項目 | 値 |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 8 |
| LoRA scaling \(\alpha/r\) | 0.5 |
| dropout | 0 |
| bias | none |
| 意図したLoRA配置 | 全34 text decoder blocks |
| text decoder target modules | q/k/v/o、gate/up/down projections |
| scope保証 | decoder projectionの完全修飾pathをPEFTへ渡し、vision towerを除外 |
| optimizer | AdamW |
| learning rate / weight decay | \(10^{-4}\) / 0.01 |
| scheduler / warmup | constant-with-warmup / 0.0 |
| micro batch / accumulation | 1 sequence / 32 micro steps |
| max gradient norm | 1.0 |
| gradient checkpointing | enabled |
| checkpoint間隔 | 50 optimizer steps |
| max optimizer steps | 10,000（安全上限） |
| long-run停止条件 | 64M student tokens |
| seeds | 42, 43, 44 |

「局在」とはLoRA配置ではなく、**state lossのlayerとtoken位置**を指す。Output lossは全aligned targetsで計算され、Studentのtext LoRAは全decoder layersへ配置される。これにより、対象stateを作る上流変更と、出力を維持する下流調整の両方を学習できる。

### 14.1 歴史的runのmodule-scope注記

scope修正前に開始したCycle 3 runでは、module名`layers`とlayer indexだけでscopeを判定していたため、SigLIP vision towerのencoder layer 0--26にあるq/k/v projectionにもLoRAを追加登録していた。text-only学習ではvision towerをforwardしないため、これら2,985,984 parameterにはgradientが付かず、AdamWも更新しない。したがって、既存runで実際に更新されたtext decoder側は29,802,496 parameterであり、全armのmatched-capacity比較と数値結果は無効化されない。一方、当該checkpointの登録総数32,788,480には未学習のvision LoRA weightが含まれる。

現行コードは、modelのdecoder projectionを完全修飾pathで先に解決し、そのpathだけをPEFTへ渡す。さらにtrainable parameter reportもdecoder pathとlayer×moduleの直積を検証するため、新規runではvision towerへLoRAを登録しない。

Runtime checkpointはscope契約を含むschema v3で保存する。scope修正前のcheckpointはoptimizer parameter groupも異なるため、現行コードへ暗黙変換してresumeしない。Adapter tensor集合とoptimizer groupを読み込み前に照合し、不一致なら「checkpointを生成したcode revisionでresumeする」よう明示して停止する。これにより、未使用vision tensorだけを無言で捨てた後にoptimizer stateを別parameterへ対応づける事故を防ぐ。進行中のscope修正前runは、そのrunを開始したworktree・revisionを変更せず完走またはresumeする。

| 数 | parameter数 |
|---|---:|
| checkpoint登録LoRA全体 | 32,788,480 |
| text-only forwardで更新可能 | 29,802,496 |
| 未使用vision LoRA（`grad=None`） | 2,985,984 |

Vision towerはtext-only runでforwardされないため、そのweightは更新されない。全armで共通なので既存matched-capacity比較を無効化しないが、歴史的checkpointとmodule-scope修正後のコードを混同してはならない。本書では、歴史的runの実態と意図した現行仕様を分けて記録する。

## 15. 学習中monitor、安全gate、lossの読み方

### 15.1 Clean保持機構

1. Clean/noisyを1:1にする。
2. Clean rowにもBase Teacherとのoutput KLを課す。
3. Baseをfreezeし、更新をLoRAへ限定する。
4. Held-out clean KLとPPLを監視する。
5. 違反時は停止し、違反前checkpointへrollbackする。

追加のclean lossを置かないのは、clean rowのoutput matchingが同じ役割を持ち、baselineとの比較を単純に保てるためである。

### 15.2 T0 monitorと停止条件

学習中にtask accuracyを見ない。

| monitor | n | 指標 |
|---|---:|---|
| held-out FineWeb-Edu clean | 200 | clean KL、PPL ratio |
| held-out natural clean/typo | 100 pair | aligned clean--typo KL |

停止条件は次である。

- clean \(KL(Base\parallel Student)>0.03\) が2回連続
- \(PPL(Student)/PPL(Base)>1.02\) が2回連続
- startupでweighted state/output gradient proxyが0.5超を3回連続
- alignment error
- NaN/Inf lossまたはnon-finite gradient norm

### 15.3 W&B train lossの解釈

`train/loss/output` のmicro-batch列は、そのまま収束判定に使わない。

- clean/noisyを交互投入するため、低いclean lossと高いnoisy lossが鋸歯状に並ぶ。
- 文書長、編集数、操作、tokenization変化が毎recordで違う。
- 64M unique streamをほぼ単一epochで読むため、同一training setへの反復fitのような単調低下を期待しない。
- 表示上のper-record平均と、実際にbackwardするaligned-token加重 `train/objective/output` は異なる。

健全性は、optimizer objectiveの50--100 step移動平均、clean/noisy別分布、gradient/clip、NaN/Inf、T0 monitor、fixed checkpoint probeを組み合わせて判定する。Proposal total lossの絶対値をoutput-only total lossと直接比較してはならない。

# Step 3: 学習後モデルの凍結paired評価

## 16. 評価契約とデータtier

全arm・全seedが同じitemのclean版と凍結済みtypo版を読む。

| | clean | typo |
|---|---:|---:|
| Base | \(B_c\) | \(B_t\) |
| Adapter | \(A_c\) | \(A_t\) |

主要estimandは、

\[
\Delta_{clean}=Acc(A_c)-Acc(B_c),
\qquad
\Delta_{typo}=Acc(A_t)-Acc(B_t)
\]

である。Cleanは非劣性、typoは優越性として別々に判定する。Robustness gapはcleanを落とすだけでも縮むので、報告専用であり成功判定には使わない。母集団はBaseのclean-correct / typo-wrongに条件付けない固定無条件sampleである。

凍結protocolはv1.4である。

| Tier | 内容 | 開封規則 |
|---|---|---|
| monitor | corpus安全性 | 学習中可、task accuracy禁止 |
| tune | cycle・ablation選択 | 反復可、headlineに不使用 |
| pre-PR gate | 候補の確証確認 | arm・3 seeds・config・checkpoint hash固定後に1回 |
| final test | 論文headline | pre-PR合格後、全設計固定後に1回 |

### 16.1 Benchmark battery

| Tier | Task battery | Corpus battery |
|---|---|---|
| tune | GSM8K / MMLU / ARC、合計500 | FineWeb 200、natural LM 100 pair、natural injection 100 |
| pre-PR | GSM8K / MMLU / ARC / MMLU-Pro / CommonsenseQA、各500、計2,500 | FineWeb 1,000、Dolma 500、natural LM 500 pair |
| final | 上記5 task各500 + MATH-500 440、計2,940 | FineWeb 1,000、Dolma 1,000、natural LM 1,000 pair |

MATH-500の440件は4つの異なる適格語を持つv1.4条件を満たす件数である。旧censusの466は使用しない。

Training IDs、localization selection/validation、Cycle 1 pilot、他tier、exact/near-duplicate groupを排除する。Natural typoはrepository単位、natural-injection辞書はcorrected word単位でもtier間を分離する。

## 17. 評価typoと推論条件

Primaryは`random-2`である。

- question内の異なる適格語2語をattributionなしで一様選択
- keyboard-neighbor substitution / deletion / duplication
- seed 42で実現textをファイル凍結
- 全arm・seedが同じfileを使用
- 英字3文字以上を対象
- few-shot、answer option本文、gold answer、option label、数字、数式、URL、email、identifierは編集しない

Secondaryは`random-1`、`random-4`、学習hold-outの`transposition-2`、evaluation-only natural injection、held-out natural LM pairである。Attribution-4は別登録のstress diagnosticにはできるが、primary batteryにもgateにも含めない。

生成条件は全armで共通である。

| Task | Prompt |
|---|---|
| GSM8K | 8-shot CoT |
| MMLU / MMLU-Pro / ARC / CommonsenseQA | 5-shot CoT |
| MATH-500 | 4-shot CoT |

Greedy、bfloat16、最大512 new tokens、task別extractorとdeterministic fallbackを用い、抽出不能は不正解とする。Few-shot例はcleanのままquestionだけを編集する。

## 18. Metric、統計、transition

各paired比較で、wrong-to-wrong、wrong-to-right、right-to-wrong、right-to-rightを報告し、netだけを提示しない。

Task-equal macroは、

\[
\Delta_{macro}
=\frac1K\sum_{k=1}^{K}
\frac1{|I_k|}\sum_{i\in I_k}(y_{A,i}-y_{B,i})\times100
\]

である。二値paired比較には両側exact McNemarを併記する。Accuracy CIはtask-stratified paired bootstrap 10,000回、3-seed estimateはlearning seedを外側、task内itemを内側とするhierarchical bootstrapを使う。Two-sided 95% CIとgate用one-sided 95% lower boundを保存する。

Clean corpusではteacher-forced NLLから、

\[
PPL=\exp\left(\frac{\sum_t-\log p(x_t\mid x_{<t})}{N_{tokens}}\right),
\qquad
PPL\ ratio=\frac{PPL(Adapter)}{PPL(Base)}
\]

を計算し、FineWeb clean上の \(KL(Base\parallel Adapter)\) median/p95も報告する。Natural pairはexact unchanged character spanから非編集next-tokenを対応付け、編集語targetを除外する。

## 19. 凍結gate

| Gate | 条件 |
|---|---|
| clean非劣性 | \(\widehat\Delta_{\mathrm{clean}}\ge-1.0\) point、one-sided 95% lower bound \(>-1.0\)、各task point estimate \(>-3.0\) |
| typo優越性 | \(\widehat\Delta_{\mathrm{typo}}\ge+2.0\) points、one-sided 95% lower bound \(>0\) |
| corpus保持 | PPL ratio \(\le1.02\)、median clean KL \(\le0.03\) |
| seed方向一致 | 3 seeds中2以上でclean変化が非負、typo変化が正 |
| natural非劣化 | point estimate \(\ge-1.0\)、lower bound \(>-2.0\)、natural LM KL gap非拡大 |

Arm間の記号もv1.4と同じtask-stratified paired bootstrap・seed階層を使う。「>」は事前指定contrastの95% CI下限が0より大きいことを意味する。「≈」は「有意差がない」だけで同等性を主張せず、優越性も非劣性も示せなかった記述的状態を表す。

「proposal ≥ all-layer」の非劣性marginは、現行evaluation config v1.4ではまだ定義されていない。従って、all-layerに対するconfirmatoryな非劣性を主張する場合は、control実行・sealed開封前にmarginをprotocol amendmentで凍結しなければならない。未登録のままなら、点推定とtwo-sided CIを記述し、「all-layerに劣らない」とは主張しない。

## 20. Mechanistic paired-patching audit

学習後モデル自身のclean runをdonor、同モデルのtypo runをrecipientとして、凍結windowをpatchする。

各item \(i\)、model状態 \(m\in\{\mathrm{Base},\mathrm{Adapter}\}\) について、編集語末後の15位置 \(t=2,\ldots,16\) をteacher forcingし、算術平均を取る。

\[
U_{i,m}=\frac1{15}\sum_{t=2}^{16}
KL\!\left(p^{\mathrm{clean}}_{i,m,t}\parallel
          p^{\mathrm{typo}}_{i,m,t}\right)
\]

\[
P_{i,m}=\frac1{15}\sum_{t=2}^{16}
KL\!\left(p^{\mathrm{clean}}_{i,m,t}\parallel
          p^{\mathrm{patched}}_{i,m,t}\right)
\]

15位置のいずれかが非有限、または \(U_{i,m}\le10^{-6}\) のitemは当該modelでinvalidとする。BaseとAdapterの両方でvalidなitemの共通集合を \(I_{\mathrm{common}}\) とし、arm間比較にはこの集合だけを使う。armごとのinvalid件数、理由、\(n_{\mathrm{common}}\)、coverageを必ず併記する。

\[
G_{i,m}=1-\frac{P_{i,m}}{U_{i,m}},
\qquad
\overline G_m=\frac1{|I_{\mathrm{common}}|}
\sum_{i\in I_{\mathrm{common}}}G_{i,m}
\]

確証的なmechanistic auditを将来実装するときの主要estimandは、BaseからAdapterへの追加patch余地の絶対減少とする。

\[
\Delta G_{\mathrm{reduction}}
=\overline G_{\mathrm{Base}}-\overline G_{\mathrm{Adapter}}
\]

\(\overline G_{\mathrm{Base}}>10^{-6}\) の場合だけ、補助的な相対量を定義する。

\[
PatchGainReductionFraction
=\frac{\overline G_{\mathrm{Base}}-\overline G_{\mathrm{Adapter}}}
       {\overline G_{\mathrm{Base}}}
\]

\(\overline G_{\mathrm{Base}}\le10^{-6}\) ならrelative fractionはundefinedとする。0除算回避のために分母へ任意のepsilonを足してfractionを作らない。

現行runnerが直接保存するpatch-gain量は、`base_mean_patch_gain`、`adapter_mean_patch_gain`、およびBase平均が \(10^{-6}\) より大きい場合の `patch_gain_reduction_fraction` である。\(\Delta G_{\mathrm{reduction}}\) 専用fieldはなく、2平均から導出できるだけである。Base平均が閾値以下の場合、runnerはrelative fractionを `None` とし、2平均だけを保存する。従って現行artifactについては2平均とrelative fractionの定義可否をそのまま報告し、絶対差を表示する場合は「保存値から事後導出」と明記する。

不確実性は、task内でitemを再抽出しtaskを等重み集計するpaired bootstrap 10,000回（seed 42）で計算する。各反復でBaseとAdapterに同じitem indexを使い、\(\overline G_m\)、\(\Delta G_{\mathrm{reduction}}\)、定義可能な場合のrelative fractionについてpercentile 95% CIを報告する。このbootstrap CIと絶対差の専用出力は現行runnerに未実装であり、inferentialなmechanistic claimを行う前に実装・回帰testが必要な明示的gapである。実装完了前は既存の2平均とrelative fractionを記述的にのみ扱う。

- Accuracy改善 + reduction: external patchが担った修復を内在化した可能性
- Accuracy改善 + reductionなし: downstream compensationの可能性
- Reduction + accuracy改善なし: 内部表現は動いたが実用改善なし

Audit、state loss、state distanceはmechanistic diagnosticであり、behavior gateをblockしない。

# 比較arm、既知結果、判断規則

## 21. 必須baseline / control

| arm | 学習loss | state scope | 何を分離するか |
|---|---|---|---|
| 学習なしBase | なし | なし | 絶対基準 |
| Kojima型output matching | \(\mathcal L_{\mathrm{out}}\) | なし | state教師なしの中心baseline |
| 現行proposal | \(\mathcal L_{\mathrm{out}}+\lambda\mathcal L_{\mathrm{state}}\) | block 0--5 | 主仮説 |
| Random-window | 同じ2項 | block 20--25 | state追加と因果座標選択を分離 |
| All-layer | 同じ2項 | block 0--33、編集語末のみ | 局在と広域state supervisionを分離 |

全armでBase revision、source order、realized typo、clean/noisy比、LoRA容量、optimizer、student-token budget、seedを一致させる。All-layerは全層×全tokenではなく、**全層×編集語末**である。

Output-onlyはfrozen self-teacher、1:1、編集語target mask、output distribution matching、LoRAというKojima型の中心構造を実装する。ただし先行研究runと完全同一ではないため「同実装内matched-budget Kojima型baseline」と表現する。

## 22. 10M seed-42 tune結果

これは反復可能なtune結果であり、pre-PR/finalの確証結果ではない。

| arm | clean acc | typo acc | \(\Delta\) clean vs Base | \(\Delta\) typo vs Base | relative patch-gain reduction |
|---|---:|---:|---:|---:|---:|
| Base | 83.0% | 80.0% | 0 | 0 | 0 |
| output matching | 82.4% | 82.6% | -0.60pt | +2.60pt | 23.83% |
| proposal 0--5 | 82.6% | 82.4% | -0.40pt | +2.40pt | 30.34% |
| random 20--25 | 82.2% | 82.6% | -0.80pt | +2.61pt | 31.74% |
| all-layer 0--33 | 82.4% | 82.0% | -0.60pt | +2.00pt | 40.32% |

右端列は現行runnerが保存した補助的なrelative fractionであり、上で定義した絶対 \(\Delta G_{\mathrm{reduction}}\) ではない。Proposalのtypo差はoutput matching比 -0.20pt、random比 -0.20pt、all-layer比 +0.40ptで、すべて95% CIが0を跨いだ。従って10M、seed 42で言えるのは次だけである。

- 大きなclean driftを避けて全armを学習できた。
- Proposalがoutput matchingやrandom windowを上回る行動的優位は確認できない。
- Patch gain reductionもproposal固有ではない。
- 1 seed・tune結果なので確証的な不成立判断でもない。

## 23. 学習健全性の独立監査

W&Bの強い上下が「baselineとproposalの両方が正常に学習できず、評価が無効」を意味するかを、10Mの全766 optimizer steps / 24,512 micro-batchesで再集計した。

| 指標 | output matching | proposal |
|---|---:|---:|
| student tokens | 10,001,217 | 10,001,217 |
| total objective、first 100平均 | 0.008734 | 0.031432 |
| total objective、last 100平均 | 0.007496 | 0.024935 |
| total objective変化 | -14.2% | -20.7% |
| proposal weighted state変化 | -- | -27.5% |
| gradient norm最大 | 0.313 | 0.256 |
| clip閾値1.0超過 | 0/766 | 0/766 |
| NaN/Inf | 0 | 0 |
| clean KL最大 / gate | 0.00248 / 0.03 | 0.00380 / 0.03 |
| PPL ratio最大 / gate | 1.0010 / 1.02 | 1.0019 / 1.02 |
| natural typo KL ratio終盤 | 約0.851 | 約0.828 |

Baselineのper-record output KLはclean平均0.00215、noisy平均0.01577、noisy最大0.608だったが、aligned-token正規化後のoptimizer objective最大は0.01771だった。Step、token countは連続し、全armのrecord/edit-count列も一致した。

従って、**数値的不安定のため10M checkpoint評価自体が無効**という強い仮説は保存証拠と整合しない。ただし次は未解決である。

- seed 42・tune tierだけであり、優位性の確証ではない。
- Baseのclean--typo差は3ptと小さく、評価感度に限界がある。
- 初期gradient calibrationは学習中のstate/output比を固定しない。単一noisy-example proxyではmedian約0.279、p95約1.35であり、null時にstate用量とtarget有効性を完全分離できない。

将来runでは、accumulation batch全体の各項gradient normとgradient cosineをログすべきである。ただしこの診断を理由に進行中の凍結run設定を変更しない。

## 24. 64M用量検証とdecision tree

10MはBase/output/proposal/random/all-layerのseed 42比較が完了した。64M matched-budget configはoutput matchingとproposalについて実装済みで、凍結設定のまま段階実行する。Random/all-layerと追加seedは、因果的新規性を主張する段階では必須である。

```text
64M proposal
  ├─ Base gate合格、outputより良い
  │    ├─ randomより良く、all-layer以上
  │    │    └─ causal localizationの増分価値を主張可能
  │    └─ random / all-layerに勝たない
  │         └─ state追加または狭い教師範囲までの縮小主張
  ├─ behavior同等、proposalだけaudit縮小
  │    └─ 実用優位なし。機構差に限定
  └─ output以下、audit固有差なし
       └─ このraw residual cosine形式を不成立と判断し、別登録の後継案へ
```

### 24.1 現時点の改善候補

改善候補は、現在の結果を見ながらlossを継ぎ足すためのメニューではない。各候補を**一つの独立仮説**として扱い、開始条件、baseline、control、kill条件を実行前に固定する。詳細な数式・計算量・risk・縮小claimは [SAE診断トラックと後継proposal roadmap](sae_and_successor_proposals_v1.ja.md#part-ii-後継proposal-roadmap) にまとめる。

| 順位 | 候補 | 変えるもの | 開始条件 | 最小比較 | 不成立時 |
|---:|---|---|---|---|---|
| A | 現行raw-residual法の64M用量検証 | token budgetだけ | 現在実行段階 | output-only vs proposal、後にrandom/all-layer | raw residual cosine形式をkill |
| B | SAE pair-specific spurious抑制 / 条件付きfixed-feature整合 | state教師target | suppressionはWP-2 + WP-5 G1/G2/G3 + 事前registry amendment、fixed featureはさらに別のselection / held-out kill test合格 | output-only、raw residual、matched-random / all-feature scope control | SAEは診断専用へ戻す |
| C | Causal Patch Distillation | clean state一致をpatched-output教師へ置換 | A不成立。B不合格または不採用 | 通常output、causal-window teacher、random-window teacher | patch可能性は蒸留可能性を保証しないと報告 |
| D | SAE energy data selection | noisy training pairの選別 | 別studyとして登録 | high / low / severity-matched random、同token予算 | energyを診断量へ限定 |
| E | Intervention-restoration data selection | noisy pairの選別 | 別studyとして登録 | high / low restoration / matched random | patch-positive subsetの境界結果へ限定 |

候補Aは新しい学習手法ではなく、10Mで見えなかった差がKojima型の代表規模に近い予算で現れるかを確認する**用量検証**である。現registryは「G1とG3だけがfeature-targeted successor-studyの**草案**を許す。学習は許可しない」と凍結している。Pair-specific suppressionを候補化するには、結果計算前のamendmentでG1を維持したままG2 routingを追加し、G1/G2/G3の全てを満たした後に別studyを事前登録する必要がある。WP-5 G1が検証するのはall-feature成分 \(Dz\) の十分性であり、固定causal feature集合の同定ではない。Fixed-feature整合にはさらに別のpreregistered selection / held-out kill testが必要である。これらが不成立でraw residualも不成立ならCへ進む。D/Eは学習targetではなくデータ効率を変える独立軸なので、B/Cと同時投入しない。

追加で必要な実験妥当性の補強は、accumulation batch全体でのoutput/state gradient normとgradient cosineの記録である。これは新しい頑健化手法ではなく、state信号が実際に届いたかを確かめるdose telemetryとして別扱いにする。Causal windowだけへLoRAを置く案もparameter-efficiency ablationであり、主たる性能改善案には数えない。

次は現行確証trackから除外する。

- behavior結果を見た後のrho grid search
- behavior結果を使うwindow再選択
- neuron/head単位学習の復活
- 大Teacherとのstate matching
- 複数の新lossを同時に追加する設計

## 25. 再現手順とsource of truth

### 25.1 再現順序

1. Frozen model/data revisionと除外registryを検証する。
2. Step 1 selection 200を生成し、全29 windowをjoint patchする。
3. 選択規則で \(W^*\) を決定し、別200で一度だけvalidationする。
4. 固定seedとSHA-256規則で、\(W^*\)と非重複の同幅random controlを1本だけ抽選する。
5. Random controlが \(s_{\mathrm{random}}\ge\lceil0.4L\rceil\) を満たすか検証する。不合格時は再抽選せずfail closedとし、事前登録済みの別実装PRでgenerator/consumer契約を整合する。
6. \(W^*\)、検証済みrandom control、ID hash、typo manifestを凍結する。
7. 同一source/typo scheduleでoutput-onlyとstate armsを学習する。
8. T0安全monitorだけを学習中に見る。
9. Checkpoint hashを固定後、tune -> pre-PR -> finalの開封規則に従う。
10. Behavior評価とmechanistic auditを別階層で報告する。

### 25.2 Source of truth

Cycle 1以前の文書には4項loss、reasoning mixture、selected-layer LoRA等が残る。Cycle 3は次の優先順位で確認する。

| 優先 | Source | Path / 状態 |
|---:|---|---|
| 1 | 10M output config | [configs/cycle3/gemma4b-output-matching-10m.yaml](../configs/cycle3/gemma4b-output-matching-10m.yaml)、完了runの設定 |
| 1 | 10M proposal config | [configs/cycle3/gemma4b-causal-window-10m.yaml](../configs/cycle3/gemma4b-causal-window-10m.yaml)、完了runの設定 |
| 1 | 10M controls | [random-window](../configs/cycle3/gemma4b-random-window-10m.yaml) / [all-layer](../configs/cycle3/gemma4b-all-layer-state-10m.yaml)、完了runの設定 |
| 1 | 64M configs | [output-only](../configs/cycle3/gemma4b-output-matching-64m.yaml) / [proposal](../configs/cycle3/gemma4b-causal-window-64m.yaml)、実行段階 |
| 2 | Localization protocol | [configs/cycle3/gemma4b-generic-joint-window.yaml](../configs/cycle3/gemma4b-generic-joint-window.yaml)、選択・検証artifactはrepo外run recordにhash固定 |
| 3 | Evaluation protocol | [configs/robustness-evaluation-v1.yaml](../configs/robustness-evaluation-v1.yaml)、v1.4 machine-readable source |
| 4 | Loss implementation | [training/losses.py](../src/typo_robust_training/training/losses.py) / [training/step.py](../src/typo_robust_training/training/step.py) |
| 4 | Training runtime | [training/runner.py](../src/typo_robust_training/training/runner.py) / [training/adapters.py](../src/typo_robust_training/training/adapters.py) |
| 4 | Localization runtime | [localization/confirmatory_runner.py](../src/typo_robust_training/localization/confirmatory_runner.py) / [confirmatory_scoring.py](../src/typo_robust_training/localization/confirmatory_scoring.py)。現行generatorは非重複同幅候補全体から抽選し、middle--late制約はtraining evidence側で検証されるため、将来の再localizationでは上記fail-closed契約を適用する |
| 4 | Evaluation runtime | [evaluation/study.py](../src/typo_robust_training/evaluation/study.py) / [evaluation/metrics.py](../src/typo_robust_training/evaluation/metrics.py) |
| 5 | Narrative | 本書。上記machine-readable sourceと衝突した場合は上記を優先 |

コード修正後の仕様と歴史的checkpointの実態が異なる場合、run recordを過去結果のsource of truthとし、差分を明記する。古い失敗runの設定や未登録の後継案を現行proposalへ混ぜない。
