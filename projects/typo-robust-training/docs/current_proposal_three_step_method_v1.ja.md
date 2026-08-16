# 現行 typo 頑健化 proposal：3-step 手法の理論・実装・評価

更新日: 2026-08-16

対象: Gemma-3-4B-IT、Cycle 3

手法名: **介入誘導型・局在状態蒸留**（intervention-guided localized state distillation）

## 0. この文書の位置づけ

本書は、現在の proposal を次の3段階に分け、理論上の狙い、実際の処理、データ、数式、比較条件、評価方法を一つにまとめたものである。

1. Activation Patchingによる、typoの影響修復に寄与するlayer windowの同定
2. 同定したlayer windowへstate教師信号を局在させる学習
3. 学習後モデルのclean/typo入力に対する凍結済みpaired評価

現行 proposal にはSAE、neuron/head単位のloss、answer cross entropy、独立したclean KL loss、`rho=0.20`は含まれない。SAEは別の診断・後継研究トラックであり、詳細は `sae_and_successor_proposals_v1.ja.md` に分ける。

本書では、次の状態を区別する。

| 状態 | 意味 |
|---|---|
| 凍結済み | 結果を見て変更しない手続き・設定 |
| 実装済み | repository上に実行可能な実装がある |
| 完了済み | 当該runまたは評価が終了している |
| 実行中 | 現在計算中 |
| 将来候補 | 未登録または未実行で、現行proposalには含まれない |

## 1. 一文で表した中心仮説

> clean stateをtypo runへ移植すると出力が回復するlayerとtoken位置をActivation Patchingで先に同定し、その座標へ小さなstate蒸留信号を与えることで、出力分布整合だけより効率的かつ機構的に監査可能なtypo頑健化を実現できるかを検証する。

ここで、Activation Patchingが直接示すのは**patch可能性**である。

> 外部からclean stateを与えれば、typoで変わった分布を戻せるか。

学習が検証するのは別の性質である。

> typo入力だけから、Studentがその修復に相当する表現または挙動を自力で生成できるか。

したがって、「patchで直る」ことは「その座標を教師にすれば学習できる」ことを保証しない。この二つを結び付けられるかが研究仮説である。

## 2. 全体フロー

```text
[Step 1: 因果窓の同定]

独立したFineWeb-Edu clean文書
        |
        +-- 1 typoをランダム・決定的に生成
        +-- Base(clean) / Base(typo)
        +-- clean residualをtypo runへwindow単位でpatch
        +-- 後続token 2--16のKL restorationで全windowを比較
        `-- 因果窓 [0,6) を選択・独立検証・凍結
                    |
                    v
[Step 2: 学習]

frozen Base Teacher                 Base + all-layer LoRA Student
clean入力                           cleanまたはtypo入力
        |                                  |
        +---- 非編集aligned tokenの出力KL--+
        +---- [0,6)×編集語末のstate cosine-+
                           |
                           v
                    LoRAだけを更新
                           |
                           v
[Step 3: 凍結評価]

同一のclean/typo pair
  × 学習なしBase
  × 出力分布整合のみ
  × 因果窓state proposal
  × random-window state
  × all-layer state
```

Activation Patchingは学習中や推論時に毎回行うものではない。現行法では、学習前にstate lossの計測座標を一度だけ選ぶために使う。学習後の推論にはclean入力、Teacher、patch hookのいずれも不要である。

## 3. 共通モデル

| 項目 | 値 |
|---|---|
| Base model | `google/gemma-3-4b-it` |
| revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| decoder block数 | 34 |
| residual hidden dimension | 2,560 |
| 学習・推論dtype | bfloat16 |
| 総parameter数 | 4,332,867,952 |
| LoRA学習parameter数 | 32,788,480 |
| 学習parameter比率 | 約0.757% |
| Teacher | 同じBase、完全freeze、clean入力 |
| Student | 同じBase + LoRA、cleanまたはtypo入力 |

TeacherとStudentに同じmodel revisionを用いる理由は、typo頑健化と大規模Teacherからの能力移植を交絡させず、tokenizer、hidden dimension、語位置alignmentを一致させるためである。Teacher parameterとStudentのBase parameterはfreezeし、gradientはStudentのLoRAだけへ流す。

# Step 1：Activation Patchingによる対象layerの同定

## 4. 問いと介入単位

Step 1では次を問う。

> typo入力中の編集語位置にclean入力のresidual stateを移植したとき、どの連続layer windowで将来tokenの分布が最もclean側へ回復するか。

Clean/typo activation差が大きいlayerを相関だけで選ぶのではなく、その座標を実際に上書きして出力分布を再計算する。

介入単位は、各decoder blockを通過した**complete residual output**である。attention headだけ、MLP neuronだけ、特定のhidden dimensionだけをpatchしない。

## 5. Localization用データ

| 用途 | source | 件数 | 分離規則 |
|---|---|---:|---|
| window選択 | FineWeb-Edu | 200文書 | 学習・評価・validationとID非重複 |
| 独立validation | FineWeb-Edu | 200文書 | selectionともID非重複 |

固定値は次のとおりである。

- dataset: `HuggingFaceFW/fineweb-edu`
- subset: `sample-10BT`
- revision: `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`
- split: `train`
- seed: 42
- 最大系列長: 512 tokens
- 読み出しに必要な後続token: 16

GSM8K、MMLU、ARC等のbehavior benchmarkはwindow選択に使わない。これにより、評価taskへ直接適合したlayer選択になることを避ける。

## 6. Localization用typo

各文書へ1 typoだけを加える。

- QWERTY keyboard-neighbor substitution
- deletion
- duplication

3操作はbalanced-deterministic規則で、できるだけ均等に割り当てる。編集語は英字2文字以上で、編集後にも少なくとも16 clean continuation tokensを残せる語から一様に選ぶ。

AttnLRPは使用しない。したがって、現在のlayer選択はattribution-targeted stress例ではなく、汎用テキスト中のランダムtypoに基づく。

## 7. Window幅と探索範囲

Decoder block数を (L) とし、window幅を結果を見る前に次で固定する。

\[
w
=
\max\left(1,\left\lfloor\frac{L}{6}+0.5\right\rfloor\right)
\]

Gemma-3-4B-ITでは (L=34) なので (w=6) となる。候補は全深度の連続windowである。

\[
W_s=[s,s+w),
\qquad
s=0,\ldots,L-w
\]

Gemmaでは開始layer 0から28までの29候補を、それぞれ実際にjoint patchする。探索をearly layersだけへ制限しない。

`round(L/6)`は最適幅の理論証明ではない。元の6-layer interventionをモデル深度へ相対化し、結果前に固定した簡略規則である。恣意性は、幅の自由な再調整を許さず、同幅random windowとall-layer controlで経験的に検証することで抑える。

## 8. Patch operator

文書 (i) のclean側word-final tokenを (u_i)、typo側word-final tokenを (v_i) とする。Frozen Baseをclean、typo、candidate-patched typoの3条件でforwardし、candidate window (W) では、

\[
h^{(l)}_{\mathrm{typo},v_i}
\leftarrow
h^{(l)}_{\mathrm{clean},u_i},
\qquad l\in W
\]

と上書きする。Typo側の他のtoken位置、window外layer、model parameterは変更しない。

## 9. Multi-token KL restoration

編集語後のclean continuationをteacher-forceし、位置 (t=2,\ldots,16) の15 tokenを読む。

\[
D_i^{\mathrm{typo}}
=
\sum_{t=2}^{16}
KL\!\left(
p_{i,t}^{\mathrm{clean}}
\parallel
p_{i,t}^{\mathrm{typo}}
\right)
\]

\[
D_i^{\mathrm{patch}}(W)
=
\sum_{t=2}^{16}
KL\!\left(
p_{i,t}^{\mathrm{clean}}
\parallel
p_{i,t}^{\mathrm{patch}(W)}
\right)
\]

Pair単位の回復率を、

\[
R_i(W)
=
1-
\frac{D_i^{\mathrm{patch}}(W)}{D_i^{\mathrm{typo}}}
\]

とする。

- (R_i=1): clean--typo KL gapが完全に消えた。
- (R_i=0): patchの効果がない。
- (R_i<0): patchによってcleanからさらに離れた。

(D_i^{\mathrm{typo}}\le10^{-9}) のpairは理由と分母を記録して除外する。少なくとも160件、かつ200件の80%以上がKL-eligibleでなければfail closedとする。

各windowのscoreはpairwise restorationの中央値である。

\[
S(W)=\operatorname{median}_i R_i(W)
\]

\[
W^*=\arg\max_W S(W)
\]

完全同点では浅いwindowを選ぶ。Pair bootstrap 10,000回はCIと選択頻度の報告だけに使用し、window選択規則には使わない。

## 10. 独立validationと凍結結果

Selectionで選んだwindowだけを別のFineWeb-Edu 200文書で検証する。95% bootstrap CI下限が0より大きいことが続行条件である。

| 項目 | 結果 |
|---|---:|
| 選択window | decoder block 0--5、`[0,6)` |
| selection median restoration | 81.59% |
| selection 95% CI | [78.33%, 86.61%] |
| validation median restoration | 77.23% |
| validation 95% CI | [68.53%, 81.74%] |
| random-window control | decoder block 20--25、`[20,26)` |

Validation合格後、`[0,6)`はtask、seed、training cycle、behavior結果に応じて再選択しない。Random-window controlも、非重複の同幅windowから固定seedとSHA-256規則で一つだけ抽選している。

# Step 2：因果窓へ教師信号を局在させる学習

## 11. 学習仮説

Kojima型の出力分布整合は、typo入力でもclean入力と同様の予測分布を出すようStudentを教える。Proposalはこれに、次の小さなinductive biasを加える。

> Activation Patchingで回復に十分だった早期residual座標において、typo Studentの状態をclean Teacherの状態へ近づける。

State教師の追加価値を分離するため、proposalとbaselineはBase revision、data order、realized typo、LoRA容量、optimizer、student-token budgetを一致させる。

## 12. 学習データ

現行Cycle 3のlong-run streamはFineWeb-Edu 100%である。

| 項目 | 値 |
|---|---|
| dataset | `HuggingFaceFW/fineweb-edu` |
| subset | `sample-10BT` |
| revision | `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5` |
| 凍結training records | 157,032 |
| 凍結unique source tokens | 64,000,037 |
| long-run student-token budget | 64,000,000 |
| 最大系列長 | 512 |
| reasoning benchmark text | 0% |
| natural typo corpus context | 0% |

GSM8K、MMLU、ARCの問題・答えは64M training streamに入らない。GitHub Typo Corpusのcontextも直接学習せず、train側から得た文字置換統計だけをgeneratorへ使う。

`student tokens`はStudentへ入力されたnon-padding token数である。各行では通常、Teacherのclean forwardとStudent forwardを行うため、総model-forward token数は64Mより大きく、概ねTeacher分を加えた約2倍相当になる。State条件ではhidden-state保持とbackwardの追加コストもある。

## 13. Clean/noisyの1:1構成

Student入力を文書単位で厳密に交互に並べる。

| 行 | Teacher入力 | Student入力 | output loss | state loss |
|---|---|---|---|---|
| clean | clean | 同じclean | あり | 0 |
| noisy | clean | 対応するtypo | あり | state条件だけあり |

設定中の`explicit_clean_pair_probability=0`はclean行がないという意味ではない。別の`exact-alternating-clean-noisy` schedulerが50:50を保証する。

Clean行にもBase Teacherとのoutput KLを課すため、同じ目的関数がself-distillation型のclean anchorとして働く。

## 14. 学習用typo

Noisy行では、record ID、epoch、counterを鍵とする決定的generatorでtypoを作る。全armは同じsource順とrealized typoを使用でき、resume後も再現する。

編集語数の分布:

| 編集語数 | 確率 |
|---:|---:|
| 1 | 0.50 |
| 2 | 0.30 |
| 3--4 | 0.20 |

操作の分布:

| 操作 | 確率 |
|---|---:|
| keyboard-neighbor substitution | 0.2667 |
| deletion | 0.2667 |
| duplication | 0.2667 |
| natural typo統計に基づくsubstitution | 0.20 |

Natural substitution統計は学習用に分離したGitHub typo recordsから推定する。Adjacent transpositionは学習から除外し、unseen-operation評価へ保持する。

学習時の対象語は英字2文字以上で、複数編集時には異なる原語を選ぶ。AttnLRPは使わない。

## 15. Clean/typo token alignment

Typoによりtoken数とtoken境界が変わるため、clean/typoの同じtoken indexを直接比較しない。文字spanを介して次を構築する。

- exact unchanged token pair
- 非編集tokenを予測するcausal-logit位置pair
- clean側の編集語word-final token
- typo側の編集語word-final token

編集語そのものを予測するtarget tokenはoutput lossから除外する。これは誤綴り文字列自体を正解targetとして覚えさせることを避けるためである。

必要なalignmentを推測で補完せず、不成立pairは棄却する。Runtimeでalignment errorが生じた場合は学習を停止する。

## 16. Output distribution matching loss

記号を次のように置く。

- (x_i^c): clean入力
- (x_i^s): Student入力。clean行では (x_i^c)、noisy行では (x_i^p)
- (T): frozen Base Teacher
- (S_\theta): LoRA Student
- (A_i): 非編集targetのaligned causal-logit位置pair集合

Forward KLを、

\[
\mathcal L_{\mathrm{out}}
=
\frac{1}{\sum_i |A_i|}
\sum_i\sum_{(u,v)\in A_i}
KL\!\left(
p_T(\cdot\mid x_i^c,u)
\parallel
p_{S_\theta}(\cdot\mid x_i^s,v)
\right)
\]

とする。

- temperatureは1.0
- Teacher logitsはdetach
- log-softmaxとKLはfp32
- gradientはStudent LoRAだけへ流す
- gradient-accumulation batch全体の有効aligned token総数で正規化する
- sampleごとの平均を単純平均しない

Clean行では (x_i^s=x_i^c) なので、このloss自体がBaseからのdriftを抑える。

## 17. Localized residual-state loss

Noisy行の編集語word-final tokenだけで、選択window (W^*=[0,6)) のcomplete block residualを近づける。

- (E_i): clean/typo編集語word-final位置pair集合
- (h^l_{T,u}): Teacherのblock (l) 出力residual
- (h^l_{S,v}): Studentのblock (l) 出力residual

\[
d_{\cos}(a,b)
=
1-
\frac{a^\top b}{\max(\lVert a\rVert_2\lVert b\rVert_2,\epsilon)}
\]

\[
\mathcal L_{\mathrm{state}}
=
\frac{1}{|W^*|\sum_i|E_i|}
\sum_i\sum_{(u,v)\in E_i}\sum_{l\in W^*}
\operatorname{clip}_{[0,2]}
\left[
d_{\cos}
\left(
\operatorname{sg}(h^l_{T,u}),
h^l_{S,v}
\right)
\right]
\]

ここで、

- εは (10^{-8})
- cosineはfp32
- Teacher stateはstop-gradient
- clean行のstate lossは0
- accumulation batch全体の編集座標総数で正規化
- lossは構造的に0から2に有界

である。Cycle 1のper-example teacher-energy正規化relative MSEはheavy tailを生じたため、現行法では使わない。

## 18. 総lossとstate係数

Proposal:

\[
\mathcal L_{\mathrm{proposal}}
=
\mathcal L_{\mathrm{out}}
+
\lambda_{\mathrm{state}}
\mathcal L_{\mathrm{state}}
\]

Output-matching baseline:

\[
\mathcal L_{\mathrm{output-only}}
=
\mathcal L_{\mathrm{out}}
\]

\(\lambda_{\mathrm{state}}\)を直接手動指定せず、最初の8 noisy pairで各lossのLoRA gradient normを別々に測る。

\[
\lambda_{\mathrm{state}}
=
\rho
\frac{
\frac{1}{8}\sum_{i=1}^{8}
\lVert\nabla_\theta\mathcal L_{\mathrm{out},i}\rVert_2
}{
\frac{1}{8}\sum_{i=1}^{8}
\lVert\nabla_\theta\mathcal L_{\mathrm{state},i}\rVert_2
}
\]

正式値は、

\[
\rho=0.05
\]

であり、導出後の係数は学習中に変更しない。Seed 42の記録値は次のとおりである。

| 条件 | state範囲 | 導出された \(\lambda_{\mathrm{state}}\) |
|---|---|---:|
| proposal | block 0--5 | 0.2022819501 |
| random-window | block 20--25 | 0.7746756726 |
| all-layer | block 0--33 | 0.3938028647 |

Raw係数が異なるのはscopeごとの未加重state gradient normが異なるためである。初期の重み付きstate gradientをoutput gradientの5%へ揃える。

`rho=0.10`のpilotはstartup安全gateで停止した。`rho=0.20`は正式手法でも登録済みfallbackでもなく、現行proposalへ含めない。

## 19. LoRAとoptimizer

| 項目 | 値 |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 8 |
| LoRA scaling \(\alpha/r\) | 0.5 |
| dropout | 0 |
| bias | none |
| LoRA配置 | 全34 decoder blocks |
| target modules | q/k/v/o、gate/up/down projections |
| optimizer | AdamW |
| learning rate | (10^{-4}) |
| weight decay | 0.01 |
| scheduler | constant-with-warmup |
| warmup ratio | 0.0 |
| micro batch | 1 sequence |
| gradient accumulation | 32 micro steps |
| max gradient norm | 1.0 |
| gradient checkpointing | enabled |
| checkpoint間隔 | 50 optimizer steps |
| max optimizer steps | 10,000、安全上限 |
| long-run停止条件 | 64M student tokens |
| training seeds | 42, 43, 44 |

### 19.1 「局在」の正確な意味

LoRAは全layerへ配置される。Output matchingも全ての有効aligned targetで計算され、layer 0--5だけへ制限されない。

局在するのは、

- state lossを計算するlayer
- state lossを計算するtoken位置

である。全層LoRAにより、Studentは対象stateを作るための上流変更と、出力を保つための下流調整の両方を学習できる。

## 20. Clean性能を守る仕組み

1. Student入力の50%をclean文書にする。
2. Clean行にもBase Teacherとのoutput KLを課す。
3. Base parameterをfreezeする。
4. 更新をLoRAへ限定する。
5. LoRA scalingを0.5にする。
6. Held-out clean KLとPPLを学習中に監視する。
7. Safety gate違反時には停止し、違反前checkpointへrollbackする。

独立clean KL lossを置かないのは、clean行のoutput matchingが同じ役割を既に担うためである。Loss項と係数を増やさず、output-matching baselineとの比較可能性を維持する。

## 21. 学習中monitorと停止条件

学習中にtask accuracyを見ない。T0 monitorは次だけを使う。

| monitor | 件数 | 指標 |
|---|---:|---|
| held-out FineWeb-Edu clean | 200 | clean KL、PPL比 |
| held-out natural clean/typo | 100 pair | aligned clean--typo KL |

主な停止条件:

- clean `KL(Base || Student) > 0.03` が2回連続
- `PPL(Student) / PPL(Base) > 1.02` が2回連続
- startup区間でweighted state/output gradient比が0.5超を3回連続
- alignment error
- NaNまたはInf loss
- non-finite gradient norm

Task accuracyをmonitorしないのは、checkpointを評価benchmarkへ逐次適合させないためである。

## 22. 生train lossが激しく上下する理由と正常性判定

W&Bのmicro-batch系列は、収束判定にそのまま使えない。

1. Clean/noisyを厳密に交互投入する。Clean self-distillation lossは小さく、noisy lossは大きいため、系列には構造的な鋸歯状変動が入る。
2. 各micro-batchは別のFineWeb文書で、長さ、編集数、操作、tokenization変化が異なる。
3. 64M unique streamをほぼ単一epochで読むため、同じ固定training setへ反復fitしたときのような単調低下は期待しない。
4. W&Bの`train/loss/output`は32 micro-batchesのper-record KLを算術平均した表示値である。一方、実際にbackwardされる`train/objective/output`は、32 micro-batchesを有効aligned-token数で重み付けして正規化した値である。

したがって正常性は、少なくとも次を分けて監査する。

- optimizer-step objectiveの50--100 step移動平均
- clean/noisy別のloss分布
- state項とoutput項の別系列
- gradient norm、clip発火、NaN/Inf
- held-out clean KLとPPL
- held-out natural clean--typo KL
- checkpointごとの固定tune評価
- Base、baseline、proposalで同一data/typoを読めているか

Raw lossが上下しているという観察だけでは不安定性を証明しない。一方、平滑化後も悪化する、gradientが継続的にclip上限へ張り付く、clean gateを超える、またはcheckpoint間の固定probeが一貫して悪化する場合は、評価結果の前提となる学習健全性を再検討する。

### 22.1 完了済み10M runの健全性監査

W&B表示への懸念を受け、seed 42の10M output-matching baselineとproposalについて、全766 optimizer steps、各32 micro-batches、合計24,512 micro-batchesを再集計した。

| 指標 | output matching | proposal |
|---|---:|---:|
| 完了student tokens | 10,001,217 | 10,001,217 |
| total objective、first 100平均 | 0.008734 | 0.031432 |
| total objective、last 100平均 | 0.007496 | 0.024935 |
| total objective変化 | -14.2% | -20.7% |
| proposalのweighted state変化 | -- | -27.5% |
| gradient norm最大 | 0.313 | 0.256 |
| clip閾値1.0超過 | 0/766 | 0/766 |
| NaN/Inf | 0 | 0 |
| clean KL最大 | 0.00248 | 0.00380 |
| clean KL gate | 0.03 | 0.03 |
| PPL ratio最大 | 1.0010 | 1.0019 |
| PPL gate | 1.02 | 1.02 |
| natural typo KL ratio、終盤 | 約0.851 | 約0.828 |

Baselineのper-record output KLは、clean平均0.00215に対しnoisy平均0.01577であり、noisyの最大値は0.608だった。一方、32件をaligned-token数で正規化したoptimizer objectiveの最大値は0.01771だった。従って、表示上の大きなspikeは主として難しいnoisy文書のbatch分散であり、optimizer updateの発散にはなっていない。

Step、micro-step、student-token countは連続し、baseline、proposal、random-window、all-layerでrecord列とedit-count列も一致した。以上から、

> baselineとproposalの双方が数値的に正常な学習を行えておらず、10M checkpoint評価そのものが無効である。

という強い懸念は、現在保存されている証拠とは整合しない。

ただし、proposalのgradient calibrationには別の留保がある。\(\rho=0.05\)は8較正例におけるgradient normの**平均同士の比**を固定するだけであり、各exampleまたは各optimizer updateでstate/output比を5%へ維持するものではない。学習中の単一noisy-example proxyではratioのmedianが約0.279、p95が約1.35であった。Total gradient、loss、clean monitorは安定しているため数値発散の証拠ではないが、

> State信号が学習中いつでもoutput信号の5%に保たれた。

とは説明できない。Proposalがbaselineに勝たない場合、「causal targetが無効」と「state用量の実効値が想定と異なる」を完全には分離できないため、今後はaccumulation batch全体について各項のgradient normとgradient cosineを記録する必要がある。

## 23. 現行学習で使わないloss・機構

- gold answerへのcross entropy
- typo textをtargetにした通常のnext-token LM loss
- 独立clean KL loss
- neuron/head relative MSE
- SAE feature loss
- dynamic loss weighting
- rho grid search
- AttnLRPによるtraining typo選択

# 比較対象と対照実験

## 24. 比較arm

| 条件 | 学習内容 | state位置 | LoRA |
|---|---|---|---|
| 学習なしBase | なし | なし | なし |
| Kojima型output matching | \(\mathcal L_{out}\) | なし | 全層 |
| 現行proposal | \(\mathcal L_{out}+\lambda\mathcal L_{state}\) | 因果窓0--5 | 全層 |
| Random-window state | 同じ2項loss | 凍結窓20--25 | 全層 |
| All-layer state | 同じ2項loss | 全層0--33、編集語末のみ | 全層 |

すべて同じBase、source order、realized typo、clean/noisy比、LoRA容量、optimizer、token budget、seedで比較する。

## 25. 各controlが答える問い

### 25.1 学習なしBase

追加学習なしのclean/typo性能を示し、絶対的なclean非劣性とtypo改善の基準になる。

### 25.2 Kojima型output matching

State教師を追加せず、clean/noisy間のoutput matchingだけでどこまで頑健化できるかを測る中心baselineである。

共通する中心要素は、frozen self-teacher、clean:noisy 1:1、編集語target除外、output distribution matching、LoRAである。ただしdataset、model、全実装条件まで先行研究のrunと同一ではないため、現段階では「Kojima型baseline」または「同実装内matched-budget baseline」と表現する。

### 25.3 Random-window state

同じlayer数、token位置、state loss、初期gradient比を用い、layerだけを非因果windowへ変える。

> State lossを足しただけで効いたのか、Activation Patchingで選んだ場所に情報があるのか。

を分離する。

### 25.4 All-layer state

全34 layersの編集語末位置へ同じstate lossを与える。

> 因果窓へ絞ることが広いstate supervisionより有効か、または同等性能を狭い教師範囲で達成できるか。

を測る。これは全層・全token alignmentではなく、**全層×編集語末token**である。

## 26. 10Mと64M比較の状態

10M student-token、seed 42では、Base、output matching、proposal、random-window、all-layerのtune比較が完了している。64M matched-budget configが実装済みなのはoutput matchingとproposalで、現在は段階実行中である。

Random-windowとall-layerを必要な予算・3 seedsで比較しなければ、因果的localizationの確証的な新規性は主張できない。先にproposalがoutput matchingを上回る兆候を確認してから高価なcontrolsを拡張する段階投入は、証拠要件を弱めるものではなく計算資源配分の規則である。

# Step 3：学習後モデルの凍結評価

## 27. 評価の基本単位

全arm・全seedが同じitemのclean版と凍結済みtypo版を読む。

| | clean入力 | typo入力 |
|---|---:|---:|
| Base | \(B_c\) | \(B_t\) |
| Adapter | \(A_c\) | \(A_t\) |

確証的estimandは二つだけである。

\[
\Delta_{clean}=\operatorname{Acc}(A_c)-\operatorname{Acc}(B_c)
\]

\[
\Delta_{typo}=\operatorname{Acc}(A_t)-\operatorname{Acc}(B_t)
\]

Cleanは非劣性、typoは優越性として判定する。Clean accuracyを落とすだけでもclean--typo gapは縮むため、gap縮小単独を成功指標にしない。

評価母集団はBaseのclean-correct/typo-wrongへ条件付けない固定無条件サンプルである。

## 28. 評価tier

凍結protocolは `typo-robustness evaluation protocol v1.4` である。

| Tier | 用途 | 開封規則 |
|---|---|---|
| monitor | 学習中の数値安全性 | task accuracy禁止 |
| tune | cycle・ablation選択 | 反復可、headlineに不使用 |
| pre-PR gate | 凍結候補の確証確認 | arm・3 seeds・config・checkpoint hash固定後に1回 |
| final test | 論文headline | pre-PR合格後、全設計固定後に1回 |

### 28.1 Tune

- GSM8K、MMLU、ARC-Challengeの合計500 task records
- FineWeb-Edu clean 200文書
- natural clean/typo LM 100 pair
- natural-injection task 100 pair

### 28.2 Pre-PR gate

各500件、合計2,500 items:

- GSM8K
- MMLU
- ARC-Challenge
- MMLU-Pro
- CommonsenseQA

Corpus batteryはFineWeb-Edu 1,000、Dolma 500、natural LM 500 pairである。

### 28.3 Final test

- GSM8K 500
- MMLU 500
- ARC-Challenge 500
- MMLU-Pro 500
- CommonsenseQA 500
- MATH-500 440

合計2,940 task items。Corpus batteryはFineWeb-Edu 1,000、Dolma 1,000、natural LM 1,000 pairである。

MATH-500の440件は、4つの異なる適格語を持つというv1.4の完全なeligibility条件を満たす数である。旧文書の466件はv1.1時点の旧censusであり使用しない。

## 29. 学習・localizationとの分離

全poolは次から分離する。

- training IDs
- localization selection IDs
- localization validation IDs
- Cycle 1 pilot IDs
- 他tier IDs
- exact duplicate
- near-duplicate group

Natural typoはrepository単位で分割し、natural-injection辞書はcorrected word単位でもtrain/tune/pre-PR/finalへ排他的に分ける。

## 30. Primary typo条件

Primaryは`random-2`である。

- question中の異なる適格語2語
- attributionなしで一様選択
- keyboard-neighbor substitution、deletion、duplication
- seed 42で実現typo textを凍結
- 全arm・seedが同じfileを使用

対象語は英字3文字以上とし、few-shot、answer-option本文、gold answer、option label、数字、数式、URL、email、identifierを編集しない。

## 31. Secondary typo条件

- `random-1`
- `random-4`
- 学習からhold-outした`transposition-2`
- 評価専用辞書によるnatural-injection
- held-out natural clean/typo LM pairs

Attribution-4はmodel-specificな別登録stress diagnosticとしては可能だが、v1.4のprimary batteryには入らず、gateへ影響しない。

## 32. Generationとanswer extraction

- task別few-shot CoT prompt
- GSM8K 8-shot
- MMLU、MMLU-Pro、ARC-Challenge、CommonsenseQA 5-shot
- MATH-500 4-shot
- greedy decoding
- bfloat16
- max 512 new tokens
- task別extractorとdeterministic fallback
- extraction不能は全armで不正解

Few-shot examplesは常にcleanで、questionだけを編集する。

## 33. Paired transitionと統計

各比較では、wrong-to-wrong、wrong-to-right、right-to-wrong、right-to-rightの4象限を必ず報告する。Netだけを提示しない。

Task-equal macroは、

\[
\Delta_{macro}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{1}{|I_k|}
\sum_{i\in I_k}
(y_{A,i}-y_{B,i})\times100
\]

で計算する。二値paired比較には両側exact McNemar testを併記する。

Accuracy intervalはtask-stratified paired bootstrap 10,000回で計算する。3-seed estimateはlearning seedを外側、task内itemを内側に再標本化するhierarchical bootstrapを用いる。Two-sided 95% CIと、gate用one-sided 95% lower boundを保存する。

## 34. Clean corpus評価

Teacher-forced negative log likelihoodから、

\[
\operatorname{PPL}
=
\exp\left(
\frac{\sum_t-\log p(x_t\mid x_{<t})}{N_{tokens}}
\right)
\]

を計算し、

\[
\operatorname{PPL\ ratio}
=
\frac{\operatorname{PPL}(Adapter)}{\operatorname{PPL}(Base)}
\]

を報告する。FineWeb-Edu clean上では (KL(Base\parallel Adapter)) のmedianとp95も報告する。

Natural pairではexact unchanged character spanから非編集next-token座標を対応付け、編集語targetを除外する。

## 35. Mechanistic paired-patching audit

学習後モデル自身のclean runをdonorとし、同じモデルのtypo runへ凍結windowをpatchする。

\[
G_i
=
1-
\frac{\overline{KL}_{i,patched,2:16}}
     {\overline{KL}_{i,untreated,2:16}}
\]

BaseからAdapterへの追加patch余地の減少を、

\[
\operatorname{PatchGainReduction}
=
\frac{\overline G_{Base}-\overline G_{Adapter}}
     {\overline G_{Base}}
\]

で表す。

- 正でaccuracyも改善: external patchが担った修復を内在化した可能性
- accuracy改善、reductionなし: downstream compensationの可能性
- reductionあり、accuracy改善なし: 内部表現は動いたが実用改善なし

Patch audit、state loss、state distanceはmechanistic diagnosticであり、behavior gateをblockしない。

## 36. 凍結gate

Baseに対する主要gateは次のすべてである。

### 36.1 Clean非劣性

\[
\widehat\Delta_{clean}\ge-1.0\ \text{point}
\]

かつone-sided 95% lower boundが (-1.0) pointより大きい。各taskのclean point estimateは (-3.0) pointsより大きくなければならない。

### 36.2 Typo優越性

\[
\widehat\Delta_{typo}\ge+2.0\ \text{points}
\]

かつone-sided 95% lower boundが0より大きい。

### 36.3 Corpus保持

\[
\operatorname{PPL}(Adapter)/\operatorname{PPL}(Base)\le1.02
\]

\[
\operatorname{median}KL(Base\parallel Adapter)\le0.03
\]

### 36.4 Seed方向一致

3 seeds中2 seeds以上でclean変化が非負、typo変化が正である。

### 36.5 Natural typo非劣化

Natural-injectionの点推定が (-1.0) point以上、one-sided lower boundが (-2.0) pointsより大きく、natural LM KL gapがBaseより拡大しない。

## 37. Proposal固有の新規性判定

Base gateを通るだけでは、output matchingに対する新規性は立たない。

| 必要比較 | 意味 |
|---|---|
| proposal > output matching | state教師の増分価値 |
| proposal > random window | causal target選択の情報量 |
| proposal ≥ all-layer state | 局在しても性能を失わない、または局在が有利 |
| proposal clean非劣 | 頑健化のためにclean能力を犠牲にしていない |

結果別の主張範囲:

| 結果 | 主張可能範囲 |
|---|---|
| proposal > output、> random、≥ all-layer | 因果的target選択が学習上有効 |
| proposal > output、≈ random | State整合は有効だがcausal selectionの価値は未支持 |
| proposal ≈ all-layer > output | 狭い教師範囲で全層と同等という効率・解釈性 |
| proposal ≈ output、auditだけ固有に改善 | 行動的優位なし。機構差だけ |
| proposal ≤ output | 現形式のlocalized residual-state補助信号は不成立 |

# 既知の結果・限界・解釈規律

## 38. 10M seed-42 tune結果

以下は反復可能なtune結果であり、pre-PR/finalの確証結果ではない。

| 条件 | clean acc | typo acc | task-equal Δclean vs Base | task-equal Δtypo vs Base | patch gain reduction |
|---|---:|---:|---:|---:|---:|
| Base | 83.0% | 80.0% | 0 | 0 | 0 |
| output matching | 82.4% | 82.6% | -0.60pt | +2.60pt | 23.83% |
| proposal、0--5 | 82.6% | 82.4% | -0.40pt | +2.40pt | 30.34% |
| random、20--25 | 82.2% | 82.6% | -0.80pt | +2.61pt | 31.74% |
| all-layer、0--33 | 82.4% | 82.0% | -0.60pt | +2.00pt | 40.32% |

Proposalと他の学習条件のtypo差は、output matching比 (-0.20) point、random比 (-0.20) point、all-layer比 (+0.40) pointで、いずれも95% CIは0を跨いだ。

従って、10M、seed 42では次だけが言える。

- 大きなclean driftを避けながらoutput matching系の頑健化を実行できた。
- Proposalのstate信号に、output matchingまたはrandom windowを上回る行動的優位は確認できなかった。
- Patch gain reductionもproposal固有ではなかった。
- 1 seed・tune結果なので確証的結論ではない。

## 39. 主要な理論上の限界

### 39.1 Patch可能性と学習可能性

外部からclean stateを与えることと、曖昧なtypoだけから同じstateを生成することは異なる。Teacher stateにtypoから一意に復元できないclean文脈情報が含まれる場合、raw residual全体への一致には到達不能な床がある。

### 39.2 Window幅

`round(L/6)`は自然定数ではない。非恣意性は、最適性の主張ではなく、結果前固定、全深度scan、独立validation、再選択禁止、random/all-layer controlsという手続きで担保する。

### 39.3 同一Teacher

同じBaseをTeacherにするため、新しい知識や能力を注入しない。Clean時に既に持つ能力をtypo時にも引き出せるようにする自己蒸留である。

### 39.4 先行研究との表現

同じ64M予算でoutput-only baselineを上回った場合でも、dataset、model、全実装条件が先行研究と同一でない限り、「Kojima et al.そのものを上回った」ではなく、「64M matched-budgetのKojima型baselineを上回った」と表現する。

## 40. Source of truth

旧training planにはCycle 1以前の4項loss、reasoning mixture、selected-layer LoRA等が残る。現行Cycle 3については、次をsource of truthとする。

1. Cycle 3 training config
2. generic joint-window localization configと凍結artifact
3. evaluation protocol v1.4とmachine-readable config
4. 現在のloss/runtime implementation

古い文書や失敗runの設定を現行proposalへ混ぜない。
