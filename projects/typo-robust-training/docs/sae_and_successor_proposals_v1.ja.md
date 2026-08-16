# SAE診断トラックと後継proposal候補

更新日: 2026-08-16

## 0. 状態の区別

本書では、現在の正式proposal、SAE診断、将来の改善案を明確に分ける。

| 表記 | 意味 |
|---|---|
| 現行proposal | Activation Patchingで選んだraw residual windowへのstate蒸留。SAE不使用 |
| SAE診断 | typoによる表現逸脱の測定とfeature因果kill test。現行学習を変更しない |
| 条件付き後継 | 事前登録gateを通った場合だけ別studyとして実行可能 |
| 未登録案 | 理論候補。まだ現行手法でも実行計画でもない |

実装済み、事前登録済みだが未実行、未実装の研究案を混同しない。

## 1. 現行proposalとSAEの関係

現在の正式proposalは、SAEを使わない次の2項学習である。

\[
\mathcal L_{current}
=
\mathcal L_{out}
+
\lambda_{state}\mathcal L_{state}
\]

- \(\mathcal L_{out}\): clean Base Teacherとclean/typo Studentの、非編集aligned tokenにおけるoutput matching
- \(\mathcal L_{state}\): Activation Patchingで選んだlayer 0--5のresidual streamを、編集語末位置でclean Teacherへ近づけるcosine loss
- \(\lambda_{state}\): 初期gradient比 \(\rho=0.05\) から一度だけ導出して固定
- Teacher: frozen Gemma-3-4B-IT、clean入力
- Student: 同じGemma-3-4B-IT + all-linear LoRA

SAEは、このloss、adapter、評価条件に入っていない。現在の用途は二つだけである。

1. Typo入力がclean入力の表現多様体からどのように逸脱するかを、L0とenergyで診断する。
2. Residual全体より細かなfeature単位の後継proposalが因果的に成立するか、学習前にkill testする。

SAE kill testが不合格でも、直ちにlayer-level proposalが否定されるわけではない。不合格が否定するのは、

> 層窓が運ぶ因果情報を、現在のSAE feature基底へ十分疎に分解し、そのfeatureを学習targetにできる。

という追加仮説である。

# SAEの定義

## 2. 対象活性とarchitecture

SAEへ入力するのは、Gemma-3-4B-ITのcomplete decoder-block residual outputである。

\[
x\in\mathbb R^{d_{model}}
\]

| 項目 | 値 |
|---|---:|
| Base model | `google/gemma-3-4b-it` |
| decoder layers | 34 |
| \(d_{model}\) | 2,560 |
| expansion factor | 16 |
| \(d_{SAE}\) | 40,960 |
| feature activation | ReLU |
| regularizer | L1 |
| SAE precision | fp32 |
| Base activation取得 | bf16 |
| decoder normalization | 各optimizer stepで列を単位norm化 |

Encoderとdecoderは、

\[
z=\operatorname{ReLU}(W_{enc}x+b)
\]

\[
\hat x=Dz
\]

である。

- \(z\in\mathbb R^{d_{SAE}}\): sparse feature activation
- \(D\) の列 \(d_f\): feature \(f\) のdecoder方向
- \(\hat x\): SAE再構成
- \(\epsilon=x-\hat x\): feature基底で説明されない再構成誤差

Top-k制約は使わない。L0とenergyは、制約なしで実際に何featureが正に発火したかを必要とするためである。

## 3. SAE学習loss

\[
\mathcal L_{SAE}
=
\operatorname{mean}\left[(x-\hat x)^2\right]
+
\lambda_{L1}
\operatorname{mean}\left[\sum_f |z_f|\right]
\]

\(\lambda_{L1}\) はtypo挙動やtask accuracyでは選ばない。Clean FineWeb-Edu上だけで、

1. median L0が事前登録範囲 \([30,150]\) に入る候補を残す。
2. その中でFVUが最小の候補を選ぶ。

という規則を用いる。

許可された1回のamendment後のgridは、

\[
\lambda_{L1}\in\{0.01,0.1,1.0\}
\]

であり、calibration budgetは10M activation tokensである。本学習で選択された値はlayer 5が0.01、layer 20が0.1である。

## 4. L0

Token位置 \(t\) のL0を、正に発火したfeature数として定義する。

\[
L_0(x_t)=\sum_f\mathbf 1[z_{t,f}>0]
\]

Clean--typo pairでは、

\[
\Delta L_0
=
L_0(x_{typo})-L_0(x_{clean})
\]

を測る。正の値は、typo入力がclean入力より多くのfeatureを動員したことを意味する。ただしL0は相関的なOOD診断であり、単独では因果性を意味しない。

## 5. Energy score

Clean SAE学習分布上のfeature発火率を、

\[
p_f=\Pr(z_f>0)
\]

とし、tokenごとの再構成誤差、

\[
e(x)=\lVert x-Dz\rVert_2^2
\]

のclean中央値を \(s\) とする。Energyを、

\[
F(x)
=
\frac{\lVert x-Dz\rVert_2^2}{s}
+
\sum_f z_f\log\frac{1-p_f}{p_f}
\]

で定義する。

- 第1項: SAEで説明しにくい残差
- 第2項: clean学習分布で希少なfeatureの発火

Energyも現行proposalの成功判定には使わず、Tier 3診断と将来のデータ選別候補に限定する。

# SAE作業パッケージ

## 6. WP-1：SAE学習

**状態: 実装済み・100M activation-token本学習完了**

学習対象は3本である。

| 対象 | 目的 |
|---|---|
| layer 5、seed 42 | 因果窓 `[0,6)` の出口を表現 |
| layer 5、seed 43 | feature結果の初期化seed再現性 |
| layer 20、seed 42 | 相対深さ20/34付近の中層OOD現象 |

学習データはtrain split内のclean FineWeb-Eduだけである。

- typo textをSAE学習へ入れない
- tune、pre-PR、final、localization selection/validationを除外
- current adapter runが使う保護prefixを除外
- record/source/group ID、正規化content、character 5-gram近重複を検査
- 最大系列長512
- 最低100M training activation tokens
- 別の10M clean activation tokensで \(p_f\) と \(s\) を計算

SAEをclean分布だけで学習することが、typo側のL0・energy増加をOOD逸脱として解釈する前提になる。

## 7. WP-2：SAE検収

**状態: 実装済み・gate事前登録済み・2026-08-16時点で検収実行中**

各SAEについて、以下を全て満たした場合だけWP-3以降で使用できる。

### 7.1 FVU

\[
\mathrm{FVU}
=
\frac{\sum_t\lVert x_t-\hat x_t\rVert_2^2}
     {\sum_t\lVert x_t-\bar x\rVert_2^2}
\le0.35
\]

少なくとも65%の分散を説明することを要求する。

### 7.2 Sparsity

\[
30
\le
\operatorname{median}_tL_0(x_t)
\le150
\]

### 7.3 Dead feature率

\[
\frac{1}{d_{SAE}}
\sum_f\mathbf 1[p_f<10^{-5}]
\le0.20
\]

### 7.4 Splice KL

Clean runでprobe layerのresidual \(x\) をSAE再構成 \(\hat x=Dz\) に置換し、元の出力とのKLを測る。

\[
\operatorname{median}
KL\!\left(
p_{original}
\parallel
p_{SAE\ splice}
\right)
\le0.15\ \text{nats/token}
\]

これは、SAEが診断対象のmodel behaviorを大きく壊していないことを要求するgateである。

### 7.5 必須artifact

- feature firing probability \(p_f\)
- clean reconstruction-error median \(s\)
- model、data ID、weights、statisticsのhash

不合格時は、登録済み規則に従って1回だけ再学習できる。2回目も不合格ならSAEトラックを停止する。

## 8. WP-3：Base統合診断

**状態: 予測と仕様を事前登録済み。専用runnerは未実装**

次の因果連鎖を検証する。

```text
早期層での編集語表現の破損
        ↓
中層での余剰・希少feature動員
        ↓
出力分布の変化
```

Localization validation用FineWeb-Edu 200件を用い、各例へ1 typoを加える。Baseで、

1. clean
2. typo
3. typo + `[0,6)` full-state patch

を比較する。編集語末位置と後続16 tokenでlayer 5/20のL0とenergyを測る。

事前登録予測:

\[
P1:\quad
L_0^{20}(typo)-L_0^{20}(clean)>0
\]

\[
P2:\quad
L_0^{20}(patched),F^{20}(patched)
\text{ が typo より clean へ近づく}
\]

P2が成立すれば、早期patchが下流の余剰feature動員を因果的に抑えることを示唆する。不成立でも既存のlayer-window patch証拠は独立に残るため、現行runや評価は変更しない。

## 9. WP-4：Checkpoint遡及診断

**状態: 予測と仕様を事前登録済み。専用runnerは未実装**

保存済みadapter checkpointを後解析し、学習に伴う表現逸脱を追う。

使用データ:

- natural clean--typo 100 pair
- synthetic fixed 100 pair

比較条件:

- output matching
- causal-window state proposal
- random-window state
- all-layer state
- 追加seed

各checkpointのlayer 5/20について、編集語末と後続16 token平均の、

\[
\Delta L_0=L_0(typo)-L_0(clean)
\]

\[
\Delta F=F(typo)-F(clean)
\]

およびclean側energyのBase比driftを記録する。

事前登録予測:

- P3: causal-window stateはoutput matchingよりlayer 20の \(\Delta L_0,\Delta F\)を強く抑える。
- P4: 両条件ともclean側energy driftは小さい。

これらはTier 3診断であり、checkpoint選択、学習停止、確証評価のgateには使用しない。

## 10. WP-5：Feature因果kill test

**状態: operatorとgateを事前登録済み。専用runnerは未実装**

目的は、layer 5 residualが運ぶ因果情報をSAE feature成分とreconstruction errorへ分解し、feature学習を始める前に介入で検証することである。

全既存tierとID非重複のFineWeb-Edu 200件を新たに固定し、各例へ1 typoを加える。Layer 5の編集語末位置で、

\[
h=Dz+\epsilon
\]

と分解する。

### 10.1 Full-state単層patch

\[
h'=h_{clean}
\]

このrestorationを \(R_{full}\) とする。既知の6-layer patchも文脈用参照として併記する。

### 10.2 Feature swap

\[
h'=Dz_{clean}+\epsilon_{typo}
\]

Feature成分だけをcleanへ交換し、restorationを \(R_z\) とする。

### 10.3 Reconstruction-error swap

\[
h'=Dz_{typo}+\epsilon_{clean}
\]

誤差成分だけをcleanへ交換し、restorationを \(R_\epsilon\) とする。

### 10.4 Spurious-feature suppression

\[
S_i
=
\{f:z^{typo}_{i,f}>0\land z^{clean}_{i,f}=0\}
\]

\[
h'
=
h_{typo}
-
\sum_{f\in S_i}z^{typo}_{i,f}d_f
\]

Typoでだけ発火したfeatureを除去し、restorationを \(R_{sup}\) とする。

任意のsparsity curveとして、\(|z_{clean}-z_{typo}|\)上位 \(k\in\{8,32,128\}\) だけをswapする。

### 10.5 Readout

全operatorで既存のmulti-token KL restorationを使う。

\[
R_C
=
1-
\frac{
\sum_{t=2}^{16}KL(p_{clean,t}\parallel p_{C,t})
}{
\sum_{t=2}^{16}KL(p_{clean,t}\parallel p_{typo,t})
}
\]

統計はpaired bootstrap 10,000回、seed 42である。

### 10.6 Gate

\[
G1:\quad
\operatorname{median}(R_z)
\ge0.50\operatorname{median}(R_{full})
\]

\[
G2:\quad
\operatorname{median}(R_{sup})
\ge0.25\operatorname{median}(R_{full})
\]

G3は、layer 5の第2seed SAEでもG1/G2の方向が再現することを要求する。

解釈:

- \(R_z\)が大きい: 因果情報の相当部分がSAE feature成分にある。
- \(R_\epsilon\approx R_{full}\)かつ \(R_z\approx0\): 因果情報はSAEのdark matter側にあり、このfeature基底は学習targetに不適。
- \(R_{sup}\)が大きい: clean featureを完全復元せず、typo固有featureの除去だけでも修復できる。

Feature-targeted successorへ進むには少なくともG1とseed再現性G3が必要である。G2は、後継lossを片側抑制型にするか、feature一致型にするかを分ける。

## 11. WP-6：Feature-level後継arm

**状態: 文書草案だけ許可。現行decision tree決着前の学習実行は禁止**

### 11.1 G1・G2とも合格：片側spurious-feature抑制

\[
\mathcal L_{spur}
=
\frac1{|C|}
\sum_i\sum_{f\in S_i}w_fz^{typo}_{i,f}
\]

\[
w_f=\log\frac{1-p_f}{p_f}
\]

\[
\mathcal L
=
\mathcal L_{out}
+
\lambda_{spur}\mathcal L_{spur}
\]

利点は、元のclean語を一意に復元できない例でclean state全体への到達不能な一致を要求せず、cleanで発火しない余剰featureだけを抑える点である。

### 11.2 G1合格・G2不合格：Feature部分空間整合

WP-5で因果性を確認したdecoder方向が張る固定部分空間への射影を \(P\) とする。

\[
\mathcal L_{feature}
=
1-
\cos(Ph_{clean},Ph_{typo})
\]

\[
\mathcal L
=
\mathcal L_{out}
+
\lambda_{feature}\mathcal L_{feature}
\]

Raw residual lossをこれで**置換**し、3項lossへ増やさない。

必須control:

- 同数・同発火率または希少度のrandom features
- all features
- 同じlayer/positionのraw residual loss
- output matchingのみ
- 学習なしBase

WP-5不合格ならfeature学習を行わず、SAEを診断専用に留める。

# 現状考えている改善案

## 12. 候補A：現行raw residual proposalの64M scale test

**状態: config実装済み・matched-budget比較の実行段階。新しい手法ではなく既存仮説のscale test**

### 仮説

10M student tokensではstate信号の追加効果をbehaviorで検出するには不足していた可能性がある。Kojima型の代表予算に揃え、差がdata scaleで現れるかを調べる。

### 条件

- output matchingのみ、64M student tokens
- output matching + layer 0--5 residual cosine、64M student tokens
- 同じ64M unique source stream
- 同じLoRA、optimizer、typo、clean:noisy 1:1
- 正式な \(\rho=0.05\) を維持
- window、loss、評価を変更しない

### 判定

64Mでもproposalがoutput matchingと必要なsame-budget controlsを上回らず、auditにも固有差がなければ、

> Activation Patchingでpatch可能なraw residual座標は、このcosine整合形式では追加の学習価値を示さない。

と結論する。結果を見てrhoやwindowを再調整しない。

## 13. 候補B：SAE feature proposal

**状態: WP-5合格時だけ許可される条件付き将来study**

### 仮説

Raw residual全体には修復に不要な情報やclean文脈固有情報が含まれる。因果修復に寄与する疎なfeature部分だけをtargetにすれば、到達不能な教師とclean干渉を減らせる。

| 現行法 | SAE feature案 |
|---|---|
| layer 0--5のresidual全次元 | WP-5で因果確認したfeaturesまたは部分空間 |
| clean state全体へのcosine | spurious suppressionまたは因果部分空間cosine |
| SAE不要 | accepted SAEと固定decoder方向が必要 |
| layer-level因果証拠 | feature-level kill testを追加要求 |

条件はWP-2合格、WP-5 G1とseed再現性合格、別studyとしての事前登録である。G2合格なら片側抑制、G2不合格なら部分空間整合とする。

## 14. 候補C：Causal Patch Distillation

**状態: 未実装・未登録。SAE feature仮説が不成立の場合の第一候補**

### 14.1 中心仮説

現行state lossはtypo Studentへclean residualそのものの再構成を要求する。しかし、曖昧なtypoからclean語を一意に復元できない場合、このtargetは原理的に到達不能になり得る。

Activation Patching後の出力分布は、clean stateの全成分ではなく、介入によって実際に生じた**修復挙動**を表す。そこでpatched typo Baseの出力をTeacherとし、patchなしStudentへ蒸留する。

### 14.2 Noisy行

1. Frozen Baseへcleanを入力し、layer 0--5の編集語末stateを取得する。
2. Frozen Baseへtypoを入力する。
3. Typo runの同じ座標へclean stateをpatchする。
4. Patched typo出力分布 \(q_{patch}\) を取得する。
5. Patchなしtypo Studentを \(q_{patch}\) へ近づける。

\[
\mathcal L_{patchKD}
=
\frac1{\sum_i|A_i|}
\sum_i\sum_{t\in A_i}
KL\!\left(
q_{patch,i,t}
\parallel
p_{S_\theta,i,t}
\right)
\]

TeacherとStudentが同じtypo token列を見るため、clean--typo間よりlogit位置対応が単純である。編集語targetは従来どおりmaskする。

### 14.3 Clean行

\[
\mathcal L_{clean-selfKD}
=
KL\!\left(
p_{Base}(\cdot\mid x_{clean})
\parallel
p_{S_\theta}(\cdot\mid x_{clean})
\right)
\]

Row種別で教師分布を切り替える単一output lossとして表せる。

\[
q_i=
\begin{cases}
p_{Base}(x_i^c) & clean\ row\\
p_{patched\ Base}(x_i^p;h_{Base}(x_i^c)) & noisy\ row
\end{cases}
\]

\[
\mathcal L=KL(q_i\parallel p_{S_\theta})
\]

State lossとrhoは不要になる。

### 14.4 必須control

- 学習なしBase
- Kojima型clean-output matching
- causal-window patched-teacher distillation
- random-window patched-teacher distillation
- 必要ならall-layer patched teacher

Window以外のdata、LoRA、token budget、clean/noisy比を一致させる。

### 14.5 利点とリスク

利点:

- Oracle patchをStudent weightsへ償却する構造が直接的
- Raw residual全次元一致もSAE feature仮定も不要
- `localize -> intervene -> distill -> audit`の閉ループになる
- Lossは一つで、rhoを持たない

リスク:

- noisy例ごとにclean Base、patched typo Base、Studentの3 forwardが必要
- patchが悪化させる例もあり得る
- training時だけoracle clean runを必要とする
- patch restorationが小さい例では教師信号が弱い
- 近接研究との重複を別途文献調査し、実行前に新studyとして登録する必要がある

## 15. 候補D：SAE energyによるdata selection

**状態: 未実装・将来study草案**

同じtypo数でも内部OOD逸脱が強い例ほど学習効率が高いという仮説を検証する。Noisy候補をSAE energyでscoreし、edit countとoperationのstratum内で高energy側を選ぶ。

LossはKojima型output matchingのままである。

\[
\mathcal L=\mathcal L_{out}
\]

必須control:

- 同じtoken budgetのrandom sampling
- severity、operation、文書長をmatchedしたrandom sampling
- high-energy対low-energy
- 固定clean保持評価

Energyは因果指標ではなく相関的OOD指標である。Severity交絡を制御し、causal-window proposalとは混ぜず独立armとして扱う。

## 16. 候補E：介入効果によるdata selection

**状態: 未登録の探索案。優先度はPatch Distillationより低い**

各pairのpatch restorationを、

\[
R_i(W^*)
=
1-
\frac{D_{patched,i}(W^*)}{D_{typo,i}}
\]

として事前計算し、patch-positiveまたは高restoration例へ学習を集中する。

必須control:

- 同件数・同token budgetのrandom selection
- edit count、operation、文書長matched random
- low-restoration examples
- output matchingのみ

全候補へのpatch forwardが高価で、修復しやすい例だけへの選択biasを持つ。Patch Distillationの方が連続的な介入情報を捨てないため、第一候補にはしない。

## 17. 候補比較

| 方法 | 因果情報 | 主変更 | 追加loss/係数 | 現在の状態 |
|---|---|---|---:|---|
| 64M residual-state | causal window | 学習量だけ増加 | なし | 実装・実行段階 |
| SAE feature | WP-5合格feature | state targetを疎なfeatureへ置換 | 1項・1係数 | 条件付き草案 |
| Causal Patch Distillation | patch後のoutput | Teacher targetを変更 | state項/rho不要 | 未登録案 |
| SAE energy selection | 相関的OOD score | noisy data selection | なし | 未登録案 |
| patch-effect selection | pairwise restoration | noisy data selection | なし | 未登録案 |

# 推奨decision tree

## 18. SAE側

```text
WP-2 SAE acceptance
  ├─ 不合格
  │    └─ SAEを後続実験へ使用しない
  │
  └─ 合格
       ├─ WP-3 / WP-4を診断として実行
       └─ WP-5 feature kill test
             ├─ G1 + seed再現性 合格
             │    ├─ G2合格
             │    │    └─ spurious-feature suppression study
             │    └─ G2不合格
             │         └─ feature-subspace alignment study
             └─ 不合格
                  └─ feature学習を中止しSAEを診断専用にする
```

## 19. 現行proposal側

```text
64M causal-window residual proposal
  ├─ output matchingとsame-budget controlsを上回る
  │    └─ 追加seed・凍結評価・cross-modelへ
  │
  ├─ behavior同等、mechanistic auditだけ固有に改善
  │    └─ 実用優位は主張せず機構差として限定報告
  │
  └─ performance・auditとも固有差なし
       ├─ WP-5合格
       │    └─ SAE feature proposalを別studyとして登録
       └─ WP-5不合格
            └─ Causal Patch Distillationを第一後継案として登録
```

SAE不合格と現行proposal不成立が同時に起きても、研究を監査だけで終える必要はない。その場合は、clean residualそのものではなく、因果patchが生んだ修復済みoutputを蒸留するCausal Patch Distillationへ移るのが、現在考えられる最も単純で因果仮説を保った再構成である。

## 20. 複雑化を防ぐ原則

1. 現行64M runの結果を見てrho、window、lossを変更しない。
2. 新案は現行studyへ継ぎ足さず、仮説を一つだけ変える別studyとして登録する。
3. Output matchingを必ず中心baselineにする。
4. Feature案はkill test合格前に学習しない。
5. State/feature/data selectionを一度に混ぜない。
6. Behavior評価、random control、all-layerまたはall-feature controlを揃える。
7. 診断指標が改善しても、clean非劣性とtypo優越性を代替させない。
