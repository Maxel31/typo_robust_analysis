# SAE診断トラックと後継proposal roadmap：現行仕様と未登録候補

更新日: 2026-08-16

対象: Gemma-3-4B-IT / SAE統合トラック
重要: **SAEは現行のresidual-state proposalには含まれない**

## 1. Executive summary

### 1.1 現在から後継候補までのdecision map

~~~text
現在
  ├─ 現行raw-residual proposalを64Mでoutput-matching baselineと比較
  └─ 独立SAE診断trackでWP-1 -> WP-2 -> WP-3/4/5

判定
  ├─ 64M proposalがbehavior controlsに勝つ
  │    └─ 現行proposalを追加seed・凍結評価・cross-modelへ
  ├─ WP-5 G1+G2+G3が合格
  │    └─ 事前registry amendmentがある場合だけsuppression studyを草案化
  ├─ WP-5 G1+G3だけ合格
  │    └─ 現registryが許すfeature-targeted study草案まで。学習は未認可
  └─ raw residualもSAE featureも不成立
       └─ Causal Patch Distillationを第一の別proposalとして登録

低優先の独立候補
  ├─ SAE energyによるdata selection
  └─ patch restorationによるdata selection
~~~

SAEの現在の役割は二つだけである。

1. **診断**: typo入力がclean入力の表現分布からどのように逸脱するかを、feature数 \(L_0\)、energy、再構成誤差で測る。
2. **学習前の因果検証**: pair-specificなfeature除去を将来の教師targetにできるか、feature patchのkill testで先に棄却可能にする。全feature swapの成功だけから固定feature集合を導かない。

現行proposalの学習はSAEを使わない。

\[
\mathcal L_{\mathrm{current}}
=
\mathcal L_{\mathrm{out}}
+
\lambda_{\mathrm{state}}\mathcal L_{\mathrm{state}}
\]

- \(\mathcal L_{\mathrm{out}}\): 非編集aligned tokenのoutput matching
- \(\mathcal L_{\mathrm{state}}\): Activation Patchingで選んだblock 0--5のraw residual cosine
- \(\rho=0.05\)から \(\lambda_{\mathrm{state}}\) を一度だけ較正

SAE診断が良く見えても、現行run、凍結評価、現行lossを変更しない。現registryはG1+G3だけをfeature-targeted successor-studyの**草案**条件とし、後継学習を一切認可していない。Pair-specific suppressionには、結果計算前のregistry amendmentでG1を維持したG2 routingを追加し、WP-5 G1/G2/G3を全て満たした後、別studyとして新たに事前登録する必要がある。Fixed-feature trainingにはさらに別のcausal selection / held-out kill testが必要である。

### 1.2 状態snapshot

| 要素 | 状態 | 現在の意味 |
|---|---|---|
| WP-1 SAE学習 | **完了済み** | layer 5×2初期化seed、layer 20×1 seed、各100M activation tokens |
| WP-2 SAE検収 | **実装・事前登録済み、検収実行段階** | FVU / L0 / dead / splice KLで使用可否を判定 |
| WP-3 Base統合診断 | **予測・3条件登録済み、集計amendment未完成、runner未実装** | early patchがmid-layer OOD指標を潰すか |
| WP-4 checkpoint診断 | **予測・data用途登録済み、集計amendment未完成、runner未実装** | 学習に伴う \(L_0\) / energy軌跡 |
| WP-5 feature kill test | **pre-execution amendment未完成、runner未実装** | core operator・G1--G3数値だけ登録済み。eligible/seed/前提gate/後継許可範囲を結果前に凍結する必要 |
| WP-6 feature後継study | **文書草案のみ・現registryでは学習未認可** | suppression routingは事前amendment + G1/G2/G3が必要。固定feature案には別kill testも必要 |
| 64M residual-state比較 | **現行studyの実行段階** | SAEとは独立した用量検証 |
| Causal Patch Distillation等 | **未登録案** | 現行studyにもWP-6にも未包含 |

本書では次を厳密に区別する。

| ラベル | 定義 |
|---|---|
| 現行proposal | Raw residual-state学習。SAE不使用 |
| 診断 | 学習や評価gateを変えないTier 3解析 |
| 事前登録済み | 実行前にoperator、データ、seed、閾値を固定済み |
| 条件付き後継 | kill test合格後に別studyとして登録可能 |
| 未登録案 | 理論候補。実装・実行の承認ではない |

### 1.3 SAE不合格が意味する範囲

WP-2またはWP-5不合格は、layer-level Activation Patchingの証拠を否定しない。不合格が否定するのは、より狭い仮説である。

> 因果窓が運ぶ修復情報を、今回学習したSAEのfeature成分またはpair-specificなtypo-only feature除去として回収し、後継targetにできる。

SAE feature仮説が不成立でも、raw residual proposalの成否は独立に64M比較で判定する。両方が不成立なら、clean residualそのものではなくpatch後の修復出力を蒸留する別proposalを新規登録する。

# Part I: SAE診断track

## 2. 理論、数式、用語

### 2.1 Superposition、polysemantic neuron、featureの非正準性

Transformerの一つのneuronは、複数の意味・機能を重ね合わせて表現するpolysemanticな単位になり得る。Cycle 1では13 neuron/head componentを選んだが、joint causal patchはKL restoration 4.36%、95% CI \([-4.52,12.38]\)であり、component-level state lossを大きく下げてもbehaviorは改善しなかった。

SAEはresidual streamを過完備な疎featureへ分解し、単一neuronより安定した候補単位を得る。ただし個別featureは初期化に依存し、異なるSAEのfeature index同士に自然な一対一対応はない。Featureは**非正準**であり、見つかったfeatureが因果的だとも限らない。

従って、本研究では次を分離する。

- \(L_0\)、energy、再構成品質: 大域的・相関的な診断
- Feature swap / suppression: feature成分の因果介入
- Pair-specific feature training: 事前amendment後にG1/G2/G3が揃った場合だけ草案化できる後継仮説
- Fixed-feature training: 別のcausal selection / held-out kill test合格後だけ許される後継仮説

### 2.2 対象activationとarchitecture

入力はGemma-3-4B-ITのcomplete decoder-block residual outputである。

\[
x\in\mathbb R^{d_{\mathrm{model}}},\qquad d_{\mathrm{model}}=2560
\]

SAEは次で定義する。

\[
z=\operatorname{ReLU}(W_{\mathrm{enc}}x+b_{\mathrm{enc}})
\]

\[
\hat x=Dz
\]

\[
\epsilon=x-\hat x
\]

| 記号 | 意味 |
|---|---|
| \(z\in\mathbb R^{d_{\mathrm{SAE}}}\) | sparse feature activation |
| \(D=[d_1,\ldots,d_{d_{\mathrm{SAE}}}]\) | feature decoder方向 |
| \(\hat x\) | SAEで説明されたresidual成分 |
| \(\epsilon\) | SAE基底で説明されない再構成誤差 |

| 項目 | 値 |
|---|---:|
| Base model | google/gemma-3-4b-it |
| decoder blocks | 34 |
| \(d_{\mathrm{model}}\) | 2,560 |
| expansion factor | 16 |
| \(d_{\mathrm{SAE}}\) | 40,960 |
| activation | ReLU |
| sparsity penalty | L1 |
| SAE training precision | fp32 |
| Base activation取得 | bf16 |
| decoder制約 | 各optimizer stepで列 \(d_f\) をunit norm化 |
| optimizer | Adam、lr \(3\times10^{-4}\)、betas \((0.9,0.999)\)、epsilon \(10^{-8}\) |
| activation batch | 2,048 token activations |
| shuffle buffer | 1,000,000 activations |

Top-kは使わない。診断で測る \(L_0\) とenergyが、固定kではなく自然に正発火したfeature数・強度を必要とするためである。

### 2.3 SAE training loss

\[
\mathcal L_{\mathrm{SAE}}
=
\underbrace{
\frac{1}{N d_{\mathrm{model}}}
\sum_{t=1}^{N}\sum_{j=1}^{d_{\mathrm{model}}}
(x_{t,j}-\hat x_{t,j})^2
}_{\mathcal L_{\mathrm{recon}}}
+
\lambda_{L1}
\underbrace{
\frac{1}{N}
\sum_{t=1}^{N}\sum_{f=1}^{d_{\mathrm{SAE}}}|z_{t,f}|
}_{\mathcal L_{\mathrm{L1}}}
\]

\(N\) はactivation batch内のtoken数である。再構成項はtoken・hidden element全体のmean、L1項はtokenごとにfeature絶対値をsumしてからtoken meanを取る。これは現行実装のelementwise MSEと一致する。

\(\mathcal L_{\mathrm{L1}}\) はfeature次元でmeanを取らないため、\(\lambda_{L1}\) の数値scaleは \(d_{\mathrm{SAE}}=40{,}960\) とこの正規化定義に依存する。Expansion factorやL1 normalizationを変えたSAEへ、0.01 / 0.1をそのまま移植してはならない。

\(\lambda_{L1}\) はtypo taskの結果では選ばない。Clean calibration activationだけで、

1. median \(L_0\in[30,150]\) の候補を残す。
2. その中でFVUが最小の候補を選ぶ。

許可済みの1回のamendment後のgridは \(\{0.01,0.1,1.0\}\)、各候補のcalibration budgetは10M activation tokensである。選択結果はlayer 5が0.01、layer 20が0.1である。

### 2.4 診断量

#### \(L_0\): 発火feature数

\[
L_0(x_t)=\sum_f\mathbf 1[z_{t,f}>0]
\]

\[
\Delta L_0=L_0(x_{\mathrm{typo}})-L_0(x_{\mathrm{clean}})
\]

正の \(\Delta L_0\) は、typo側がclean側より多くのfeatureを動員したことを表す。ただしこれは相関的なOOD診断であって、原因を示さない。

#### Reconstruction error

\[
e(x)=\|x-Dz\|_2^2
\]

大きいほど、そのresidualがSAE学習済みclean feature基底では説明しにくい。

#### Feature firing probability

\[
p_f=\Pr_{\mathrm{clean}}(z_f>0)
\]

Clean statistics streamで推定する。小さい \(p_f\) はclean分布で希少なfeatureを表す。

#### Energy score

Clean reconstruction errorの中央値を \(s\) として、

\[
F(x)
=
\frac{\|x-Dz\|_2^2}{s}
+
\sum_f z_f\log\frac{1-p_f}{p_f}
\]

と定義する。

- 第1項: feature基底で説明しにくいresidual
- 第2項: clean分布で希少なfeatureの発火

現行の診断実装はlogit発散を避けるため \(p_f\) を \([10^{-8},1-10^{-8}]\) へclampする。後継lossのrare-feature weightへ同じ値を流用するかは未登録であり、WP-6学習前に固定する。Energyは現行studyではTier 3診断だけ、将来studyでは独立data-selection候補として扱う。

### 2.5 用語表

| 用語 | 本書での定義 |
|---|---|
| feature | SAE latent \(z_f\) とdecoder方向 \(d_f\) の組 |
| L0 | 正に発火したfeature数 |
| dead feature | clean統計上 \(p_f<10^{-5}\) のfeature |
| dark matter | 再構成誤差 \(\epsilon=x-Dz\) 側の情報 |
| splice | model residualをSAE再構成 \(Dz\) で置換する介入 |
| spurious feature | paired typoで正発火し、対応cleanで発火しないfeature |
| kill test | 学習前に介入で因果十分性を検証し、不合格なら学習しない規則 |

## 3. 現行proposalとの関係とデータ境界

### 3.1 役割の分離

~~~text
現行proposal
  raw residual window [0,6)をcosine整合
  └─ SAEとは独立に64Mで成否判定

SAE診断
  clean residualでSAEを学習
  ├─ L0 / energyでtypo OODを記述
  ├─ patchが下流OODを因果的に抑えるか診断
  └─ feature patchが因果修復に十分かkill test

後継study
  事前amendment + G1/G2/G3合格時だけpair-specific suppressionを草案化
  fixed-feature案は別selection/held-out kill test合格後だけ新規登録
~~~

SAEの結果を用いて、進行中runのrho、window、loss、checkpoint、評価gateを変更しない。

### 3.2 データ分離

SAE学習へ使えるのは**training split内のclean FineWeb-Eduだけ**である。

| データ | SAE学習 | SAE検収 | WP-3 | WP-4 | WP-5 |
|---|---|---|---|---|---|
| SAE train clean | 可 | 不可 | 不可 | 不可 | 不可 |
| SAE clean statistics | 不可 | \(p_f,s\)算出 | 不可 | 不可 | 不可 |
| localization validation 200 | 不可 | 不可 | 可 | 不可 | 不可 |
| monitor natural/synthetic | 不可 | 不可 | 不可 | 可 | 不可 |
| 新規diagnostic 200 | 不可 | 不可 | 不可 | 不可 | 可 |
| tune / pre-PR / final | **全WPで学習・選択禁止** |  |  |  |  |

SAE trainから除外したadapter dataは、先行10M runを想定した先頭30,000 recordだけである。このprefixは概算約12.26M source tokensだが、この値は64M streamと初期eligibleの差から逆算したものでdedup amendment分を含む。64M adapter stream全体を保護する値ではない。Prefix除外後の初期eligible pool約51.74M source tokensは64M streamの残りと重複し得る。

SAE corpus構築時の `--training-budget` は現在registryで一意に凍結されていない。`minimum` では100M training + 10M statistics + 200文書×512 = 110,102,400 source tokensとなり、約58.37Mを64M stream外の新規FineWeb-Eduから補充する。`preferred` では200M trainingを使うため合計210,102,400 source tokens、stream外補充は約158.37Mになる。いずれもSAE corpusは64M streamの部分集合ではなく、最大重複部分が初期eligible約51.74Mである。実行時に選んだbudget operatorと最終corpus hashをregistryへ保存する。

Fail-closedに強制される除外roleは、tune / pre-PR / final / localization selection / localization validationの5種だけである。従ってWP-3はlocalization validationとして保護されるが、monitorを使うWP-4と未抽出のWP-5 diagnostic 200は現registryだけでは非重複が保証されない。各WPの実行前にID/hashをexclusion inventoryへ登録し、SAE train manifestとの非重複をrunnerで検証する。重複時は結果を計算せず、分離済みreplacement cohortを事前登録する。この契約を追加するまでは「monitor/全diagnostic itemをSAE trainから除外済み」と主張しない。

## 4. Work package依存関係

~~~text
WP-1: SAE学習
  layer 5 init-seed42 ─┐
  layer 5 init-seed43 ─┼─> WP-2 acceptance
  layer20 init-seed42 ─┘       │
                               ├─ pass -> WP-3 causal-chain diagnostic
                               ├─ pass -> WP-4 checkpoint diagnostic
                               └─ pass -> WP-5 feature kill test
                                              │
                         G1 + G2 + G3 pass ----┴─> amendment後だけsuppression草案
                         G1 + G3 only ------------> 現registryのstudy草案まで。学習未認可
                         G1 fail / G2 fail --------> suppressionを認可しない
                         fixed C案 ----------------> 別selection/held-out kill testが必要
~~~

| WP | 目的 | 依存 | 出力 | 現状 |
|---|---|---|---|---|
| WP-1 | clean residualのSAE学習 | なし | weights、data hash、train stats | 完了 |
| WP-2 | SAEが診断に耐えるか検収 | WP-1 | gate report、\(p_f,s\) | 実行段階 |
| WP-3 | early patch→mid OODの因果連鎖 | WP-2 pass | condition curve | 未実装 |
| WP-4 | checkpointに沿うOOD軌跡 | WP-2 pass | trajectory | 未実装 |
| WP-5 | feature成分の因果十分性 | WP-2 pass、layer5 2 init seeds | kill/pass判定 | 未実装 |
| WP-6 | feature後継study | suppression: 事前amendment + WP-5 G1/G2/G3、fixed C: 別kill test | preregistration draft | 実行禁止 |

WP-3/4は診断である。WP-5の現registryはG1+G3で後継studyの草案だけを許し、学習を認可しない。Suppression routingを追加するなら結果計算前のamendmentが必要であり、G1を外してはならない。G1だけで固定feature学習の生死も決めない。

## 5. WP-1: Data -> activation -> optimization -> statistics

**状態: 実装済み・`minimum` operatorによる100M activation-token本学習完了。ただし採用operatorのregistry記録は要補完**

### 5.1 対象層、初期化seed、budget

| SAE | 理由 | \(\lambda_{L1}\) | budget |
|---|---|---:|---:|
| layer 5, init-seed 42 | 因果窓 \([0,6)\) の出口 | 0.01 | 100M activation tokens |
| layer 5, init-seed 43 | feature結論の初期化再現性 | 0.01 | 100M |
| layer 20, init-seed 42 | 相対深さ20/34、mid-layer OOD probe | 0.1 | 100M |

Layer 5は現行state target空間の出口である。Layer 20は先行するSAE OOD研究の相対深さ約0.59に対応し、random-window \([20,26)\) の始点でもある。ただしlayer 20を因果窓とはみなさない。

### 5.2 Pipeline

1. **Data**: train splitのclean FineWeb-Eduだけを読み、除外registryと近重複検査を適用する。
2. **Activation**: Frozen Baseへ最大512 tokensで入力し、同じforwardで対象blockのcomplete residualをhookする。
3. **Streaming**: paddingを除いたtoken activationをshuffle bufferへ入れ、SAE mini-batchをsampleする。
4. **Optimization**: Baseへgradientを流さず、SAE parameterだけをfp32で更新する。
5. **Constraint**: 各step後にdecoder列をunit normへ戻す。
6. **Calibration**: 分離clean activationでL0/FVUを測り、登録規則に従い \(\lambda_{L1}\) を決める。
7. **Statistics**: 完成本体と分離した10M clean activationで \(p_f,s\) を算出し凍結する。
8. **Acceptance**: WP-2の4 gateとartifact completenessを評価する。

Typo textをSAE学習へ入れない。SAEがclean manifoldを学ぶことで、typo側の \(L_0\)・energyをOOD逸脱として解釈できる。

### 5.3 Data budgetとartifact

- 完了済みbundle: `minimum` operator、SAEごとに100M clean activation tokens
- Config上の別operator: `preferred`、SAEごとに200M。現bundleには使っていないが、configから削除もされていない
- Statistics: 本学習と分離したclean 10M activation tokens
- Activation batch 2,048、shuffle buffer 1M、dead feature resamplingなし
- Layer 5/11/20/26の再利用用subsampleは別artifactとして扱い、本学習token countと混同しない
- Data registry: record/source/group ID、正規化content hash、character 5-gram近重複情報
- 必須保存: SAE weights、optimizer/config、\(\lambda_{L1}\)、init seed、model revision、data hash
- Statistics保存: feature別 \(p_f\)、clean error median \(s\)、L0/FVU/dead分布

\(p_f,s\) を学習mini-batchの都度変えず、完成したSAEに対して分離streamで一度算出して凍結する。完了済みbundleをWP-2以降へ渡す前に、採用した `minimum` operatorと最終corpus hashをregistryへ記録し、configが許す `preferred` と区別する。

### 5.4 \(\lambda_{L1}\) 選択規則

Calibration 10M tokensで各grid候補を学習し、clean上だけで次を適用する。

~~~text
candidates = {lambda: median_L0 in [30,150]}
if candidates is empty:
    calibration failure
else:
    choose argmin(FVU), deterministic tie-break
~~~

TypoのL0差、WP-5 restoration、task accuracyを見て \(\lambda_{L1}\) を選び直さない。

最初の登録grid \(\{10^{-4},3\times10^{-4},10^{-3}\}\)、1M-token calibrationでは、layer 5のmedian L0が5,465 / 5,330 / 4,874、layer 20が3,890 / 3,881 / 3,856で、全候補が許容範囲を外れた。本学習やWP-2/5を始める前に、許可された一度だけのamendmentとしてgridを \(\{0.01,0.1,1.0\}\)、calibrationを10Mへ変更した。これはtypo behaviorを見た後の調整ではなく、clean sparsity gateを満たすための較正である。残りの較正amendment枠は0である。

## 6. WP-2: SAE acceptance gate

**状態: 実装済み・数値事前登録済み・検収実行段階**

各SAEは次を全て満たした場合だけWP-3/4/5で使用できる。

### 6.1 FVU

\[
\mathrm{FVU}
=
\frac{\sum_t\|x_t-\hat x_t\|_2^2}
     {\sum_t\|x_t-\bar x\|_2^2}
\le0.35
\]

Clean residual分散の65%以上を説明することを要求する。

### 6.2 Sparsity

\[
30\le\operatorname{median}_tL_0(x_t)\le150
\]

### 6.3 Dead feature rate

\[
\frac1{d_{\mathrm{SAE}}}
\sum_f\mathbf1[p_f<10^{-5}]
\le0.20
\]

### 6.4 Splice KL

Probe layerのresidual \(x\) を再構成 \(\hat x=Dz\) へ置換し、元のmodel出力とのKLを測る。

\[
\operatorname{median}
KL(p_{\mathrm{original}}\parallel p_{\mathrm{SAE\ splice}})
\le0.15\ \mathrm{nats/token}
\]

集計単位は**文書medianのmedianではない**。Statistics用10M clean activation tokensの直後から、未使用のclean文書200件を固定順で取り、各文書の全causal next-token位置（末尾を除く）でKLを計算する。200件から得たtoken-level KLを一つの列へ連結し、そのpooled token列のmedianを各SAEについて1個だけ算出する。したがって長い文書は有効token数に比例して寄与する。3本のSAEは同一200文書を使い、splice cohortをSAEごとに変えない。

FVUが良くても、modelが利用する小さな方向を失って挙動を壊すSAEはここで除外する。

### 6.5 Completenessと不合格時

- weights/config/model revision/data hashが揃う
- \(p_f\) と \(s\) が分離clean streamで計算済み
- gateのper-SAE値と判定がmachine-readableに保存される

WP-2のacceptance単位は、layer 5 seed 42、layer 5 seed 43、layer 20 seed 42を含む**3本一組のtraining bundle**である。3本すべてが合格したときだけbundleをacceptする。

最初のbundleが不合格の場合、registry上の研究方針は**project全体で最大1回の新しい3本一組full-retrain bundle**だけを許す。SAEごとに1回ずつではない。初回ledgerだけでは新しい親directoryで履歴をリセットできたため、runnerへproject-global claimを追加した。別レビュー済みauthorizationが初回config・事前登録・training・validation・acceptance・ledgerのhash chainと単一の改訂config / 事前登録hashを結び、救済trainingはruntime/model初期化前に`O_EXCL` claimを取得する。完全一致するclaimだけをresumeでき、validationはclaim済みtraining-run SHAだけを受理するため、親directoryを変えても追加bundleを検収できない。

WP-2不合格時は自動再学習せず停止する。救済を使う場合は、結果に応じて複数値を探索せず、開始前に単一変更を固定したregistry amendment、新しいconfig / preregistration hash、失敗bundleを参照するauthorizationを別PRでレビューする。project-global lineage enforcementは実装済みだが、科学的な救済authorizationや改訂値はまだ追加していないため、現時点では救済runを実行しない。変更できる種類は \(\lambda_{L1}\) **または**SAE training activation-token budgetのどちらか一方だけで、Architecture、clean-only data contract、WP-2 gate、splice cohortは変更しない。再学習bundleも不合格なら、3本をWP-3/4/5へ使わず、2回目の救済を与えない。

このWP-2 retrain枠は、WP-1前に行ったL1 calibration amendmentとは別である。Calibration amendmentは「全候補のL0が範囲外だったため、gridとcalibration token数を一度だけ変更した」履歴で、残り枠は0である。WP-2 retrainは完成した3-SAE bundleがall-three acceptance gateを外した場合の救済であり、calibration grid再探索、複数runからの選択、gate変更を許可しない。つまり、calibration amendmentはglobalに1回消費済み、WP-2救済も研究方針上はglobalに最大1 full-retrain bundleである。Global enforcementは実装済みだがauthorizationは未作成なので、別PRで科学的変更とhash chainが承認されるまで実行禁止である。

## 7. WP-3 / WP-4: 診断

### 7.1 WP-3 Base統合診断

**状態: 予測・データ用途・3条件は登録済み、専用runner未実装。以下の集計・CI・支持判定はpre-execution amendmentとして未凍結**

検証する因果連鎖は次である。

~~~text
早期層の編集語表現が破損
        -> 中層で余剰・希少featureを動員
        -> 将来token分布が変化
~~~

Localization validation用FineWeb-Edu 200件を再利用し、1 typo/文書でBaseを3条件実行する。

1. clean
2. typo
3. typo + \([0,6)\) full-state patch

編集語末を \(t_i\) とし、17位置 \(t_i,\ldots,t_i+16\) についてlayer 5/20のL0とenergyを測る。指標
\(q\in\{L_0,F\}\)、条件 \(c\in\{\mathrm{clean},\mathrm{typo},\mathrm{patched}\}\) に対し、item内集計を次の算術平均へ固定する。

\[
\bar q^{\,\ell}_{i,c}
=\frac1{17}\sum_{r=0}^{16}q^\ell_{i,c,t_i+r}
\]

17位置と3条件の全値が有限なitemだけを、layer・指標ごとのcommon-valid subsetへ入れる。条件ごとに別のitem集合を使わない。除外件数と理由を報告する。

P1のitem-level estimandと集計量は次である。

\[
\delta^{P1}_i
=\bar L^{\,20}_{0,i,\mathrm{typo}}
-\bar L^{\,20}_{0,i,\mathrm{clean}},
\qquad
\widehat\Delta^{P1}
=\frac1n\sum_i\delta^{P1}_i
\]

P2は「patch後がcleanへ近付く量」を、符号付き差ではなく距離差として定義する。

\[
\delta^{P2}_{i,q}
=\left|\bar q^{\,20}_{i,\mathrm{typo}}-\bar q^{\,20}_{i,\mathrm{clean}}\right|
-\left|\bar q^{\,20}_{i,\mathrm{patched}}-\bar q^{\,20}_{i,\mathrm{clean}}\right|,
\qquad
\widehat\Delta^{P2}_{q}
=\frac1n\sum_i\delta^{P2}_{i,q}
\]

不確実性はcommon-valid itemをpair単位で10,000回再抽出するpaired bootstrap（seed 42）のpercentile 95% CIで求める。

- P1支持: \(\widehat\Delta^{P1}\) の95% CI下限が0より大きい。
- P1反証: 95% CI上限が0以下。それ以外はinconclusive。
- P2支持: \(q=L_0,F\) の両方で \(\widehat\Delta^{P2}_{q}\) の95% CI下限が0より大きい。
- P2反証: どちらか一方でも95% CI上限が0以下。それ以外はmixed / inconclusive。

P2のprimary layerは20である。Layer 5の編集語末そのものはpatchによって自明にclean一致するため、layer 5は後続位置を含む記述値としてのみ報告し、P2判定に使わない。

この集計・bootstrap seed・支持規則はまだmachine-readable registryへ入っていない。結果計算前にamendmentとしてcommitしない限り、WP-3は実行せず「仕様未完成」とする。P2成立はearly patchがmid-layerの余剰feature動員を因果的に抑えることを支持する。不成立でも既存window patch restorationは独立に残り、現行学習や評価を変更しない。

### 7.2 WP-4 checkpoint遡及診断

**状態: 予測・データ用途を登録済み、専用runner未実装。以下のestimand・5% clean-drift margin・bootstrap規則はpre-execution amendmentとして未凍結**

使用データはnatural clean/typo 100 pairと、monitor poolから決定的に選ぶsynthetic 100 pairである。2 stratumを混ぜて200件poolせず、stratum内平均を取ってからnatural / syntheticを等重みmacroする。

Monitor roleは現行のfail-closed除外roleに含まれない。従ってWP-4実行前に、この200 pairのID/hashをexclusion inventoryへ登録し、SAE train manifestとの非重複をrunnerで検証する。重複が一件でもあれば結果を計算せず、同じ決定規則で分離済みreplacement cohortを事前登録する。

Arm間のcheckpointはoptimizer stepでなく累積student tokensが完全一致する点だけを対応させる。補間しない。最終判定点は両armに存在する最大の共通student-token checkpointとし、それ以前の共通点はtrajectoryの記述にだけ使う。

各checkpoint \(k\)、arm \(a\)、layer \(\ell\in\{5,20\}\)、指標 \(q\in\{L_0,F\}\) について、WP-3と同じ17位置算術平均を使う。

\[
\bar q^{\,\ell}_{i,a,c,k}
=\frac1{17}\sum_{r=0}^{16}q^\ell_{i,a,c,k,t_i+r}
\]

\[
\Delta q^{\,\ell}_{i,a,k}
=\bar q^{\,\ell}_{i,a,\mathrm{typo},k}
-\bar q^{\,\ell}_{i,a,\mathrm{clean},k}
\]

P3のfinal-checkpoint item-level contrastは、output matchingに残るclean--typo逸脱の絶対値からproposalの逸脱の絶対値を引く。

\[
\psi^{\,q}_{i}
=\left|\Delta q^{\,20}_{i,\mathrm{output},k_*}\right|
-\left|\Delta q^{\,20}_{i,\mathrm{proposal},k_*}\right|
\]

Stratum \(s\) 内で \(\bar\psi^q_s=n_s^{-1}\sum_{i\in s}\psi^q_i\) を計算し、
\(\widehat\Psi^q=(\bar\psi^q_{\mathrm{natural}}+\bar\psi^q_{\mathrm{synthetic}})/2\) をprimary aggregateとする。

P4のclean側driftは、Base cleanに対するarm cleanの絶対差をBase水準で割った相対量とする。

\[
D^{\,\ell,q}_{a}
=
\frac{
\frac12\sum_s\left[n_s^{-1}\sum_{i\in s}
\left|\bar q^{\,\ell}_{i,a,\mathrm{clean},k_*}
-\bar q^{\,\ell}_{i,\mathrm{Base},\mathrm{clean}}\right|\right]
}{
\max\!\left(
10^{-12},
\frac12\sum_s\left[n_s^{-1}\sum_{i\in s}
\left|\bar q^{\,\ell}_{i,\mathrm{Base},\mathrm{clean}}\right|\right]
\right)
}
\]

全arm、全共通checkpoint、clean/typo、layer 5/20、L0/energyで有限なitemだけをstratumごとのglobal common-valid subsetへ入れる。除外件数と理由を報告する。Bootstrapは各stratum内で同じpaired item indexを10,000回再抽出し（seed 42）、各反復で等重みmacroを再計算する。

- P3支持: layer 20の \(q=L_0,F\) の両方で \(\widehat\Psi^q\) の95% CI下限が0より大きい。
- P3反証: どちらか一方でも95% CI上限が0以下。それ以外はmixed / inconclusive。
- P4支持: output / proposalの両arm、layer 5/20、\(q=L_0,F\) の全組合せで \(D^{\ell,q}_a\) の95% CI上限が0.05以下。
- P4反証: いずれかの組合せで95% CI下限が0.05より大きい。それ以外はmixed / inconclusive。

5%はbehavior gateではなく、「small clean-side SAE drift」というP4の意味を事前に一意化するためのdiagnostic marginである。このmargin、global common-valid規則、bootstrap seedはまだregistryへ入っていないため、結果計算前のamendmentがない限りrunnerを起動しない。

WP-4はcheckpoint選択、学習停止、behavior gateに使わない。中間checkpointのP3/P4値はtrajectoryとして全て表示するが、最良checkpointを選ばない。診断列を追加した事実だけをamendment logへ記録する。

## 8. WP-5: feature因果kill test

**状態: pre-execution amendmentが未完成のdraft。operator、G1--G3のcore数値、bootstrap 10,000回 / seed 42はconfigで固定済み。専用runnerは未実装・結果は未計算**

現時点でmachine-readable source群に未凍結なのは次の8点である。本節後半にdraft値があっても、amendment commitまでは有効な事前登録ではない。

1. typo生成seed
2. typo操作3種（keyboard-neighbor substitution / deletion / duplication）の混合比、適用順、適格語規則
3. diagnostic 200のID/hashとSAE train manifest非重複をfail-closedにするrole / inventory規則
4. near-zero denominatorと全operator共通eligible規則
5. \(R_{\mathrm{full}}\) が正に効くことを確認する前提gate
6. CIの計算法と境界値での判定
7. G1/G2から後継study草案へ進める範囲（現registryのG1+G3要件を維持したrouting）
8. 使用するSAE artifact hashとWP-2合格記録

Bootstrapの反復数10,000とseed 42は [gemma4b-sae-v1.yaml](../configs/sae/gemma4b-sae-v1.yaml) の必須fieldとして既に固定済みであり、新しい選択肢ではない。`registry-v1.yaml` の説明欄に重複記載がないことを「未凍結」と解釈しない。

これらを**結果を一切見ずに**machine-readable registry amendmentへcommitし、hashを固定するまでWP-5 runnerを起動しない。以下はそのamendment草案であり、登録済み仕様や実行済み結果として引用してはならない。

### 8.1 問い、データ、3種類のseed

問いは次である。

> Layer 5 residualが運ぶ修復情報の十分な部分を、SAE feature成分 \(Dz\) だけで移植・除去できるか。

全既存tierとID非重複の新規FineWeb-Edu 200件を固定し、各例へ1 typoを加える。操作集合は既存研究と同じkeyboard-neighbor substitution / deletion / duplicationに限定するが、3種の混合比、決定的な適用順、適格語規則は現registryにない。そのため、実行前amendmentでtypo seedと同時に固定する。Seedは役割を分けて記録する。

| seed | 役割 | 再現性の対象 |
|---|---|---|
| SAE init seed 42 / 43 | feature基底の初期化 | G3: 基底が変わってもgate方向が同じか |
| typo seed | 同じ文書に作る実現typo | 両SAEで同一pairを共有し、基底差だけを比較。draft値42、現registryは空 |
| bootstrap seed | 10,000回のresampling | configで42 / 10,000回に固定済み。feature基底の再現性とは別 |

Layer 5の編集語末で \(h=Dz+\epsilon\) と分解する。

### 8.2 Operator

#### Full-state単層patch

\[
h'=h_{\mathrm{clean}}
\]

Restorationを \(R_{\mathrm{full}}\) とする。既知の6-layer window patchは文脈値として別に併記する。

#### Feature swap

\[
h'=Dz_{\mathrm{clean}}+\epsilon_{\mathrm{typo}}
\]

Feature成分だけclean化したrestorationを \(R_z\) とする。

ここで入れ替えるのは選択済みの少数featureではなく、SAEが再構成に使う**全feature成分 \(Dz\)** である。従って \(R_z\) は「SAE feature成分全体の十分性」を測るが、「疎な固定causal feature集合が存在する」ことは測らない。

#### Reconstruction-error swap

\[
h'=Dz_{\mathrm{typo}}+\epsilon_{\mathrm{clean}}
\]

Dark matter側だけclean化したrestorationを \(R_\epsilon\) とする。

#### Spurious-feature suppression

\[
S_i=\{f:z^{\mathrm{typo}}_{i,f}>0\land z^{\mathrm{clean}}_{i,f}=0\}
\]

\[
h'
=h_{\mathrm{typo}}
-\sum_{f\in S_i}z^{\mathrm{typo}}_{i,f}d_f
\]

Cleanで不発火、typoで発火したfeatureだけを除去したrestorationを \(R_{\mathrm{sup}}\) とする。

#### Sparsity curve

\(\lvert z_{\mathrm{clean}}-z_{\mathrm{typo}}\rvert\) 上位 \(k\in\{8,32,128\}\) だけをswapし、必要feature数とrestorationの関係を記述する。これは主要gateを変更しない。

### 8.3 Eligibilityと実行前に追加凍結する前提gate

各operatorのreadoutは既存multi-token KL restorationである。

\[
R_{i,C}
=1-
\frac{\sum_{t=2}^{16}KL(p_{\mathrm{clean},i,t}\parallel p_{C,i,t})}
     {\sum_{t=2}^{16}KL(p_{\mathrm{clean},i,t}\parallel p_{\mathrm{typo},i,t})}
\]

Core G1--G3だけでは、near-zero denominatorやfull-state単層patchが効かない場合の比率解釈が定義不足である。WP-5は未実行なので、次を**結果を見る前のregistry amendment**として固定してからrunnerを起動する。現時点では登録済み数値と混同しない。

次の順序を守る。

1. Denominator \(D_i^{\mathrm{typo}}\le10^{-9}\) のpairを全operator共通で除外する。
2. Full-state、feature、error、suppressionの全条件が有限なcommon-eligible subsetを作る。
3. G3比較では両SAE init seedのcommon-eligible subsetの共通部分を使い、seedごとに有利なpair集合へ変えない。
4. \(n_{\mathrm{eligible}}\)、除外理由、分母分布をSAE seed別とseed共通subsetの両方で報告する。
5. \(R_{\mathrm{full}}\) のitem-level medianを主集計とし、common-eligible itemを10,000回再抽出するpercentile paired bootstrap 95% CI（configで固定済みのseed 42）の下限が0より大きいことを**feature gateの前提**とする。
6. 前提不合格なら、単層layer 5に十分な修復余地がないためG1/G2を比率で判定せず、WP-5をinconclusiveとしてfeature学習をkillする。

この前提が必要なのは、\(R_{\mathrm{full}}\) が近零または負のとき、\(R_z/R_{\mathrm{full}}\) 型のgateが不安定または意味を失うためである。

統計はpair単位paired bootstrap 10,000回、seed 42を使う。両値はconfigで固定済みである。各反復では全operatorと両SAE init seedへ同じresample indexを使い、operatorごとのmedianを再計算する。Operatorごとに異なるsubsetを使って見かけの比率を変えない。

### 8.4 登録済みG1--G3とG3の論理

\[
G1:\quad
\operatorname{median}(R_z)
\ge0.50\,\operatorname{median}(R_{\mathrm{full}})
\]

\[
G2:\quad
\operatorname{median}(R_{\mathrm{sup}})
\ge0.25\,\operatorname{median}(R_{\mathrm{full}})
\]

G3は別の第三指標ではなく、**G1/G2のseed再現要件**である。

- G1合格を主張するには、SAE init-seed 42と43の両方でG1方向を満たす。
- G2合格を主張するには、両方でG2方向を満たす。
- 両SAEは同じitem、同じrealized typo、同じbootstrap resample indexを使う。
- Feature indexの一致は要求しない。非正準な基底を跨いで、feature**成分全体の因果量**が再現するかを見る。

#### G1を過大解釈しないための制約

G1の \(R_z\) は全 \(d_{\mathrm{SAE}}=40{,}960\) featureを通した \(Dz\) のswapである。Decoder方向数は
\(d_{\mathrm{model}}=2{,}560\) を大きく上回り、そのspanのrankはresidual空間全体へ近付く可能性がある。この場合、G1合格はSAE再構成がfull residual swapに近いことの反映でもあり、「少数の意味featureを局在した」証拠にはならない。All-feature subspaceはidentity / raw residualに近いscope controlとして扱う。

現registryの凍結済みdecisionは、**G1とG3だけがfeature-targeted successor-studyの草案を許し、後継学習は認可しない**というものである。以下のG2 routingは、共同研究者指示を実装可能な形へ具体化するためのamendment草案であり、結果計算前にregistryへ反映されない限り効力を持たない。Amendmentを行う場合もG1要件を外さない。

従って、後継studyへの許可候補を次のように限定する。

- G1 + G3: **全feature成分 \(Dz\) が因果情報を十分含む**ことだけを支持し、現registryのfeature-targeted study草案条件を満たす。固定feature集合 \(C\) の学習、feature-subspace学習、causal feature localization、実際の後継学習は許可しない。
- G1 + G2 + G3: 事前amendmentでG2 routingを追加した場合に限り、clean/typo pairごとに定義する \(S_i\) の除去が介入として効くことを支持する。同じpair-specific \(S_i\) を使うspurious-suppression **study草案**だけを追加でき、学習開始には別途完全な事前登録が必要である。
- 固定集合 \(C\): 別の選択データで \(C\) を選び、独立held-out dataで \(C\)-swapがmatched-random \(C\) より効くことを確認する、別の事前登録causal selection / kill testが必要である。そのtestなしに固定部分空間lossへ進まない。

| 観測 | 解釈 | 次の処理 |
|---|---|---|
| G1 + G2 + G3 pass | all-feature成分が十分で、pair-specificなtypo-only feature除去も有効 | 事前amendmentがある場合だけ同じ \(S_i\) を使うsuppression studyを草案化。学習は別登録まで禁止 |
| G1 + G3 pass、G2 fail | all-feature成分は十分だが片側除去は不足 | 現registryのfeature-targeted study草案まで。固定 \(C\) 用の別preregistered selection / kill testを設計 |
| G1 fail（G2の向きにかかわらず） | 現registryのsuccessor-study条件を満たさない | suppression / fixed-feature studyを認可しない |
| \(R_z\approx0, R_\epsilon\approx R_{\mathrm{full}}\) | 因果情報はdark matter側 | feature学習をkill |
| G2がSAE init seed間で方向不一致 | pair-specific suppressionの結論が不安定 | suppression studyをkill |
| G1だけがSAE init seed間で方向不一致 | all-feature十分性の結論が不安定 | 固定feature / subspace案をkill。G2の判断とは分離 |

不合格は実験失敗ではなく、無効な教師targetへ高価な学習を行う前に仮説を棄却できたというプロセス上の成功である。

# Part II: 後継proposal roadmap

## 9. 改善候補の設計原則

1. 現行studyへlossを継ぎ足さず、別studyとして一つの仮説だけを変える。
2. Output matchingを必ず中心baselineにする。
3. Pair-specific suppressionは結果計算前のregistry amendmentとWP-5 G1/G2/G3の全通過前、fixed-feature案は別のselection / held-out kill test前に学習しない。
4. SAE weights、encoder、decoder、\(p_f,s\) は後継Student学習中にfreezeする。
5. Diagnostic改善ではなく、凍結済みclean非劣性・typo優越性で成功判定する。
6. Random、all-feature/all-layer、raw residual controlで「featureを足しただけ」と「因果target」を分離する。
7. 新しい係数や閾値はbehavior結果を見る前に一度だけ登録する。

## 10. 候補matrix

### 10.1 手法、計算、risk

| 候補 | 仮説 | 現行との差 | 追加loss / HP | 事前kill test | Compute / tokens | Clean risk | Model-agnostic |
|---|---|---|---|---|---|---|---|
| A. 64M raw residual | 10Mは用量不足 | 手法不変、10M→64M | 追加0 | 既存window validation | 64M/arm、Teacher+Student | 現行gateで監視 | 手続きは可搬、窓は再選択 |
| B1. Fixed-feature subspace | 因果featureだけなら到達可能 | residual全次元を固定部分空間へ置換 | 補助loss 1、係数1、selection / rank rule | WP-2 + **別の固定 \(C\) causal selection / held-out kill test** | SAE済み + selection test + matched training | all-feature spanがidentity化、selection overfit | layer/SAE/固定 \(C\) をモデルごとに再導出 |
| B2. Spurious suppression | 余剰除去はclean復元より容易 | typo-only featureを片側抑制 | 補助loss 1、係数1、\(p_f\) clamp | 事前amendment + WP-5 G1/G2/G3 | B1相当 | 希少だが有用なfeature抑制 | 定義は可搬、SAE固有 |
| C. Causal Patch Distillation | patch後のbehaviorはstateより学習可能 | state lossをpatched-output KDへ置換 | output loss 1、rhoなし | window patchの教師品質 | noisy row 3 forwards | 悪いpatch教師 | 高い。windowのみ再選択 |
| D. SAE energy selection | 高OOD例はdata-efficient | loss不変、noisy dataだけ選別 | quantile 1 | 因果testなし、相関仮説 | candidate scoring + 同token学習 | hard-example過適合 | SAEをモデルごとに必要 |
| E. Patch-effect selection | patch可能例へ容量集中 | loss不変、restorationで選別 | threshold/quantile 1 | score自体が介入 | 全候補patchで最も高い | easy-to-fix bias | window再選択が必要 |

### 10.2 Controls、成功、縮小claim、priority

| 候補 | Baseline / controls | Behavior成功条件 | Diagnostic | Kill条件 | 新規性 | 不成立時の縮小claim | 状態 / trigger / priority |
|---|---|---|---|---|---|---|---|
| A | Base、output-only、後にrandom/all-layer | proposal > output、clean gate、因果主張にはcontrols | patch audit | 64Mでも固有差なし | causal targetの増分価値 | patch可能性≠raw-state学習価値 | 実行段階 / 現在 / 1 |
| B1 | output、raw residual、matched-random \(C\)、all-feature scope control | output/raw/randomよりPareto優位 | held-out \(R_C\)、feature distance | 固定 \(C\) kill test fail、seed不再現、clean harm | 因果feature選択 | 固定feature集合は学習targetにならない | 未認可 / 別selection test合格 / 2 |
| B2 | output、raw residual、matched random、all feature | output/randomより優位、clean gate | spurious mass、energy | G1/G2/G3のいずれかfail、natural harm | 因果的余剰feature除去 | suppression単独は不十分 | 未認可 / amendment + G1/G2/G3 / B内1 |
| C | output KD、causal/random window teacher、必要ならall-layer | causal teacher > output/random、clean gate | patch gain reduction | teacher悪化、output非優位 | oracle介入のweightsへの償却 | patch teacherは学習targetにならない | 未登録 / A+B fail / 3 |
| D | severity/length matched random、high/low energy | same tokensでrandomより優位 | energy曲線 | matching後差なし、OOD harm | OOD-aware data efficiency | energyは診断に限定 | 未登録 / 独立 / 4 |
| E | matched random、high/low restoration | same tokensでrandomより優位 | restoration分布 | biasだけ、natural harm | intervention-aware sampling | patch-positive subsetの境界結果 | 未登録 / 独立 / 5 |

全候補は既存評価protocolから独立に設計し、評価itemやsealed結果をselectionへ使わない。

## 11. 候補A: 現行residual-state手法の64M用量検証

**状態: config実装済み・matched-budget比較の実行段階。新手法ではない**

10Mではproposalとoutput-onlyの差が検出できなかった。Loss、window、rho、data distributionを変更せず、student-token budgetだけを64Mへ伸ばす。

- output-only: 64M student tokens
- proposal: output + block 0--5 residual cosine、64M
- 同じ64M unique source stream、realized typo、1:1
- 同じLoRA、optimizer、checkpoint grid
- 正式値 \(\rho=0.05\) を維持
- window、loss、評価gateを変更しない

64Mでもproposalがoutput-onlyと必要controlsを上回らず、auditにも固有差がなければ、

> Activation Patchingでpatch可能なraw residual座標は、このcosine整合形式では追加の学習価値を示さない。

と結論する。結果を見てrhoやwindowを再調整しない。

## 12. 候補B / WP-6: SAE feature後継study

**状態: 現registryでは後継学習未認可。Suppressionは事前amendment後のWP-5 G1/G2/G3合格時だけ草案化でき、fixed-feature subspaceはさらに別のcausal selection / held-out kill test合格時だけ許可される**

### 12.1 共通training contract

- Accepted layer 5 SAEのweights、encoder、decoder、\(p_f,s\) をfreezeする。
- Frozen Baseのclean/typo activationとStudent typo activationを同じfrozen SAEへ通す。
- GradientはStudent LoRAへ、frozen SAE encoderを通じてだけ流す。
- SAE parameter、feature方向、発火統計は更新しない。
- Feature lossは有効edited positionsとfeature数で正規化し、accumulation batch全体で集計する。
- Raw residual補助lossを**置換**し、3項lossにしない。
- 係数は初期gradient norm方式なら1回だけ較正し、値とsampleを登録する。

記法を次で固定する。\(E\) はfrozen SAE encoder、\(h_B^c,h_B^p\) はadapterを無効化した同一Baseのclean/typo residual、\(h_{S_\theta}^p\) はStudent typo residualである。

\[
z_{B,i}^c=E(h_{B,i}^c),\qquad
z_{B,i}^p=E(h_{B,i}^p),\qquad
z_{S,i}^p=E(h_{S_\theta,i}^p)
\]

Pair-specific target集合は、WP-5で介入検証したoperatorと一致させるため、**frozen Base pairから一度だけ定義してstop-gradientする**。

\[
S_i^B=\{f:z_{B,i,f}^p>0\land z_{B,i,f}^c=0\}
\]

同じrealized typoに対する \(S_i^B\) は、offline cacheまたは学習step内のadapter-disabled Base forwardのどちらで得てもよいが、結果は同一でなければならず、provenanceを保存する。Studentの更新に伴って \(S_i\) を再計算するdynamic-target版は、WP-5で検証したoperatorと異なる未検証手法なので、この候補には含めない。

### 12.2 G1 / G2 / G3 pass後の草案: spurious-feature suppression

\[
\mathcal L_{\mathrm{spur}}
=
\frac{\mathbf 1[\sum_i|S_i^B|>0]}{\max(1,\sum_i|S_i^B|)}
\sum_i\sum_{f\in S_i^B}
w_fz^p_{S,i,f}
\]

\[
w_f=-\log\tilde p_f,
\qquad
\tilde p_f=\operatorname{clip}(p_f,\epsilon_p,1-\epsilon_p)
\]

\[
\mathcal L
=\mathcal L_{\mathrm{out}}
+\lambda_{\mathrm{spur}}\mathcal L_{\mathrm{spur}}
\]

\(-\log\tilde p_f\ge0\) なので、cleanで頻出するfeatureを「抑制lossの最小化によって増やす」という符号反転を起こさない。主案ではlog-oddsや負weightを使わない。

Accumulation batch全体で \(\sum_i|S_i^B|=0\) の場合、\(\mathcal L_{\mathrm{spur}}=0\) とし、この項からStudentにもSAEにもgradientを発生させない。SAEは常にfreezeする。\(\epsilon_p\) と任意のweight上限は未登録であり、実行前に固定する。Clean語の完全復元を要求せず、frozen Baseの対応cleanで発火せずBase typoで発火したfeatureについて、Student typo activationだけを抑える。

### 12.3 固定feature部分空間整合: G1だけでは未認可

G1がswapするのは全feature成分 \(Dz\) であり、固定集合 \(C\) を選ばない。従ってG1 pass / G2 failからこの案へ直接進んではならない。先に、独立studyとして次の手続きを事前登録する必要がある。

1. selection用clean/typo pairでcandidate featureをscreenし、feature数とscoreを凍結する。
2. Candidate集合 \(C\) と、同数・同発火率または希少度のmatched-random集合を結果計算前に固定する。
3. ID非重複held-out pairで \(C\)-swap、matched-random swap、all-feature swap、raw full-state patchを比較する。
4. \(C\)-swapがmatched-randomより大きなKL restorationを示すためのCI・marginを実行前に登録する。
5. SAE init seedを変えて判定方向が再現した場合だけ、固定 \(C\) をcausal training target候補と呼ぶ。

この別testのデータ、score、feature数、marginは未定義であり、現時点では固定 \(C\) 学習を認可しない。

上記testを通過した場合に限り、因果性を確認したdecoder方向を列に持つ \(D_C\) から非直交性を除いた射影を作る。単純に \(D_CD_C^\top\) を射影とみなしてはならない。

候補定義は、

\[
D_C=QR,\qquad P=QQ^\top
\]

のようにrank-revealing QRまたはSVDで直交基底 \(Q\) を作ることである。

\[
\mathcal L_{\mathrm{feature}}
=1-\cos(Ph_{\mathrm{clean}},Ph_{\mathrm{typo}})
\]

\[
\mathcal L
=\mathcal L_{\mathrm{out}}
+\lambda_{\mathrm{feature}}\mathcal L_{\mathrm{feature}}
\]

ただし、feature集合 \(C\) の定義、rank tolerance、feature数、正規化は**現時点で未定義の草案**である。全feature decoder spanはrankが \(d_{\mathrm{model}}\) へ達すると \(P\approx I\) となりraw residual lossと区別できないため、all-feature版はlocalized methodではなくscope / identity controlに固定する。

### 12.4 必須control

- 学習なしBase
- output matching only
- 同じlayer/positionのraw residual loss
- 同数・同発火率または希少度のrandom features
- all features（identity / raw residualへ近付くscope controlとして解釈）
- random layerまたはrandom-window

Behavior評価をprimary、SAE量をdiagnosticに維持する。

## 13. 候補C: Causal Patch Distillation

**状態: 未実装・未登録。WP-5不合格かつraw residual不成立時の第一候補**

Raw residualそのものはtypoから到達不能でも、Activation Patchingが実際に生んだ**修復済みoutput分布**は学習可能なbehavior teacherになり得る。

Noisy rowでは、

1. Frozen Base(clean)からwindow stateを取得する。
2. Frozen Base(typo)へそのstateをpatchする。
3. Patched typo出力 \(q_{\mathrm{patch}}\) を得る。
4. PatchなしStudent(typo)を \(q_{\mathrm{patch}}\) へ蒸留する。

\[
\mathcal L_{\mathrm{patchKD}}
=
\frac1{\sum_i|A_i|}
\sum_i\sum_{t\in A_i}
KL(q_{\mathrm{patch},i,t}\parallel p_{S_\theta,i,t})
\]

Clean rowでは通常のself-distillationを使う。

\[
q_i=
\begin{cases}
p_{\mathrm{Base}}(x_i^c), & \text{clean row}\\
p_{\mathrm{patched\ Base}}(x_i^p;h_{\mathrm{Base}}(x_i^c)), & \text{noisy row}
\end{cases}
\]

\[
\mathcal L=KL(q_i\parallel p_{S_\theta})
\]

State lossとrhoを廃止できる。事前にtraining candidate上のcausal-window patchがrandom windowよりKLを改善し、悪化pair率が許容範囲であることを登録する。比較はBase、通常output matching、causal-window patched teacher、random-window patched teacher、必要ならall-layer teacherである。

主なriskはnoisy rowあたりclean Base + patched typo Base + Studentの3 forward、oracle cleanをtrainingだけで必要とすること、patchが悪化する例を教師にする可能性である。近接するintervention distillation研究との重複調査も実行前に必要である。

## 14. 候補D: SAE energyによるdata selection

**状態: 未実装・未登録の独立study候補**

Lossはoutput matchingのまま、noisy candidateをenergyで順位付けする。

\[
\mathcal L=\mathcal L_{\mathrm{out}}
\]

Edit count、operation、文書長のstratum内でhigh-energyとmatched randomを比較し、severity交絡を防ぐ。High / low / randomを同じtoken budgetで走らせる。

Energyは相関的OOD scoreであり、因果的localizationの主張には使えない。SAE feature lossやresidual-state lossと同時に混ぜず、**データ効率だけ**を検証する。

Kill条件は、matched randomとの差がない、clean harmが増える、natural/OOD typoで悪化する場合である。

## 15. 候補E: 介入効果によるdata selection

**状態: 未登録の探索案。優先度はC/Dより低い**

Pairごとの既存window restorationを、

\[
R_i(W^*)
=1-\frac{D_{\mathrm{patched},i}(W^*)}{D_{\mathrm{typo},i}}
\]

として事前計算し、高restorationまたはpatch-positive例へ学習容量を集中する。

比較には同件数・同token budget・edit count/operation/length matched random、low-restoration、通常output matchingを置く。

介入に基づくscoreである点はenergyより因果的だが、全候補のpatch forwardが高価で、修復しやすい例だけへ偏るselection biasがある。Patch Distillationの方が連続的なteacher分布を捨てないため優先しない。

## 16. 統合decision tree

~~~text
現行64M residual-state comparison
  ├─ proposalがoutput + controlsを上回る
  │    └─ 現行手法を追加seed・凍結評価・cross-modelへ
  ├─ behavior同等、auditだけproposal固有
  │    └─ 実用優位を主張せず機構差に限定
  └─ proposal固有差なし
       └─ raw residual cosine形式をkill

並行するSAE track
  WP-2
  ├─ fail -> SAEを後続へ使用しない
  └─ pass
       ├─ WP-3/4は診断として実行
       └─ WP-5
            ├─ G1 + G2 + G3 pass
            │    └─ 事前amendmentがある場合だけsuppression studyを草案化
            ├─ G1 + G3 pass、G2 fail
            │    └─ 現registryのstudy草案まで。固定C用の別kill testへ
            └─ G1 fail / seed不再現
                 └─ pair-specific suppression studyをkill

固定C用の別causal selection / held-out kill test
  ├─ matched-randomより優位、SAE seed再現
  │    └─ fixed-feature subspace studyを候補化
  └─ fail
       └─ fixed-feature学習をkill

raw residual kill + feature kill
  └─ Causal Patch Distillationを第一の別proposalとして事前登録
       └─ 不成立ならdata-selection候補D/Eを独立に検証
~~~

## 17. 禁止事項、再現性、報告規律

### 17.1 禁止事項

1. SAE結果を見て現行64M runのrho、window、loss、dataを変えない。
2. Tune / pre-PR / final / localization IDをSAE trainへ混ぜない。
3. WP-2/5の閾値を結果後に変更しない。
4. WP-5のG1合格だけで「causal featureを同定した」と表現しない。Pair-specific suppressionは事前amendment + G1/G2/G3、固定 \(C\) は別のcausal selection / held-out kill test合格まで学習しない。
5. Feature loss、energy selection、patch selectionを一つのstudyへ同時投入しない。
6. State/feature/L0/energy改善をbehavior gateの代わりにしない。
7. WP-6の学習を現行decision tree決着と新規事前登録前に開始しない。

### 17.2 Registryに保存するもの

- Base model revision
- SAE architecture、init seed、\(\lambda_{L1}\)、weights hash
- train/statistics record ID hashと近重複検査結果
- \(p_f,s\)、FVU、L0、dead rate、splice KL
- WP-2判定と許可された再学習履歴
- WP-3/4/5 data IDと予測登録日時
- WP-5のSAE init seed、typo seed、typo操作の混合・適用規則、bootstrap seed
- WP-5 operator、eligible規則、前提gate、主要gate、2-seed結果
- 各後継studyの事前登録とamendment log

### 17.3 再現順序

1. Data exclusion registryとmodel revisionを確認する。
2. Clean calibration 10Mでlayer別 \(\lambda_{L1}\) を規則どおり選ぶ。
3. `--training-budget` の `minimum` / `preferred` を実行前にregistryへ固定し、前者なら100M、後者なら200M clean activation tokensで3 SAEを学習する。
4. 分離10M cleanで \(p_f,s\) を計算して凍結する。
5. WP-2の4 gateとartifact completenessを検収する。
6. PassしたSAEだけを使い、各diagnostic cohortとSAE train manifestの非重複をfail-closedに確認してからWP-3/4を実行する。
7. WP-5開始前に未登録のdiagnostic 200のID/hashと非重複規則、typo seed、typo操作3種の混合・適用規則、eligible規則、full-state前提gate、G1要件を維持したG2 routingをamendmentで固定する。Bootstrap 10,000回 / seed 42はconfigの既登録値を使う。
8. 事前amendmentがありG1/G2/G3が全てpassした場合だけpair-specific suppressionを別studyとして事前登録する。固定 \(C\) は別のselection / held-out kill testを登録・合格してから候補化する。

### 17.4 Source of truth

| Source | Path / 状態 |
|---|---|
| SAE machine-readable config | [configs/sae/gemma4b-sae-v1.yaml](../configs/sae/gemma4b-sae-v1.yaml)、WP-1/2の実行仕様 |
| SAE registry | [configs/sae/registry-v1.yaml](../configs/sae/registry-v1.yaml)、data/run/gate provenance |
| 凍結plan | [docs/sae_track_plan_v1.ja.md](sae_track_plan_v1.ja.md)、WP予測・禁止事項 |
| SAE model/loss | [sae/model.py](../src/typo_robust_training/sae/model.py) |
| Data / duplicate guards | [sae/data.py](../src/typo_robust_training/sae/data.py) / [sae/duplicates.py](../src/typo_robust_training/sae/duplicates.py) |
| Runtime / runner | [sae/runtime.py](../src/typo_robust_training/sae/runtime.py) / [sae/runner.py](../src/typo_robust_training/sae/runner.py) |
| Metrics / registry code | [sae/metrics.py](../src/typo_robust_training/sae/metrics.py) / [sae/registry.py](../src/typo_robust_training/sae/registry.py) |
| 本書 | 説明・roadmap。machine-readable sourceと衝突時は上記を優先 |

WP-3/4/5専用runnerは本書更新時点で未実装であり、実装済みと読める記述を禁止する。後継candidate C--Eはconfigも登録もない。

### 17.5 結果の表現

- \(L_0\) / energy増加: 「typoと相関する表現逸脱」まで。
- WP-3 patchで減少: 「early patchがdownstream SAE指標を因果的に抑える」まで。
- WP-5 G1+G3 pass: 「このSAEの**全feature成分**が単層patch restorationの事前登録割合を運ぶ」まで。疎な固定feature集合の同定とは言わない。
- WP-5 G1/G2/G3 pass: 「全feature成分が十分で、pairごとのtypo-only feature除去も事前登録割合のrestorationを生む」まで。固定feature集合への一般化はしない。事前amendmentがなければG2 routingを後継認可に使わない。
- Feature trainingがbehavior controlに勝つ: 初めて「feature-targeted頑健化が有効」と主張可能。
- WP-5 fail: 「因果情報は今回の疎feature基底では十分に回収されない、またはdark matter側にある」と限定し、SAE一般の否定に拡張しない。
