# Typo頑健化 評価プロトコル v2.0 事前登録

状態: **学習armの結果を参照する前の凍結候補**

machine-readable contract: `configs/robustness-evaluation-v2.yaml`

registry template: `configs/robustness-evaluation-v2-registry.template.json`

## 1. 目的と仮説

本プロトコルは、Linear Probeで同定したword-identity noise penaltyの低下境界
`b`より前をfreezeし、`b`以降だけにLoRAを置いてclean-teacher/noisy-studentの
output distribution matchingを行う手法を評価する。旧early-window hidden-state
distillationは主手法に含めない。

確証factorialはBaseとv7学習実装の次の5条件である。

1. `base`: 学習なし。
2. `factorial-all-layers-all-tokens`: 全decoder層×全aligned token。v7内で予算と
   LoRA容量を揃えたKojima-style対照。
3. `factorial-all-layers-downstream-horizon`: 全decoder層×編集後horizon。
   targetingだけを変更する対照。
4. `factorial-probe-suffix-all-tokens`: Probe境界`b`以降×全aligned token。
   placementだけを変更する対照。
5. `factorial-probe-suffix-downstream-horizon`: Probe境界`b`以降×編集後horizon。
   **完全提案条件**。
6. `factorial-random-layers-downstream-horizon`: Probe suffixと同数のrandom layer×
   編集後horizon。場所の情報量を検定する対照。maskは全learning seedで共通の
   `sha256-seed42-count-matched-random-freeze/v1`に固定し、seedごとに引き直さない。

さらにMistralだけで、公開手順を忠実に再現する
`kojima-faithful-output-matching`を直接比較する。主要比較には提案と同じ
matched replication seed `{42,43,44}`を使う。公開defaultのseed 1 runは
`kojima-faithful-output-matching-public-seed1-anchor`として再現性確認用に保持するが、
seed平均、bootstrap、優越性CIへ混ぜない。

完全な主張には、完全提案条件がBaseに対してclean非劣性かつtypo優越であり、
all/all、all/horizon、suffix/all、random/horizonの4条件を所定の比較で上回ることが
必要である。Mistralではmatched-seed faithful Kojimaにも勝つ必要がある。
学習契約は未統合feature branchのcommitではなく、config schema
`robustness-adapter-training-config/v7`、factorial method identity
`probe-output-factorial/v1`、evidence schema
`probe-output-factorial-evidence-binding/v1`、faithful method identity
`kojima-faithful-output-matching/v1`という意味的identityへ固定する。実行に使う最終merged
commitとcanonical source-tree SHA-256は、統合完了後かつ学習開始前にregistryへ記録する。
source-tree SHA-256は`git ls-tree -r --full-tree HEAD`のraw LF bytesをSHA-256化する
`sha256-of-git-ls-tree-r-full-tree-head-lf/v1`で一意に計算する。未統合commitを科学契約として
扱わない。

生成・採点ハーネスはv1.4から変更しない。v2 configに記録したv1.4 exact SHA-256が、
`generation`（paper CoT prompt、task別shot数、task別抽出器、greedy、bf16、最大512
new tokens、抽出不能=不正解）、`typos.eligibility`、`corpus_runtime`を継承契約として
固定する。さらにfull generation、prompt+shot、extractor、decoding、eligibility、
corpus runtimeをcanonical JSONとして個別にSHA-256化し、v2 protocolとsealed registryの
両方で一致を必須化する。v2のseverity・母集団・統計だけを変更し、armごとにprompt、
decoding、抽出器を分岐させない。

## 2. 固定モデル

モデル集合はseverity calibrationより先に固定し、calibration後の交換を禁止する。

| role | model | exact revision |
|---|---|---|
| development anchor | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Kojima direct comparison | `mistralai/Mistral-7B-v0.1` | `7231864981174d9bee8c7687c24c8344414eae6b` |

モデルごとのBase gapを見て一方を削除したり、第三モデルへ差し替えたりしない。
いずれかでcalibrationが成立しない場合も、その不成立を結果として残す。

## 3. Base-only severity calibration

### 3.1 データ分離

calibrationはGSM8K、MMLU、ARC-Challenge、MMLU-Pro、CommonsenseQAから
各200件、計1,000件をstable hashで選ぶ。IDは学習、Linear Probe、tune、
pre-PR、finalと非重複とする。calibrationに入力できる推論結果は、上表のexact
revisionをadapterなしで動かしたBaseのものだけである。

各観測は`condition=base`、`adapter_checkpoint_sha256=null`、
`training_run_sha256=null`を満たさなければならない。提案手法、baseline、途中
checkpointを含むadapter出力は、severity選択へ一切入力できない。

### 3.2 固定候補と選択式

候補編集数は事前に

\[
K=\{2,4,8\}
\]

へ固定する。各item・severityにつき3変種を静的生成する。操作はQWERTY近傍置換、
削除、重複、targetは適格なquestion wordから一様・非復元抽出する。数字、数式、
URL、email、identifier、選択肢本文、gold answerは対象外とする。

モデル`m`、severity `k`についてtask等重みのBase精度を
`A_clean(m)`、`A_typo(m,k)`とし、

\[
g_{m,k}=100\{A_{clean}(m)-A_{typo}(m,k)\}
\]

とする。次をすべて満たす最小の`k`をprimary severity `k*`とする。

\[
\frac{1}{|M|}\sum_m g_{m,k}\ge 8\text{ pp},\qquad
\min_m g_{m,k}\ge 5\text{ pp},\qquad
\min_m\frac{A_{typo}(m,k)}{A_{clean}(m)}\ge 0.5.
\]

第一条件は2ppの実用改善に対して十分な修復余地を確保し、第二条件は一方のモデル
だけがmacroを駆動することを防ぎ、第三条件は極端なfloorを避ける。

該当する`k`がなければstatusを`stopped-no-eligible-severity`として保存する。
`16`編集を追加する、閾値を下げる、モデルを交換する、特定task/itemを除く、という
fallbackは禁止する。別設計を行う場合はprotocol versionを上げ、全armを新規に
評価する。

### 3.3 凍結成果物

calibrationのitem manifest、実現typo manifest、Base observations、選択結果を
それぞれSHA-256でregistryへ登録する。モデル横断で同じsource itemと同じ実現typo
textを使用する。選択後に再生成しない。

ファイル全体のSHA-256だけでは十分でない。item manifestは各行に`task`、
`record_id`、`source_text`、`source_text_sha256`、`reference_answer`、
`reference_answer_sha256`を持ち、typo manifestはsourceへの
参照に加えて`severity_edit_count`、`variant`、`realized_typo_text`、
`realized_typo_sha256`を持つ。較正CLIは両manifestを厳格にparseし、textからhashを
再計算したうえで、全observationのrecord/source/reference-answer/typo hashが対応行と一致すること、
過不足なく全gridを覆うことを検証する。したがって、古いmanifestを単に再hashした
ファイルや、任意ファイルのdigestだけを登録したものは有効な凍結成果物ではない。
text digestは正規化を加えないUTF-8 exact text bytesのSHA-256とする。

凍結済みBase observationsを選択規則へ通すコマンドは次の通り。終了コード`0`は
severity選択、`2`は規則どおりの停止、`1`は入力・provenance違反を表す。

```bash
uv run --project projects/typo-robust-training typo-cot \
  calibrate-evaluation-v2-severity \
  --protocol projects/typo-robust-training/configs/robustness-evaluation-v2.yaml \
  --base-observations /path/to/base-observations.jsonl \
  --item-manifest /path/to/calibration-items.jsonl \
  --realized-typo-manifest /path/to/calibration-typos.jsonl \
  --output-dir /path/to/severity-calibration
```

## 4. 確証評価

### 4.1 母集団

5 taskから各1,000件、計5,000件をstable hashで抽出する。calibrationおよび全ての
学習・probeデータとはID非重複にする。各itemにつき`k*` typoを2変種生成し、全arm・
全seedで同じbytesを読む。Baseのclean正解・typo不正解へ条件付けたflip cohortは
診断に限り、primaryには使用しない。
同じ`model × arm × item × learning seed`のclean入力はtypo variantによらず同一なので、
`clean_correct`がvariant間で変化する出力は評価配管の不整合としてfail closedする。
各outcome行は`source_text_sha256`、`reference_answer_sha256`、
`realized_typo_sha256`も持つ。統計処理は、確証item
manifestと2-variant typo manifestをUTF-8 exact textから再検証し、各行のhashが対応する
凍結行と一致する場合に限って開始する。同じ`record_id`と`variant`というラベルだけでは
同一評価例の証拠にならず、arm・seed・model間で入力bytesまたはreference answerが
異なる出力は拒否する。

旧v1.4の`random-2`は`secondary-continuity-only`として残す。旧結果を破棄せず、
同じhash-bound manifest上で全新armを再評価する。primary severityの代わりには
用いない。

### 4.2 確証endpoint

以下はintersection-unionとして**すべて**満たす必要がある。

1. clean非劣性: 提案−Baseのmodel/task等重みmacro 95% CI下限が`-1.0pp`より大きい。
   task単位の点推定が`-3.0pp`以下にならず、clean PPL比が`1.02`以下。
2. 対all/all優越: full−all/allの95% CI下限が0より大きく、点推定が`+1.5pp`以上。
3. placement寄与: full−all/horizonの95% CI下限が0より大きい。
4. targeting寄与: full−suffix/allの95% CI下限が0より大きい。
5. Probe配置特異性: full−random/horizonの95% CI下限が0より大きい。
6. 対Base typo改善: full−Baseの95% CI下限が0より大きく、点推定が`+2.0pp`以上。
7. Mistral直接比較: full−faithful Kojimaの95% CI下限が0より大きい。両armとも
   matched seed `{42,43,44}`だけを使う。
8. seed方向: 3 seedすべてでfull−all/allが正で、Mistralでは
   full−faithful Kojimaも正。

clean−typo gapの縮小単独はclean性能を下げても達成できるため、成功判定に使わない。
wrong→right、right→wrong、netは必ず同時に報告する。

### 4.3 検出力

paired binary差の不一致率を`q=0.12`、検出対象を`1.5pp`とすると、近似必要数は

\[
n\simeq
\frac{(z_{0.975}+z_{0.8})^2q}{0.015^2}
\approx4,200.
\]

5,000 itemは約1.5ppの差に対する80%以上の検出力を意図する。1pp差を安定して
検出する設計ではなく、1pp未満を強い性能優位と表現しない。

## 5. Clustered paired bootstrap

統計単位はsource itemであり、`2 typo variants × 3 matched learning seeds`を独立な
6標本とは数えない。比較arm`a,b`のitem差を

\[
d_{m,t,i}=
\frac{1}{2\times3}\sum_{v=1}^{2}\sum_{s=1}^{3}
\{Y_{b,m,t,i,v,s}-Y_{a,m,t,i,v,s}\}
\]

とする。Baseにはlearning seedがないため、Base値はvariant内で一度だけ用いる。
faithful Kojimaの主要比較もseed `{42,43,44}`で同じ式を用いる。公開seed 1 anchorは
この式へ投入せず、単独の記述値としてのみ報告する。
各outcome行はmodel IDだけでなくexact revisionも持ち、§2のrevisionと異なる行は
統計処理前に拒否する。
各bootstrap反復ではmodel/task cell内でsource itemを復元抽出し、cell平均を計算後、
taskとmodelを等重みで平均する。arm、variant、seedはitemと一緒に動かし、別々に
再標本化しない。10,000反復、seed 42、95% percentile CIを用いる。

各model・seed・cellの二値比較にはexact McNemarを補助的に併記する。seed別結果、
seed平均、seed SDをすべて公開する。

## 6. Secondaryとdiagnostic

- 固定severity曲線: random-1/2/4/8。
- 旧random-2: v1.4との連続性。
- 未学習操作: transposition-2。
- natural injection。
- clean PPL、Base→adapter clean KL、文字忠実性probe。
- trainable parameter数、optimizer状態、peak memory、student token数、GPU時間。
- frozen Linear Probeによる層別noise penalty。
- Activation Patchingの追加patch gain。

Linear Probeとpatchingは機構診断でありblocking gateではない。行動性能が成立しても
diagnosticが成立しない場合、頑健化効果は主張できるが、暗黙デノイジング経路が
改善したという機構主張は行わない。

## 7. v1.4との不整合監査

v1.4を黙って書き換えず、v2を別schemaとして扱う。主な差は次の通り。

| 項目 | v1.4 | v2.0 | 移行規則 |
|---|---|---|---|
| primary typo | random-2固定 | Base-onlyで`{2,4,8}`から一度選択 | v1 random-2はsecondaryに残す |
| sealed規模 | 500件/task中心 | 1,000件/task、5 task | v1のCIと混在させない |
| typo変種 | 1実現値 | 2実現値/item | 変種を独立nと数えない |
| model | Gemma中心 | Gemma/Mistral exact revision固定 | calibration後の交換禁止 |
| bootstrap | learning-seed/task/item階層 | item cluster、seed/variantを先に平均 | 3 seedを3nと数えない |
| 対照 | Base対adapter中心 | Base+5 factorial、Mistral-only faithful | 全armを同一hashで再実行 |
| mechanistic audit | random-2、nonblocking | 同じくsecondary、nonblocking | 成功指標へ昇格しない |

v1のfrozen manifestをprimaryへ流用せず、v2用calibration/confirmatory manifestを新規に
hash凍結する。一方、旧random-2のregistry hashをv2 registryから参照し、過去評価の
追跡可能性を維持する。

## 8. 停止・縮小規則

- calibration不成立: severity/modelをshopせず停止。
- all/allが所定予算でBaseを改善しない: matched-budget Kojima-style対照が成立せず、
  その比較に対する性能優位を主張しない。faithful Kojimaとの直接比較は別に報告する。
- clean非劣性不合格: 提案手法失敗。
- Base比typo改善不合格: 頑健化手法として失敗。
- all/allと同等: matched-budget性能優位なし。parameter/GPU時間が小さければ効率結果のみ。
- all/horizonと同等: Probe suffix配置の付加価値は不成立。
- suffix/allと同等: downstream-horizon targetingの付加価値は不成立。
- random/horizonと同等: Linear Probeによる配置選択の価値は不成立。
- Mistral faithfulと同等: Kojimaへの直接的性能優位は不成立。
- 低token予算でのみ優位: データ効率主張に縮小。
- 1モデルだけ成功: model-specificと明記。
- 開封後の閾値、severity、モデル、item、loss、境界変更は禁止。

## 9. Registryの二段階freeze

学習開始条件と評価開封条件を同じstateにすると、まだ存在しないcheckpoint hashを学習前に
要求する循環依存が生じる。そこでregistryは次の二段階だけを順に通る。

1. `training-preregistered`: protocol、severity、item/typo、model、Linear Probe成果物、
   arm別training config/data、実装commit/source treeを固定する。checkpoint registry、公開seed 1
   checkpoint、opening logはすべてnullでなければならない。学習runnerが受理できるのはこのphaseだけ。
2. `evaluation-opening-sealed`: 学習完了後、全checkpoint registryとopening logを追記する。
   確証評価runnerが受理できるのはこのphaseだけ。学習前のphaseを評価へ流用してはならない。
   このphaseは直前のexact `training-preregistered` registry SHA-256を必須参照し、post-only
   fields以外の差分を拒否する。

templateのnullを埋めるだけでは足りない。各phaseへ進める前に、該当する次の条件をすべて確認する。

1. protocol file自身のSHA-256。
2. exact model revision inventory。
3. calibration item/typo/Base observation/resultの各SHA-256。
4. selected severityが`{2,4,8}`の最小eligible値であること。
5. `adapter_outputs_used_for_calibration=false`。
6. `severity_grid_extended=false`。
7. `model_inventory_changed_after_calibration=false`。
8. 学習実装の意味的identityがprotocolと一致し、実行に使う最終merged commitとcanonical
   source-tree SHA-256がregistryへ記録されていること。
9. confirmatory item/2-variant typo/5 factorial arm定義、Linear Probe成果物、training config、
   training data registryの各SHA-256（`training-preregistered`で必須）。
10. 5 factorial arm checkpoint registryのSHA-256（`evaluation-opening-sealed`で必須）。
11. Mistral faithful matched seed `{42,43,44}` registryのSHA-256（同上）。
12. 公開seed 1 anchor checkpointのSHA-256と`reproducibility-only-not-pooled`表記（同上）。
13. opening logのSHA-256（同上）。
14. v1 random-2 frozen registryのSHA-256。

さらに、training、Linear Probe選択、Linear Probe検証、tune、pre-PR、calibration、
confirmatoryの全roleを列挙したID manifestを凍結し、record IDだけでなくexact source-text
SHA-256のrole間重複も拒否する。これにより、同じ本文へ別IDを振る再ラベル化も分離とは
扱わない。評価を開く実行環境はregistryのfinal merged commitを`HEAD`としてcheckoutし、
tracked/untracked差分がゼロでなければならない。登録commitの`git ls-tree` hashだけが正しくても、
dirty worktreeから評価コードを実行することは禁止する。

runtime loaderはregistryに記録されたSHA-256文字列をそのまま信用しない。実際の
calibration item/typo/Base observation/resultを再解析してBase-only calibrationを再計算し、
selected severityと要約統計が一致することを検証する。同様に、実際の全role ID manifestを
再解析してrole間のrecord ID/source-text SHA-256非重複を検証し、confirmatory manifestの
source text、reference answer、realized typoを再hashする。その後にのみ、各実ファイルの
SHA-256をregistryと照合する。ファイルを改変して自己再hashしたregistryも、このsemantic
validationを通過しない限り拒否する。

同じ規則を学習・checkpoint側の成果物にも適用する。Linear Probe、factorial arm、training
config/data、legacy v1 registryは学習前に実ファイルを読み、その実hashと照合する。評価開封時は
matched-seed registry、公開seed 1 checkpoint、全arm checkpoint registry、opening logの実ファイルを
照合する。64桁の架空文字列や、存在しない成果物をregistryへ記録しただけではphaseを通過できない。

これらのいずれかが欠けるregistryから、確証評価を開始してはならない。
