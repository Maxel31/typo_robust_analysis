# Typo頑健化 評価プロトコル v2.0 事前登録

状態: **学習armの結果を参照する前の凍結候補**

machine-readable contract: `configs/robustness-evaluation-v2.yaml`

registry template: `configs/robustness-evaluation-v2-registry.template.json`

## 1. 目的と仮説

本プロトコルは、Linear Probeで同定したword-identity noise penaltyの低下境界
`b`より前をfreezeし、`b`以降だけにLoRAを置いてclean-teacher/noisy-studentの
output distribution matchingを行う手法を評価する。旧early-window hidden-state
distillationは主手法に含めない。

比較する確証armは次の4条件である。

1. `base`: 学習なし。
2. `output-matching-all-layers`: 全decoder層へLoRAを置くKojima型対照。
3. `probe-boundary-output-matching`: `b`以降だけへLoRAを置く提案条件。
4. `random-freeze-output-matching`: `b`個の層をhashで無作為にfreezeし、残りへ
   同一LoRAを置くparameter-count対照。freeze maskはmodel revision、protocol、
   learning seedから結果を見る前に決定する。

完全な主張には、提案条件がBaseに対してclean非劣性かつtypo優越であり、
全層対照とrandom-freeze対照の両方を上回ることが必要である。random-freezeに
勝てない場合は、freeze自体の効果は残り得るが、Linear Probeが配置を情報したとは
主張しない。

## 2. 固定モデル

モデル集合はseverity calibrationより先に固定し、calibration後の交換を禁止する。

| role | model | exact revision |
|---|---|---|
| development anchor | `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Kojima-family replication | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` |

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

旧v1.4の`random-2`は`secondary-continuity-only`として残す。旧結果を破棄せず、
同じhash-bound manifest上で全新armを再評価する。primary severityの代わりには
用いない。

### 4.2 確証endpoint

以下はintersection-unionとして**すべて**満たす必要がある。

1. clean非劣性: 提案−Baseのmodel/task等重みmacro 95% CI下限が`-1.0pp`より大きい。
   task単位の点推定が`-3.0pp`以下にならず、clean PPL比が`1.02`以下。
2. 対全層typo優越: 提案−全層の95% CI下限が0より大きく、点推定が`+1.5pp`以上。
3. 対Base typo改善: 提案−Baseの95% CI下限が0より大きく、点推定が`+2.0pp`以上。
4. Probe配置特異性: 提案−random-freezeの95% CI下限が0より大きい。
5. seed方向: 3 seedすべてで提案−全層のtypo差が正。

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

統計単位はsource itemであり、`2 typo variants × 3 learning seeds`を独立な6標本とは
数えない。比較arm`a,b`のitem差を

\[
d_{m,t,i}=
\frac{1}{2\times3}\sum_{v=1}^{2}\sum_{s=1}^{3}
\{Y_{b,m,t,i,v,s}-Y_{a,m,t,i,v,s}\}
\]

とする。Baseにはlearning seedがないため、Base値はvariant内で一度だけ用いる。
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
| 対照 | Base対adapter中心 | Base/全層/Probe境界/random-freeze | 全armを同一hashで再実行 |
| mechanistic audit | random-2、nonblocking | 同じくsecondary、nonblocking | 成功指標へ昇格しない |

v1のfrozen manifestをprimaryへ流用せず、v2用calibration/confirmatory manifestを新規に
hash凍結する。一方、旧random-2のregistry hashをv2 registryから参照し、過去評価の
追跡可能性を維持する。

## 8. 停止・縮小規則

- calibration不成立: severity/modelをshopせず停止。
- 全層対照が所定予算でBaseを改善しない: Kojima型再現が成立せず、対Kojima優位を
  主張しない。
- clean非劣性不合格: 提案手法失敗。
- Base比typo改善不合格: 頑健化手法として失敗。
- 全層と同等: 性能優位なし。parameter/GPU時間が小さければ効率結果のみ。
- random-freezeと同等: Linear Probeによる配置選択の価値は不成立。
- 低token予算でのみ優位: データ効率主張に縮小。
- 1モデルだけ成功: model-specificと明記。
- 開封後の閾値、severity、モデル、item、loss、境界変更は禁止。

## 9. Registryをsealedへ進める条件

templateのnullを埋めるだけでは足りない。次をすべて確認してからstateをsealedへ進める。

1. protocol file自身のSHA-256。
2. exact model revision inventory。
3. calibration item/typo/Base observation/resultの各SHA-256。
4. selected severityが`{2,4,8}`の最小eligible値であること。
5. `adapter_outputs_used_for_calibration=false`。
6. `severity_grid_extended=false`。
7. `model_inventory_changed_after_calibration=false`。
8. confirmatory item/2-variant typo/全arm checkpoint registryの各SHA-256。
9. v1 random-2 frozen registryのSHA-256。

これらのいずれかが欠けるregistryから、確証評価を開始してはならない。
