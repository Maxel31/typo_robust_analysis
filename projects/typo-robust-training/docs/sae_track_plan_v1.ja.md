# SAE 診断トラック v1

このトラックは、凍結済み 10M-token 頑健化比較と独立して進めます。GPU 5/6 の run は保護対象で、
設定変更・再起動・停止・途中の task accuracy 評価を行いません。SAE は事前登録した専用GPUを使い、凍結評価と
GPU・人手が競合する場合は必ず凍結評価を優先します。

## 作業順

1. output distribution matching と causal-window localized state distillation の 10M run を無変更で完走。
2. clean FineWeb-Edu のみで ReLU+L1 SAE を学習。層 5 は seed 42/43、層 20 は seed 42。
3. SAE を診断へ使う前に、事前登録済み WP-2 gate を適用。
4. 合格 SAE だけを Base 統合実験、checkpoint 遡及診断、feature 因果 kill test に使用。
5. feature 対象の後継学習は、既存 decision tree の決着と kill test 合格までは文書草案に限定。

SAE は `z = ReLU(Wx+b)`, `x_hat = Dz` とし、過完備率は 16 倍です。top-k は用いず、decoder の
各列を optimizer step ごとに単位ノルムへ戻します。3 点の L1 係数を同じ凍結済みclean stream
で較正し、median L0 が [30,150] に入るものから FVU 最小を選びます。typo や task performance を
係数選択に使いません。

追加corpus builderはpinしたFineWeb-Eduをshuffleせず再走査し、凍結済みtune、pre-PR、
final-test、localization-selection、localization-validationを必須除外入力とします。
record/source/group ID、正規化contentの一致、およびcharacter 5-gram MinHash候補のうち
exact Jaccardが0.99以上の近重複を除外します。minimum budgetには100M学習activation、
10M held-in統計activation、未使用のsplice 200文書を含めます。

### 実行前データamendment（2026-08-15）

最初のcorpus buildは、artifact作成やmodel forwardの前に、保護対象source manifest内の
正規化content重複131件を検出して停止しました。source manifestのhashと予約済み30,000件の
prefixは変更しません。予約prefixの全行を除外anchorとして保持し、eligible行だけを既存の
`sae-clean-source-order/v1`で走査して、予約prefixまたは先行eligible行と正規化contentが同じ行を
除きます。初期eligibleは126,901件・51,735,090 source tokensとなり、不足分は新規FineWeb-Eduで
補います。acceptance gate、kill-test閾値、評価項目、model設定は変更していません。

続くcalibration入力検査はruntime初期化前に停止し、最初のsupplementへ保護対象record IDが8件
再流入していることを検出しました。これらは正規化content重複としてeligibleから除かれた後、
異なるsegmentationでFineWeb-Eduから再生された文書です。このため、eligibleから除いた行も含め、
保護manifestの全record/source/group IDをsupplement除外集合へ保持します。最初のsupplementは隔離して
再構築し、model forwardは開始していません。派生eligible値はcorpus build、calibration、training、
validationの入力読込時にこの事前登録と機械的に照合します。

### 1回限りの較正amendment（2026-08-15）

初期の1M-token較正はcleanデータ上で489 optimizer stepを完走しましたが、登録済み係数
`[0.0001, 0.0003, 0.001]`はいずれも凍結済みmedian-L0範囲へ入りませんでした。最大係数でも
層5のmedian L0は4,874、層20は3,856で、上限150の25倍超でした。WP-2、WP-3、WP-5の結果を
見る前に、事前登録で許可した1回の調整を消費し、係数を対数間隔の
`[0.01, 0.1, 1.0]`へ固定します。その後の反証優先レビューにより、489 optimizer stepでは
encoder biasが疎な領域へ到達するには不足することが示されました。product側optimizer経路を使った
合成反証では、係数を1,000倍にしてもmedian L0は1.2%しか変化しない一方、step数を10倍にすると
大きく低下しました。このため、同じ事前登録済みlambda/token amendmentの範囲内で、較正予算も
1Mから10M activation tokensへ引き上げます。決定的な1M-token bufferをちょうど10本、順次
streaming学習し、学習後に同じ10M-token activation streamを再生して最終統計を計算します。
1M-tokenというbufferサイズと10本という分割数も同じregistryに固定します。選択規則、clean-only
データ、WP-2/WP-5 gateは変更しません。失敗したW&B run IDと6個の観測L0をregistryへ記録し、
追加の係数・token調整は以後認めません。

その後GPU 1が無関係なworkloadで占有されたため、待機中の再較正をGPU 0へ割り当てました。
これは運用上の変更だけで、データ、モデル、loss、閾値、評価設定は変えていません。

WP-2 は FVU <= 0.35、median L0 in [30,150]、dead feature 率 <= 20%、splice KL 中央値
<= 0.15 nats/token、`p_i` と再構成誤差中央値 `s` の保存を要求します。WP-5 は層 5 の 2 seed で
`median(R_z) >= 0.5 median(R_full)` と因果方向の再現を必須とします。合格前に neuron/head/feature
を因果 component として対外的に主張しません。

初回の失敗WP-2 ledgerは変更不能です。新規の初回v1事前登録は絶対pathかつ作成済みdirectoryの
`wp2_project_root`を含めます。既存のrepository内unrooted v1はbyte単位で不変に保ち、既存救済chainの
証拠としてのみ読み込めます。runnerを実装しただけではfull-bundle救済は許可されません。別レビュー済み
authorizationが、初回config・事前登録・training・validation・acceptance・ledgerの完全なhash chainと、
単一の改訂config / 事前登録hashを結ぶ必要があります。改訂事前登録v2自体が同じproject rootを持つ
trusted retry markerであり、`initial_attempt_ledger_path`は厳密に
`wp2_project_root/wp2_attempts.json`と一致させます。救済modeはCLIから選択も回避もできず自動的に
強制され、authorizationはv2事前登録全体のdigestをbindします。

初回と救済のvalidationはいずれも、output作成やGPU/runtime workより前にproject rootへ`O_EXCL`
予約を作成します。各排他recordはfile本体をfsyncした後、事前作成済みの親directoryもfsyncします。
救済trainingもruntime/model初期化前に唯一のclaimを作成し、resume時はoutputを
一切変更する前にclaimを完全照合します。claimには、順序付き入力のraw hash、memoryへ実際にloadした
全source値とreserved/eligible role・順序のcanonical digest、load済みL1 mapping、レビュー対象source
closure、実効W&B identityもbindします。parse/load直後にはraw入力を再hashし、救済invocation全体で
project-rootのnonblocking `flock`を保持します。

claimだけが作成された区間でcrashした場合、同一claimかつoutputが未作成/空または完全一致するsource
registryだけを持つ場合に限り再開できます。製品が残したsource-registry一時fileも、正のASCII PID標準表記名を持ち、
期待するcanonical registryのbyte prefixである同一user所有・single-linkの通常file・非symlinkの場合に限り許可し、
権威的registryとしてはloadしません。
未知のruntime artifactはfail closedです。runtime/model/
optimizer初期化後、W&B・activation・学習より先にcursor-zero checkpointをatomicかつdurableに保存し、
claim由来のW&B ID intentもremote初期化前に永続化します。通常runのresume意味論は変更しません。
救済validationはclaim済みtraining-run SHAをmodel load直前と
最終completion記録直前に再検証し、completionへattempt番号、pass/fail、親ledger、training、予約、
validation、acceptanceのhashを保存します。crash後も予約を暗黙再利用しないfail-closed設計です。
出力親directoryを変えても追加bundleは作れません。初回validation outputはproject root外でもよく、
ledgerが絶対pathを、authorizationがartifact hashをbindし、project-root ledgerだけがbudgetのauthorityに
なります。本実装は救済authorization、v2救済事前登録、科学的な改訂値を追加しません。
