# SAE 診断トラック v1

このトラックは、凍結済み 10M-token 頑健化比較と独立して進めます。GPU 5/6 の run は保護対象で、
設定変更・再起動・停止・途中の task accuracy 評価を行いません。SAE は GPU 1 を使い、凍結評価と
GPU・人手が競合する場合は必ず凍結評価を優先します。

## 作業順

1. output distribution matching と causal-window localized state distillation の 10M run を無変更で完走。
2. clean FineWeb-Edu のみで ReLU+L1 SAE を学習。層 5 は seed 42/43、層 20 は seed 42。
3. SAE を診断へ使う前に、事前登録済み WP-2 gate を適用。
4. 合格 SAE だけを Base 統合実験、checkpoint 遡及診断、feature 因果 kill test に使用。
5. feature 対象の後継学習は、既存 decision tree の決着と kill test 合格までは文書草案に限定。

SAE は `z = ReLU(Wx+b)`, `x_hat = Dz` とし、過完備率は 16 倍です。top-k は用いず、decoder の
各列を optimizer step ごとに単位ノルムへ戻します。3 点の L1 係数を同じ 1M-token clean stream
で較正し、median L0 が [30,150] に入るものから FVU 最小を選びます。typo や task performance を
係数選択に使いません。

追加corpus builderはpinしたFineWeb-Eduをshuffleせず再走査し、凍結済みtune、pre-PR、
final-test、localization-selection、localization-validationを必須除外入力とします。
record/source/group ID、正規化contentの一致、およびcharacter 5-gram MinHash候補のうち
exact Jaccardが0.99以上の近重複を除外します。minimum budgetには100M学習activation、
10M held-in統計activation、未使用のsplice 200文書を含めます。

WP-2 は FVU <= 0.35、median L0 in [30,150]、dead feature 率 <= 20%、splice KL 中央値
<= 0.15 nats/token、`p_i` と再構成誤差中央値 `s` の保存を要求します。WP-5 は層 5 の 2 seed で
`median(R_z) >= 0.5 median(R_full)` と因果方向の再現を必須とします。合格前に neuron/head/feature
を因果 component として対外的に主張しません。
