# Mistral Linear Probe cohort構築契約 v1

`build-probe-transition-data` は、`select-probe-transition` が読む既存schemaの入力を、
モデル出力を一切参照せずに構築するCPU前処理です。対象は
`mistralai/Mistral-7B-v0.1@7231864981174d9bee8c7687c24c8344414eae6b` です。

## 入力

- tracked template:
  `configs/proposals/mistral7b-probe-transition-data.template.yaml`
- `robustness-clean-record/v1` だけを含む、専用のFineWeb-Edu
  `training_sources.jsonl`形式manifest
- training / localization / tune / pre-pr / sealedの5 tierを含む
  `typo-protected-split-registry/v1`
- exact Mistral tokenizerのfrozen producer runと、その事前登録済みrecord SHA-256

source manifestはFineWeb-Eduの固定revision
`fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`だけを許します。既存training streamを
そのまま候補sourceへ流用してはいけません。protected training tierとのgroup / parent /
normalized-content照合により、そのような流用は除外され、quota不足ならrun全体が失敗します。
同じくlocalization / tune / pre-pr / sealedのいずれかに現れるrecord、source group、parent、
normalized contentも使用不可です。したがってproduction入力は、これら5 tierすべてと照合済みの
**別途用意した未使用clean pool**でなければなりません。このcommand自身はFineWeb-Eduをdownload・
抽出しません。明示pathのmanifestを検証して消費するだけです。

templateのstratum quotaを凍結する前に行った一度限りの実現可能性監査では、既存のpinned
FineWeb-Edu training-source poolを用いてexact tokenizerの発生頻度だけを数えました。この監査は
「不可能な全直積を仕様にしない」ためのmodel-output-free設計作業であり、そのtext/IDはproduction
class選択にもcohort構築にも使用しません。各production run内の20% feasibility splitも、上記の
**別の未使用pool**から取り、残り80%の実cohortとは完全に非重複です。

## 選択規則

1. 各documentの最大1,024文字の決定的prefixから、4--20文字のlowercase ASCII語を列挙する。
2. documentごとにSHA-256順で1語だけをword-identity class候補にする。
3. source record hashのmodulo 5により、20%のfeasibility splitと80%のmaterialization splitへ
   **tokenize前に**分離する。両者はsource/group/parent/content単位で交差しない。
4. feasibility splitの出現頻度だけで17 classを選び、同split上で凍結quotaを一度模擬充足する。
   この事前検収に使ったrecordはfit/selection/validationへ一切流さない。
5. keyboard-neighbor substitution、deletion、duplicationを各候補へ1回ずつ決定的に生成する。
6. exact tokenizerはtoken数の差を `same` / `plus-one` / `plus-two-or-more` に分類するためだけに
   使用する。hidden state、logit、loss、accuracyなどのモデル出力は計算しない。
7. group、parent、clean/noisy normalized contentのいずれかが一致するrecordを、role横断かつ
   protected tier横断で同一componentとみなし、複数roleへ割り当てない。
8. 17 classについて、fitはclassあたり20 clean record、selection/validationはclassあたり9 pairを
   割り当てる。同時に、cohort全体のoperation×token-inflation quotaをexact max-flowで満たす。
   quotaを満たせなければ出力を残さず失敗する。

class balanceとstratum balanceは、既存producer contractどおり**別々の大域制約**です。各classに
3操作×3 bucketの全直積を要求してはいけません。exact Mistral tokenizerを用いたモデル出力なしの
実現可能性監査では、例えばduplicationの`same`はほぼ発生せず、全直積は不成立でした。凍結templateは
3操作をすべて含めたうえで、実現可能な7 stratumだけを使います。operationごとの総数は各51 pair、
合計153 pair/roleです。これは結果を見て緩和する規則ではなく、production構築前に固定したquotaです。

fitにはclean textしか含めません。selectionとvalidationだけが1-edit clean/typo pairです。
2つのprobe fitは既存のclass-stratified SHA-256 balanced-halves規則により、fit cohortの互いに
素な半分を使用します。

## 実行

```bash
TRAIN_PROJECT=projects/typo-robust-training

uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-probe-transition-data \
  --template "${TRAIN_PROJECT}/configs/proposals/mistral7b-probe-transition-data.template.yaml" \
  --template-sha256 "${PROBE_TEMPLATE_SHA256}" \
  --source-manifest /absolute/path/to/disjoint-fineweb-edu-clean-sources.jsonl \
  --source-manifest-sha256 "${PROBE_SOURCE_SHA256}" \
  --protected-registry /absolute/path/to/protected-split-registry.json \
  --protected-registry-sha256 "${PROTECTED_REGISTRY_SHA256}" \
  --tokenizer-freeze-run /absolute/path/to/tokenizer-attestation-freeze-run.json \
  --tokenizer-freeze-run-sha256 "${TOKENIZER_FREEZE_RUN_SHA256}" \
  --output-dir /absolute/path/to/mistral-probe-transition-data
```

実行checkoutはcleanでなければならず、builderはそのcommitをproducer configへ固定します。
template / source manifest / protected registryのSHA-256も実行前に外部で固定し、CLIへ明示します。
入力symlink、duplicate JSON key、source content hash不一致、外部pin後の再hash、tokenizer
freeze-run不一致、protected tier相互のtransitive overlapはすべてfail-closedです。

## 出力

- `class_inventory.json`: `typo-word-identity-classes/v1`
- `fit_manifest.json`: clean-only `typo-probe-cohort/v2`
- `selection_manifest.json`: paired `typo-probe-cohort/v2`
- `validation_manifest.json`: paired `typo-probe-cohort/v2`
- `protected_split_registry.json`: 入力registryのbyte-identical copy
- `probe_cohort_feasibility.json`: materializationと非重複な20% splitでの事前quota検収
- `probe_producer_config.json`: 全入力hashと実行commitを固定した
  `typo-linear-probe-producer-config/v4`
- `build_probe_transition_data_run.json`: source、tokenizer、規則、artifact hashの監査記録。
  `self_hash`は当該fieldを除くcanonical JSONのSHA-256。表示された値を事前登録へ外部pinする

GPU側は個別manifestを任意に渡さず、外部pin済みbundleだけを受理します。

```bash
uv run --project "${TRAIN_PROJECT}" --locked typo-cot select-probe-transition \
  --cohort-build-run /absolute/path/to/mistral-probe-transition-data/build_probe_transition_data_run.json \
  --cohort-build-run-sha256 "${PROBE_COHORT_BUILD_RUN_SHA256}" \
  --gpu-id 0 \
  --output-dir /absolute/path/to/mistral-probe-transition-evidence
```

この検証済みbundleを `select-probe-transition` へ渡した後に初めてGPU forwardを実行します。cohort構築時に
モデル挙動を見ないため、Linear Probeの境界選択をtask accuracyやadapter結果へ適合させる余地は
ありません。
