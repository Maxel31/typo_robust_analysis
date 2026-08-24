# Protected exclusion denylist v1

## 目的と禁止事項

`typo-protected-exclusion-denylist/v1`は、過去に保護対象だったidentityを新しいsource poolやevaluation v2 candidateから除外するためだけのartifactです。過去tier間に重複があっても全identityのunionを保存しますが、`purpose=source-pool-exclusion-only`、`split_certified=false`をartifact、producer record、typed bundleの全境界で固定します。

このdenylistを、training/localization/tune/pre-PR/sealedの非重複証明、probe cohortの認証、または評価splitのcertificationとして使用してはいけません。新しいprimary splitはstrict protected split registryで別途freezeしなければなりません。

## Freeze

```bash
uv run --project projects/typo-robust-training \
  typo-cot freeze-protected-exclusion-denylist \
  --inventory "$HISTORICAL_INVENTORY" \
  --inventory-sha256 "$EXTERNALLY_PINNED_INVENTORY_SHA256" \
  --output-dir "$DENYLIST_BUNDLE"
```

成功時に出力されるproducer-record SHA-256をbundle外へ記録します。bundleは以下だけを含むclosed path setです。

- externally pinned inventoryのcopy
- inventoryに列挙された全JSONLのcopy
- `denylist.json`
- `overlap_audit.json`
- `freeze_protected_exclusion_denylist_run.json`

`overlap_audit.json`はtier、identity kindとSHA-256、source relative path、line、record IDだけを含みます。clean/typo本文、回答、metadataは含みません。

## Verifyとconsumer contract

```bash
uv run --project projects/typo-robust-training \
  typo-cot verify-protected-exclusion-denylist \
  --producer-run \
    "$DENYLIST_BUNDLE/freeze_protected_exclusion_denylist_run.json" \
  --producer-record-sha256 "$EXTERNALLY_PINNED_PRODUCER_RECORD_SHA256"
```

Python consumerは`load_protected_exclusion_denylist_bundle(...)`だけをtrust boundaryとして使います。返り値`ProtectedExclusionDenylistBundle`の`identity_sets`は、copied inventoryと全入力を再解析して得た次のimmutable `frozenset` unionです。

- `source_group_sha256`
- `parent_source_sha256`
- `normalized_content_sha256`

loaderはdenylistとoverlap auditを再計算し、producer record、外部hash、code provenance、全artifact hash、closed file set、symlink/hardlink/TOCTOU条件を検証します。raw `denylist.json`の直接parse、strict registry型へのcast、`split_certified`の上書きは禁止です。
