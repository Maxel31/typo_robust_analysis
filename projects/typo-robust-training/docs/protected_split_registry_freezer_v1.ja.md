# Protected split registry freezer v1

このコマンドは、学習・localization・tune・pre-PR・sealed の5階層に属する既存JSONLを、Mistral linear-probe cohort builderが読む
`typo-protected-split-registry/v1`へ変換します。モデルやGPUは使いません。

## 入力をローカルだけに置く

2026-08-22 snapshotそのもの、sealed本文、絶対パスはGitへ追加しません。専用のローカルdirectoryを作り、templateだけをコピーしてください。

```bash
SNAPSHOT_ROOT=/local-only/typo-evaluation-snapshot-2026-08-22
cp configs/proposals/protected-split-registry.inventory.template.json \
  "$SNAPSHOT_ROOT/protected-split-inventory.json"
```

`tiers`は `training`, `localization`, `tune`, `pre-pr`, `sealed` の順で正確に5個必要です。各tierの`inputs`には、snapshot rootからの相対regular-file path、外部で計算したSHA-256、1個以上の`accepted_schemas`、JSONL行の`split`と一致するroleを列挙します。schema配列は辞書順・重複なしで固定し、単一schema入力も長さ1の配列にします。evaluation corpus manifestのようにclean corpus行とnatural pair行を同じJSONLへ格納する実データでは、両schemaを明示的に列挙します。宣言していないschema行は拒否されます。globやdirectory discoveryはありません。1 tierに複数manifestがある場合も、すべてを個別に列挙します。

対応schemaは次の4つです。

- `robustness-clean-record/v1`
- `robustness-fixed-typo-pair/v1`
- `robustness-natural-pair/v1`
- `robustness-evaluation-corpus-record/v1`

JSONLはUTF-8、LF改行、末尾LF必須です。duplicate JSON key、blank line、CRLF、path traversal、symlink、hardlinkを拒否します。既存のraw/normalized hashも本文全体から再計算します。pairではcleanとtypoの両方を登録するため、typo本文が別tierのclean本文と一致する漏洩も検出します。

## freeze

まず、入力JSONLのSHA-256をinventoryへ書き、そのcanonical JSON自体のSHA-256を別に記録します。

```bash
INVENTORY="$SNAPSHOT_ROOT/protected-split-inventory.json"
INVENTORY_SHA256=$(sha256sum "$INVENTORY" | awk '{print $1}')

uv run --project projects/typo-robust-training \
  typo-cot freeze-protected-split-registry \
  --inventory "$INVENTORY" \
  --inventory-sha256 "$INVENTORY_SHA256" \
  --output-dir "$SNAPSHOT_ROOT/protected-split-registry-v1"
```

出力はclosed-world bundleです。

- `registry.json`: downstream consumer用の既存schema
- `inventory.json`: externally pinned inventoryのbyte-identical copy
- `inputs/*.jsonl`: 検証に使った入力のbyte-identical copy
- `freeze_protected_split_registry_run.json`: checkout tree、全入力、件数、出力を束縛するproducer record

コマンドが表示するproducer-record SHA-256を、bundleの外側へ保存してください。run fileや隣接hashをすべて書き換えても、この外部値は再生成できないため、自己rehash偽造を拒否できます。publish直前に元inventoryと全入力を再hashし、完全なstaging bundleを検証してから、Linux `renameat2(RENAME_NOREPLACE)`でsibling directoryをatomicかつno-clobber publishします。このprimitiveが利用できない環境では安全性を弱めずfail closedします。

## verification only

```bash
uv run --project projects/typo-robust-training \
  typo-cot verify-protected-split-registry \
  --producer-run \
    "$SNAPSHOT_ROOT/protected-split-registry-v1/freeze_protected_split_registry_run.json" \
  --producer-record-sha256 "$EXTERNALLY_PINNED_PRODUCER_RECORD_SHA256"
```

verificationはcopied inventoryと全JSONLを再解析し、3種類のidentityを同じnamespace関数で再生成し、union-findによるtransitive overlap検査を再実行します。`registry.json`だけを信用しません。

Python consumerは`load_protected_split_registry_bundle(...)`が返すbundleの`identity_sets`を使用します。`source_group_sha256`、`parent_source_sha256`、`normalized_content_sha256`はいずれも、検証済み5 tierのimmutable `frozenset` unionです。大規模source-poolはregistryを独自再parseせず、この集合へstreamingで照合します。

## overlapの意味

各recordについて、source group、parent source、clean/typoのnormalized full-text identityを同じconnected componentへ結合します。componentが2 tier以上に属した場合は失敗します。このため、trainingとlocalizationがgroupを共有し、localizationとsealedが別のparent identityを共有するような3-tier bridgeも検出できます。同じtier内の完全に同一なrecordはdeduplicateできますが、同じsource identityで本文やmetadataが異なるrecordはconflictとして拒否します。
