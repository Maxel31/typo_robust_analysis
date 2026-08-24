# Mistral Linear Probe source pool v1

このfreezerは、Linear Probeのclass・境界選択より前に、未使用の
FineWeb-Edu clean文書をモデル出力非依存で固定する。対象は以下の1 shardだけである。

- dataset: `HuggingFaceFW/fineweb-edu`
- revision: `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`
- subset: `sample-10BT`
- shard: `sample/10BT/013_00000.parquet`
- SHA-256: `b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de`
- bytes: `540632672`

選択にmodel forward、task accuracy、typo結果は使わない。各FineWeb文書のstable ID、
parent ID、全文normalized-content hashを、歴史的protected input全体を束ねたexclusion-only
denylistと照合し、いずれかに触れる文書を除外する。
pool内の全文normalized-content重複はparquet順で最初の1件だけを残す。

`token_count`は共有JSONL schemaの必須列なので、FineWeb-Eduが公開した値をそのまま
保持する。ただしこれはProbeのclass選択、token inflation層化、境界選択には使わない。
clean/typo token inflationは後段のexact Mistral tokenizer attestationで再計算する。

実行前に、単体registry JSONやstrict split registry bundleではなく、public loaderで全inputを
replayできる`typo-protected-exclusion-denylist/v1` bundleと、そのproducer-record SHA-256を
外部で固定する。このartifactは`purpose=source-pool-exclusion-only`かつ
`split_certified=false`であり、歴史的cross-tier overlapを安全なhash audit付きunionとして
扱う。raw registryを直接読むprivate parserやstrict registryへのfallbackはない。cleanな
コードcommitも外部で固定する。

```bash
PARQUET=/absolute/path/to/sample/10BT/013_00000.parquet
PROTECTED_RUN=/absolute/path/to/protected-denylist/freeze_protected_exclusion_denylist_run.json
PROTECTED_PRODUCER_SHA256=<denylist-freeze時に別媒体へ記録したproducer-record-sha256>
CODE_REVISION=$(git rev-parse HEAD)

typo-cot freeze-probe-source-pool \
  --source-parquet "${PARQUET}" \
  --source-parquet-sha256 b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de \
  --protected-exclusion-run "${PROTECTED_RUN}" \
  --protected-exclusion-producer-sha256 "${PROTECTED_PRODUCER_SHA256}" \
  --code-revision "${CODE_REVISION}" \
  --output-dir /absolute/new/path/mistral-probe-source-pool-v1
```

output directory全体を1回の`RENAME_NOREPLACE`で公開する。rename直前に、pinned Parquet
descriptor、live checkout、copied denylistのpublic typed replay、生成済みproducer record SHAと
code revisionを使ったstaging bundleのoffline replayを再実行する。最後にprotected/staging treeと
directory inodeを再照合し、そのstaging inodeだけをno-replace publishする。成果物は次の通りである。

- `probe_source_pool.jsonl`: `robustness-clean-record/v1` のclean source pool
- `probe_source_pool_decisions.jsonl`: parquet全行の検証済みsource record、3 identity、
  `protected -> duplicate -> retained`判定を含むreplay ledger
- `protected_exclusion/`: inventory、全copied JSONL、denylist、本文を含まないoverlap audit、
  producer recordを含む、検査に使ったclosed-world bundleのcopy
- `freeze_probe_source_pool_run.json`: source shard、コードtree、除外規則、件数、成果物hashを
  束ねるproducer record

producer recordの`record_sha256`はself-checkにすぎない。後段の実行では表示されたdigestを
registry/preregistrationへ別途保存し、外部値として渡す。loaderはdecision ledgerをstreamし、
全source identity、full-text hash、protected照合、normalized dedup、優先順位、件数をdisk-backed
ledgerで再計算する。従って、run/artifact/countをまとめてself-rehashしただけでは改竄を正当化
できない。既存output、ancestorを含むsymlink、hardlink、hash不一致、path/inodeのTOCTOU、
protected overlap、closed-world外のnode、隣接成果物の書換えはfail closedとなる。

Parquet schemaは列名だけでなく、公式shardの型
`string,string,string,string,string,string,double,int64,double,int64`と全列nullableをexactに
照合する。upstream `token_count`はledger/source metadataに保持するだけで、retained/duplicate/
protected判定には入力しない。
