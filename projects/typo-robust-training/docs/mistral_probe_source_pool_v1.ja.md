# Mistral Linear Probe source pool v1

このfreezerは、Linear Probeのclass・境界選択より前に、未使用の
FineWeb-Edu clean文書をモデル出力非依存で固定する。対象は以下の1 shardだけである。

- dataset: `HuggingFaceFW/fineweb-edu`
- revision: `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`
- subset: `sample-10BT`
- shard: `sample/10BT/013_00000.parquet`
- SHA-256: `b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de`

選択にmodel forward、task accuracy、typo結果は使わない。各FineWeb文書のstable ID、
parent ID、全文normalized-content hashを、training / localization / tune / pre-pr /
sealedの5 tierを束ねた保護レジストリと照合し、いずれかに触れる文書を除外する。
pool内の全文normalized-content重複はparquet順で最初の1件だけを残す。

`token_count`は共有JSONL schemaの必須列なので、FineWeb-Eduが公開した値をそのまま
保持する。ただしこれはProbeのclass選択、token inflation層化、境界選択には使わない。
clean/typo token inflationは後段のexact Mistral tokenizer attestationで再計算する。

実行前に、保護レジストリとcleanなコードcommitを外部で固定する。

```bash
PARQUET=/absolute/path/to/sample/10BT/013_00000.parquet
PROTECTED=/absolute/path/to/protected-split-registry.json
CODE_REVISION=$(git rev-parse HEAD)

typo-cot freeze-probe-source-pool \
  --source-parquet "${PARQUET}" \
  --source-parquet-sha256 b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de \
  --protected-registry "${PROTECTED}" \
  --protected-registry-sha256 "$(sha256sum "${PROTECTED}" | cut -d' ' -f1)" \
  --code-revision "${CODE_REVISION}" \
  --output-dir /absolute/new/path/mistral-probe-source-pool-v1
```

成果物は次の3ファイルである。

- `probe_source_pool.jsonl`: `robustness-clean-record/v1` のclean source pool
- `protected_split_registry.json`: 検査に使ったregistryのbyte-identical copy
- `freeze_probe_source_pool_run.json`: source shard、コードtree、除外規則、件数、成果物hashを
  束ねるproducer record

producer recordの`record_sha256`はself-checkにすぎない。後段の実行では表示されたdigestを
registry/preregistrationへ別途保存し、外部値として渡す。既存output、symlink input、hash不一致、
protected overlap、隣接成果物の書換えはfail closedとなる。
