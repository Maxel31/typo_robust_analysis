# Linear Probe境界と下流horizonを使うstate-lossなし出力蒸留

## 目的

本実験は、Activation Patchingで修復可能性が高かった層の内部状態を直接
教師状態へ一致させる方式から離れ、Linear ProbeとActivation Patchingの知見を
**出力分布蒸留の配置**にだけ使う。学習損失は教師・生徒のhidden stateを比較せず、
教師モデルのclean入力に対する次token分布と、生徒モデルのtypo入力に対する
次token分布だけを比較する。

Linear Probeが凍結データ上で選んだtransition layerを `b` とする。提案条件では
`b` より前のdecoder blockを固定し、`b..L-1` にだけLoRAを置く。これにより、浅い層で
文字・subword差を無理に消さず、意味復元が立ち上がる境界以降だけを適応させる。

Activation Patchingは学習対象stateを選ぶためには使わない。既存の読み出し区間と
同じ、編集語末から下流offset `+2..+16` のaligned non-edited tokenを、noisy例の
出力教師信号に使う。clean no-op例は全aligned tokenで自己蒸留する。

## 損失

clean入力を受ける凍結Baseを教師 `T`、typo入力を受けるBase+LoRAを生徒 `S` とする。
noisy例 `i` の選択token集合を `H_i`、clean例の全aligned token集合を `A_i` とすると、
各optimizer accumulationの損失は

\[
\mathcal L = \frac{1}{2}\frac{1}{N_c}
\sum_{i\in B_c}\sum_{t\in A_i}
\mathrm{KL}(p_T^i(t)\|p_S^i(t))
+\frac{1}{2}\frac{1}{N_n}
\sum_{i\in B_n}\sum_{t\in H_i}
\mathrm{KL}(p_T^i(t)\|p_S^i(t)),
\]

\[
H_i=\{t:\;t=\pi(t_e)+\delta,\;\delta\in\{2,\ldots,16\},
\;t\text{ is aligned and non-edited}\}.
\]

ここで `N_c` と `N_n` はclean/noisy側で選ばれたtoken数である。clean文書とnoisy文書は
厳密に1:1で交互に入れ、両群の損失質量を常に0.5ずつにする。たとえばclean側511 token、
noisy側15 tokenでも、noisy教師信号がtoken数差で約3%へ薄まることはない。0.5は1:1混合から
一意に決まる固定値であり、調整する手法ハイパーパラメータではない。

## 5条件factorial

| 条件 | LoRA配置 | noisy出力教師token | 役割 |
|---|---|---|---|
| `factorial-all-layers-all-tokens` | 全decoder層 | 全aligned non-edited | matched output-KD基準 |
| `factorial-all-layers-downstream-horizon` | 全decoder層 | `+2..+16` | targetingのみ |
| `factorial-probe-suffix-all-tokens` | Probe境界以降 | 全aligned non-edited | placementのみ |
| `factorial-probe-suffix-downstream-horizon` | Probe境界以降 | `+2..+16` | 完全提案条件 |
| `factorial-random-layers-downstream-horizon` | 同数の固定random層 | `+2..+16` | 場所の情報量の対照 |

random層集合はdecoder層数と `b` からSHA-256 seed 42で一度だけ決め、学習seedごとに
引き直さない。全5条件は同じデータ順、同じtypo実現値、同じrank/alpha、同じlayer-keyed
LoRA初期値を共有する。したがって主比較は、LoRA配置と教師token範囲の2軸に帰属できる。

## 実行前のmaterialize

Probe artifactを検証し、5条件のconfigを同時生成する。

```bash
uv run --project projects/typo-robust-training --locked \
  typo-cot materialize-probe-output-factorial-configs \
  --template projects/typo-robust-training/configs/proposals/gemma4b-probe-output-factorial-10m.template.yaml \
  --probe-selection "${PROBE_SELECTION}" \
  --output-dir "${FACTORIAL_CONFIG_DIR}"
```

生成された各configは同じProbe artifact SHA-256を保持する。学習時は条件に対応する
`train-factorial-*` command、生成config、同じ `--probe-selection`、同じtraining dataを
渡し、さらに凍結済み `training-preregistered` phaseを指す単一の
`--evaluation-v2-registry-bundle` を必須指定する。未検証のProbe artifact、条件と異なるconfig、
registryと異なるdata tree、群欠落・順序不正・partial accumulation、
noisy horizonが空になるpairはfail closedとする。

## 主張の境界

完全提案条件がmatched基準だけでなく、targeting-only、placement-only、random-locationを
上回って初めて、Probe境界と下流horizonの組合せに固有の価値を主張する。別途、Mistral上で
公開Kojima手順を忠実に再現した条件とも比較する。state距離やpatch gainは診断に限り、
頑健性の合否には使わない。
