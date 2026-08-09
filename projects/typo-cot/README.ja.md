# typo-cot 再現パッケージ

[English](README.md) | [日本語](README.ja.md)

このパッケージは、論文 **“Edited-Word Activation Patching Reverses Selected
Typo-Induced Answer Changes after Tokenization.”** の公開再現インターフェースです。

このガイドのセットアップと操作別コマンドを使って、論文の実験を再現できます。
実験プロトコルと報告値は最終論文に従います。手元のPDFを確認したい場合は、
カタログからSHA-256を表示できます。

```bash
uv run --project projects/typo-cot typo-cot experiments source
```

実験の対応表、分母、出力ディレクトリ構成、1実験1コマンドの契約は
[`docs/paper-experiments.md`](docs/paper-experiments.md) にあります。

## セットアップ

リポジトリのルートで実行します。

```bash
uv sync --project projects/typo-cot
```

ペア生成とactivation patchingには、論文に固定されたGPU/LRP依存も必要です。

```bash
uv sync --project projects/typo-cot --extra lrp
```

## 利用できるコマンド

実験カタログの確認にはGPUは不要です。

```bash
uv run --project projects/typo-cot typo-cot experiments list
uv run --project projects/typo-cot typo-cot experiments list --format json
uv run --project projects/typo-cot typo-cot experiments show cot-swap
uv run --project projects/typo-cot typo-cot experiments show clean-prefix-scan --format json
```

各エントリには、安定した `target_command`、論文の節、実験固有の必須引数、
コホート、介入、readout、出力、CPU/GPU区分、実装状態が含まれます。
`implemented` の操作だけが実行可能で、`catalogued` は公開runnerが未実装です。

## clean/editedペアを生成する

`prepare-edited-pairs` は論文の入力準備を行います。clean入力でのgreedy生成、
最初のCoT tokenへのAttnLRPによるtargeting、seedで固定した最大4個の1文字編集、
edited入力でのgreedy生成、決定的な回答抽出、編集語末tokenのalignmentを順に実行します。
model・benchmark・targeting条件ごとに個別に実行してください。

```bash
uv run --project projects/typo-cot --extra lrp typo-cot prepare-edited-pairs \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --targeting attribution-4 \
  --num-edits 4 \
  --output-dir results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4
```

同一item内の対照条件には `--targeting random-4` を使います。
`--num-edits` は1〜4、論文既定値は `--seed 42` と
`--max-new-tokens 512` です。GPU smoke testには `--limit 1`、中断再開には
`--resume` を使えます。clean/editedの両生成でsamplingを無効化し、answer extractorは
primary、空の場合だけ決定的fallbackの順で適用します。

出力は次の2ファイルです。

- `pairs.jsonl`: clean/edited生成、target試行、実際の編集語span、clean/edited語末
  token indexを含むitem単位レコード。
- `run.json`: 固定引数、環境・dataset provenance、進捗、失敗、件数。

選択されたdataset itemは、適用可能な文字編集がない場合も `pairs.jsonl` に残ります。
その場合、`target_attempts` と `aligned_words` は空で、後続のactivation patch対象には
なりませんが、targeting fidelityの母集団には含まれます。MMLUの件数はmodelごとに
論文設定を使い、MMLU-Proは全modelでsubjectごとに100例を使います。詳細なfieldと
座標規約は [`docs/prepare-edited-pairs.md`](docs/prepare-edited-pairs.md) を参照してください。

## targeting fidelityを監査する

4編集のmodel・benchmark・targeting各cellを準備した後、Appendix Aの入力品質確認を
CPUだけで集計します。

```bash
uv run --project projects/typo-cot typo-cot targeting-fidelity-audit \
  --pairs-root results/prepare-edited-pairs \
  --output-dir results/targeting-fidelity-audit
```

`targeting-fidelity-audit` は `--pairs-root` 以下から完了済みの
`prepare-edited-pairs/v1` を再帰的に探し、path名ではなくrecordとmanifestから設定を
読み取ります。partial、重複、混在、4編集以外の入力は拒否します。記録済みflagを
信頼せず、landing offsetとSHA-seed文字編集を再計算します。

出力は以下です。

- `targeting_fidelity_records.jsonl`: itemごとの検証・監査行。
- `targeting_fidelity.csv`: 設定別／pooledのlanding、distinct-word、zero-attempt、
  gold-optionなどの集計。
- `operation_counts.json`: substitution・duplication・deletionの件数。
- `run.json`: 入出力hash、引数、論文fingerprint、件数、比較用参照値。

item分母とedit-attempt分母を混同せず、すべての比較可能性条件を満たした場合だけ
`descriptive_only` として論文値と比較します。詳細は
[`docs/targeting-fidelity-audit.md`](docs/targeting-fidelity-audit.md) を参照してください。

## layerごとに最初のCoT tokenのKL patchを走査する

`layerwise-kl-patching` は、1つのmodel・benchmark・targeting条件について、論文の
distributional RQ1走査を実行します。保存済みのclean正答・edited誤答pairを選び、
alignedされた編集語末の全位置で、decoder blockのresidual出力をlayerごとに転送します。
両方向を全layerで実行します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-kl-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4
```

readoutは最初のCoT tokenを予測するprompt末のnext-token分布です。
`clean-to-edited` の正規化scoreは
`1 - KL(clean || patched-edited) / KL(clean || edited)` で、逆方向ではcleanとeditedを
すべて交換します。未処置分母はfiniteかつ `1e-9` より大きい必要があります。
全layerがfiniteなpair/directionだけをsummaryに入れ、負のscoreも保持します。

出力は `layer_records.jsonl`、`pair_status_records.jsonl`、
`setting_summary.json`、`run.json` です。`--limit 1` はpartialなsmoke run、
`--resume` は同一条件の中断再開です。paper headlineの30設定macro averageは別の
cross-setting artifact工程です。完全な契約は
[`docs/layerwise-kl-patching.md`](docs/layerwise-kl-patching.md) にあります。

## layerごとに自由回答patchを走査する

`layerwise-answer-patching` はFigure 2右側のfree-generation走査です。2つのtargeting arm
からseed 42で各最大150 pair、合計最大300 anchorを選び、patch処理内で未処置回答を
再生成します。clean正答・edited誤答のままのpairを全layer・両方向共通の固定分母にします。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot layerwise-answer-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --attribution-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --random-pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --directions clean-to-edited edited-to-clean \
  --max-pairs 300 \
  --gpu-id 0 \
  --output-dir results/layerwise-answer-patching/gemma-3-4b-it/gsm8k
```

各layerは最大512 tokenの独立したgreedy continuationです。patchはprompt prefillだけに
適用し、以降はpatched KV cacheを使います。`clean-to-edited` はpatched edited回答が
再生成clean回答へ戻れば成功、`edited-to-clean` はpatched clean回答が元のclean回答から
変化すれば成功です。抽出不能は分母から除かず失敗として数えます。

出力は `answer_layer_records.jsonl`、`pair_status_records.jsonl`、
`setting_summary.json`、`run.json` です。論文runは `--max-pairs 300`、小さい値は
partial smokeです。詳細は
[`docs/layerwise-answer-patching.md`](docs/layerwise-answer-patching.md) にあります。

## 固定layer windowをpatchして回答を再生成する

`fixed-window-answer-patching` はTable 6の介入を実行します。各targeting armから最大150の
anchorをpoolし、未処置回答を再生成した後、prompt prefill中に `[0,6)` の6 decoder
block出力をまとめて転送します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot fixed-window-answer-patching \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/random-4/pairs.jsonl \
  --layers 0:6 \
  --directions clean-to-edited edited-to-clean \
  --gpu-id 0 \
  --output-dir results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k
```

Table 7のMMLU-Pro depth比較では、同じanchorに2つの独立した6-layer windowを使います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot fixed-window-answer-patching \
  --model Qwen/Qwen2.5-3B-Instruct \
  --benchmark mmlu-pro \
  --pairs results/prepare-edited-pairs/Qwen2.5-3B-Instruct/mmlu-pro/attribution-4/pairs.jsonl \
  --layers 0:6 6:12 \
  --directions clean-to-edited \
  --gpu-id 0 \
  --output-dir results/fixed-window-answer-patching/Qwen2.5-3B-Instruct/mmlu-pro
```

2方向の固定分母は意図的に異なります。restorationはclean正答・edited誤答pair、
reciprocal inductionは再生成clean正答の全anchorを使います。抽出不能回答はどちらも
失敗として分母に残します。`--pairs` は1または2 arm、`--layers` は重ならない半開区間を
1個以上受け取ります。`--limit 1` はsmoke、`--resume` は中断再開です。

出力は `fixed_window_records.jsonl`、`pair_status_records.jsonl`、
`setting_summary.json`、`run.json` です。fresh runのlabelは論文と同じprotocol形状を
実行したことを示し、非公開のhistorical sample IDを復元したという意味ではありません。
詳細は [`docs/fixed-window-answer-patching.md`](docs/fixed-window-answer-patching.md) にあります。

## patch座標とdonor内容を比較する

`patch-coordinate-controls` はGemma-3-4B/GSM8KについてTable 7の主座標controlを再現します。
完了済みfixed-window runの同一分母と正しい座標endpointを保ち、同一itemの+2 token座標、
matchingした別item donor、identity self-copyを比較します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-coordinate-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --controls correct offset-2 cross-item self-copy \
  --gpu-id 0 \
  --output-dir results/patch-coordinate-controls/gemma-3-4b-it/gsm8k
```

offset armはdonor/write座標を編集語末から2 token進めます。cross-item donorは同じtargeting
arm・aligned-word数の異なるitemを、sample ID順の循環shiftで決定します。offsetと
cross-itemは論文どおりpost-hoc controlです。identity self-copyは未処置edited baselineと
token列が完全一致しなければ失敗します。

出力は `coordinate_control_records.jsonl`、`pair_status_records.jsonl`、
`coordinate_control_summary.json`、`run.json` です。summaryにはarm別rateと、correct対
offset/cross-itemの両側exact McNemar比較が入ります。詳細は
[`docs/patch-coordinate-controls.md`](docs/patch-coordinate-controls.md) にあります。

## patchの書き込み位置を比較する

`patch-position-controls` は同じ完全layer grid上で、編集語末、prompt末、question末への
patchを比較します。source runのpair/direction/layer分母を固定し、位置だけを変えます。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-position-controls \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --layerwise-kl-run \
    results/layerwise-kl-patching/gemma-3-4b-it/gsm8k/attribution-4 \
  --positions edited-word prompt-final question-final \
  --gpu-id 0 \
  --output-dir results/patch-position-controls/gemma-3-4b-it/gsm8k
```

出力は `position_control_records.jsonl`、`pair_status_records.jsonl`、
`position_control_summary.json`、`run.json` です。source manifest・record・hashをmodel
load前に検証し、全positionで共通の完全分母を使います。詳細は
[`docs/patch-position-controls.md`](docs/patch-position-controls.md) にあります。

## 固定patchと完全なclean textを組み合わせる

`patch-text-combination` は、固定 `[0,6)` patchの有無と、完全なclean pre-answer textの
有無を2×2で組み合わせる記述的比較です。4 cellすべてで、参照するGemma-3-4B/GSM8K
fixed-window runのhash検証済み分母を使います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot patch-text-combination \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --layers 0:6 \
  --gpu-id 0 \
  --output-dir results/patch-text-combination/gemma-3-4b-it/gsm8k
```

出力は `patch_text_records.jsonl`、`pair_status_records.jsonl`、
`patch_text_summary.json`、`run.json` です。この操作は4つの回答正答率を報告しますが、
mediationやinteractionの推定値としては扱いません。完全な契約は
[`docs/patch-text-combination.md`](docs/patch-text-combination.md) にあります。

## clean/edited questionと完全CoTを交差させる

`cot-swap` はTable 1とAppendix Cの4 cell、A=(clean question, clean CoT)、
B=(edited, edited)、C=(edited, clean)、D=(clean, edited)を実行します。selected pre-answer
CoTをteacher-forceし、answer spanだけを自由生成します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot cot-swap \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --pairs results/prepare-edited-pairs/gemma-3-4b-it/gsm8k/attribution-4/pairs.jsonl \
  --targeting attribution-4 \
  --gpu-id 0 \
  --output-dir results/cot-swap/gemma-3-4b-it/gsm8k/attribution-4
```

sourceは適用済み編集とanswer templateを持つpairです。change rateは再生成Aが正答の
pairを分母とし、restorationはさらにBがAから変わったpairに条件づけます。回答比較には
task固有のcanonical equalityを使います。

出力は `cot_swap_records.jsonl`、`pair_status_records.jsonl`、
`cot_swap_summary.json`、`run.json` です。summaryはBoth changed、Question only、
CoT only、B→C restorationを、それぞれの整数分子・分母から計算します。
詳細は [`docs/cot-swap.md`](docs/cot-swap.md) にあります。

## clean pre-answer textの最終行を削除する

`answer-line-deletion` は、Random-4の完了済みCoT-swap runを入力として、clean CoTの
最後の非空行を除いたC cellを再実行します。near-answer内容・formatへの依存を調べる
RQ2 controlです。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot answer-line-deletion \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cot-swap-run results/cot-swap/gemma-3-4b-it/gsm8k/random-4 \
  --max-pairs 150 \
  --gpu-id 0 \
  --output-dir results/answer-line-deletion/gemma-3-4b-it/gsm8k
```

分母はsource runでAが正答かつBがAから変化したcaseです。同じpairに対しcompleteと
deletedのrestorationを比較し、抽出不能は失敗として残します。出力は
`answer_line_deletion_records.jsonl`、`pair_status_records.jsonl`、
`answer_line_deletion_summary.json`、`run.json` です。詳細は
[`docs/answer-line-deletion.md`](docs/answer-line-deletion.md) にあります。

## clean pre-answer token prefixを走査する

`clean-prefix-scan` はRQ3のFigure 3を実行します。edited questionの後へclean CoTの先頭
`k` tokenを与え、固定したrelative/absolute budget gridごとに自由生成します。

primary Gemma-3-4B/GSM8K cellは、固定172-pair sourceを使います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot clean-prefix-scan \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-4b-it/gsm8k
```

14個のextension cellは、完了済みAttribution-4/Random-4 pairから設定ごとに決定的に
最大150 targetを選びます。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot clean-prefix-scan \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --relative-budgets 0 .02 .05 .08 .12 .16 .20 .25 .325 .40 .50 .65 .80 1 \
  --absolute-budgets 1 2 4 8 16 32 64 \
  --gpu-id 0 \
  --output-dir results/clean-prefix-scan/gemma-3-1b-it/gsm8k
```

relative budgetでは `k = round(r * L_C)` を使います。scan-timeのfresh `k=0` が誤答の
valid targetを共通分母とし、各pointの正答、以後すべて正答のstable recovery、2回以上
遷移するnon-monotonicityを報告します。token IDをdecode・再tokenizeせず、そのままmodelへ
渡します。抽出不能は誤答として分母に残します。

各設定の出力は `prefix_scan_records.jsonl`、`pair_status_records.jsonl`、
`prefix_scan_summary.json`、`run.json` です。`--limit 1` はsmoke、`--resume` は
hash-bound checkpointからの再開です。14設定のcluster bootstrapは後続のCPU artifact
工程で行い、primary cellはextension aggregateへ混ぜません。詳細は
[`docs/clean-prefix-scan.md`](docs/clean-prefix-scan.md) にあります。

## clean pre-answer tokenを1個置換して再生成する

`one-token-prefix-replacement` はAppendix DとTables 10--11の補足診断です。clean question
の下で、選択位置より前のclean pre-answer token IDを与え、その位置で観測clean token
またはtypo-context top-1 tokenを強制してから回答を自由生成します。これはclean question
下のanswer sensitivityであり、typo修復やRQ3の主回答ではありません。

primary cellはprimary clean-prefix scanと同じ172-pair sourceを使います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-4b-it \
  --benchmark gsm8k \
  --cohort primary \
  --fixed-window-run \
    results/fixed-window-answer-patching/gemma-3-4b-it/gsm8k \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-4b-it/gsm8k
```

14 extensionの各設定では `clean-prefix-scan` と同じ決定的な150-target selectionを使います。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark mmlu \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/mmlu/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/mmlu
```

事前指定されたadjacent-position checkはGemma-3-1B/GSM8K、Llama-3.2-3B/ARC、
Mistral-7B/MMLUだけです。該当extension runでは `adjacent` を追加し、同じhash-bound
record setから両paper tableを作ります。

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
  typo-cot one-token-prefix-replacement \
  --model google/gemma-3-1b-it \
  --benchmark gsm8k \
  --cohort extension \
  --pairs \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/attribution-4/pairs.jsonl \
    results/prepare-edited-pairs/gemma-3-1b-it/gsm8k/random-4/pairs.jsonl \
  --max-pairs 150 \
  --position-controls distant adjacent \
  --gpu-id 0 \
  --output-dir results/one-token-prefix-replacement/gemma-3-1b-it/gsm8k
```

候補位置 `P` はclean対editedのnext-token KLを最大化し、clean tokenがedited-question
contextのtop-1でない位置です。distant control `C` は `P` から3 token以上離れた
lower-median-KL候補です。`distant` はTable 10のlocal substitutionとTable 11の4 crossingを
生成し、`adjacent` は最寄りのlower-KL位置にも `P` 由来tokenを適用します。位置の選択に
生成回答の結果は使いません。

出力は `one_token_records.jsonl`、`pair_status_records.jsonl`、
`one_token_summary.json`、`run.json` です。correctnessはgold answerとのcanonical比較、
抽出不能は誤答です。primaryは14 extension aggregateへ混ぜません。詳細は
[`docs/one-token-prefix-replacement.md`](docs/one-token-prefix-replacement.md) にあります。

## one-token論文表を構築する

15個の `one-token-prefix-replacement` 設定が完了した後、Appendix Dの表とFigure 5の
再現可能部分をCPUで構築します。

```bash
uv run --project projects/typo-cot \
  typo-cot build-one-token-tables \
  --runs-root results/one-token-prefix-replacement \
  --output-dir results/one-token-tables
```

`build-one-token-tables` は `--runs-root` 以下のproducer `run.json` を再帰的に検出し、
論文／protocol identityと出力checksumを検証した上で、JSONLの整数eventからrateを
再計算します。model、tokenizer、`torch`、`transformers` はimportしません。partial、
予期しない設定、重複、identity混在、改ざん済みrunはfail closedです。必要なgridが
欠ける場合はcoverageへ明記し、paper label付きpooled estimateを出しません。

既存の出力directoryは上書きせず、次のartifactをatomicに公開します。

- `one_token_tables.json`: cell、pool、推論metadata、historical reference、比較判断。
- `table10_one_token.csv`: Table 10の設定別token列と、完全な場合の14-extension集計。
- `table11_position_controls.csv`: distant/adjacent position control。
- `one_token_tables.md` と `one_token_tables.tex`: 読みやすい決定的table fragment。
- `figure5_validation.json`: producer recordにあるFigure 5 fieldの検証。
- `run.json`: 入出力hashと固定analysis protocol。

primary Gemma-3-4B/GSM8Kは常に別行で、14 extensionへpoolしません。adjacent controlは
事前指定3設定だけを使います。詳細は
[`docs/build-one-token-tables.md`](docs/build-one-token-tables.md) にあります。

## 編集数に対する感度を測定する

`edit-count-sensitivity` はfreshかつ検証済みの設定別artifactからAppendix C/Table 8を
構築します。accuracyとCoT-swapはgridも分母も異なるため、GPU producerをCPU集約の
内部に隠さず、別々に実行します。

最初に、51設定のaccuracy gridについて、Attribution-4の
`prepare-edited-pairs` を編集数1・2・4で個別に実行します。

```bash
ACCURACY_FULL_MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  google/gemma-3-12b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
)
ACCURACY_BENCHMARKS=(arc csqa gsm8k math-500 mmlu mmlu-pro)

prepare_edit_count_setting() {
  local MODEL="$1"
  local BENCHMARK="$2"
  local MODEL_SLUG="${MODEL##*/}"
  for EDIT_COUNT in 1 2 4; do
    CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
      typo-cot prepare-edited-pairs \
      --model "${MODEL}" \
      --benchmark "${BENCHMARK}" \
      --targeting attribution-4 \
      --num-edits "${EDIT_COUNT}" \
      --gpu-id 0 \
      --output-dir \
        "results/edit-count-pairs/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}"
  done
}

for MODEL in "${ACCURACY_FULL_MODELS[@]}"; do
  for BENCHMARK in "${ACCURACY_BENCHMARKS[@]}"; do
    prepare_edit_count_setting "${MODEL}" "${BENCHMARK}"
  done
done
for BENCHMARK in gsm8k mmlu mmlu-pro; do
  prepare_edit_count_setting Qwen/Qwen2.5-3B-Instruct "${BENCHMARK}"
done
```

restorationの6設定（Gemma-3-4B、Llama-3.2-3B、Mistral-7B × GSM8K、MMLU）では、
各edit-count sourceから同じ4 cell CoT swapを実行します。main実験の既定値は
`--source-num-edits 4` ですが、Table 8では明示した値でsensitivity protocolを識別します。

```bash
RESTORATION_MODELS=(
  google/gemma-3-4b-it
  meta-llama/Llama-3.2-3B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
)
for MODEL in "${RESTORATION_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  for BENCHMARK in gsm8k mmlu; do
    for EDIT_COUNT in 1 2 4; do
      CUDA_VISIBLE_DEVICES=0 uv run --project projects/typo-cot --extra lrp \
        typo-cot cot-swap \
        --model "${MODEL}" \
        --benchmark "${BENCHMARK}" \
        --pairs \
          "results/edit-count-pairs/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}/pairs.jsonl" \
        --targeting attribution-4 \
        --source-num-edits "${EDIT_COUNT}" \
        --gpu-id 0 \
        --output-dir \
          "results/edit-count-cot-swap/${MODEL_SLUG}/${BENCHMARK}/${EDIT_COUNT}"
    done
  done
done
```

全producerの完了後、Table 8をCPUで構築します。

```bash
uv run --project projects/typo-cot \
  typo-cot edit-count-sensitivity \
  --pairs-root results/edit-count-pairs \
  --cot-swap-runs-root results/edit-count-cot-swap \
  --edit-counts 1 2 4 \
  --output-dir results/edit-count-sensitivity
```

builderはproducer manifestを再帰的に検出し、論文fingerprint、protocol、設定identity、
source/output hash、unlimited cohort、record単位の整数eventを検証します。accuracyの
equal-setting行は条件ごとの全分母、matched行はclean・1・2・4編集でsample IDを交差した
分母です。CoT restorationは0編集では未定義で、edit countごとに「再生成A正答かつBが
Aから変化」を別々に条件づけ、3分母を交差しません。

完全gridは正確な51 accuracy設定と18 CoT-swap runです。partialでもvalidなら監査可能ですが、
不足するpaper-pooled比較は出しません。出力は `edit_count_records.jsonl`、
`edit_count_summary.json`、`table8_edit_count.csv`、`table8_edit_count.md`、
`table8_edit_count.tex`、`run.json` です。詳細は
[`docs/edit-count-sensitivity.md`](docs/edit-count-sensitivity.md) にあります。

## model規模をまたいで完全CoT swapを比較する

`model-scale-cot-swap` は、独立して再開可能なMMLU Attribution-4 producer runから
Appendix C/Table 9を構築します。論文の9 modelに対して、seed-42 MMLU loader IDの先頭
500件を含む共通selectorを使います。selectorは
`projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json` にあり、dataset ID、
model別selected-ID set hash、protocol metadataだけを含み、historical model出力は
含みません。

modelごとにpair生成とCoT swapを別々に実行します。以下はphysical GPU 0が既定です。
70B/72B modelのshardingが必要なmachineでは `MODEL_SCALE_GPU_IDS` にcomma区切りのGPUを
指定できますが、設定ごとに独立manifestと再開可能出力を作る点は変わりません。

```bash
MODEL_SCALE_GPU_IDS="${MODEL_SCALE_GPU_IDS:-0}"
MODEL_SCALE_MODELS=(
  google/gemma-3-1b-it
  google/gemma-3-4b-it
  google/gemma-3-12b-it
  google/gemma-3-27b-it
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-70B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  Qwen/Qwen2.5-72B-Instruct
)
MODEL_SCALE_COHORT=projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json

for MODEL in "${MODEL_SCALE_MODELS[@]}"; do
  MODEL_SLUG="${MODEL##*/}"
  CUDA_VISIBLE_DEVICES="${MODEL_SCALE_GPU_IDS}" \
    uv run --project projects/typo-cot --extra lrp \
    typo-cot prepare-edited-pairs \
    --model "${MODEL}" \
    --benchmark mmlu \
    --targeting attribution-4 \
    --num-edits 4 \
    --sample-ids "${MODEL_SCALE_COHORT}" \
    --gpu-id "${MODEL_SCALE_GPU_IDS}" \
    --output-dir "results/model-scale-pairs/${MODEL_SLUG}"

  CUDA_VISIBLE_DEVICES="${MODEL_SCALE_GPU_IDS}" \
    uv run --project projects/typo-cot --extra lrp \
    typo-cot cot-swap \
    --model "${MODEL}" \
    --benchmark mmlu \
    --pairs "results/model-scale-pairs/${MODEL_SLUG}/pairs.jsonl" \
    --targeting attribution-4 \
    --gpu-id "${MODEL_SCALE_GPU_IDS}" \
    --output-dir "results/model-scale-cot-swap-runs/${MODEL_SLUG}"
done
```

どちらかのproducerが中断した場合は、そのmodelの同一コマンドへ `--resume` を追加して
再実行します。新しい出力directoryを作る場合は、そのflagを付けません。

9設定すべての完了後、Table 9をCPUで構築します。

```bash
uv run --project projects/typo-cot \
  typo-cot model-scale-cot-swap \
  --pairs-root results/model-scale-pairs \
  --cot-swap-runs-root results/model-scale-cot-swap-runs \
  --cohort projects/typo-cot/data/cohorts/model_scale_mmlu_first500.json \
  --output-dir results/model-scale-cot-swap
```

共通selectorはinference前にmodelごとのfinal-paper MMLU source cohortと交差します。
5つのsmall-model設定ではsubjectごと50件、Gemma-12B/27Bと70B/72Bでは100件だったため、
selectorからmodel別に250または500 source IDを保持します。これはsubmitted producerから
復元した詳細であり、PDFへ新しい主張を追加するものではありません。pair生成、CoT swap、
CPU builderの3段階ですべてexact ID setを検証します。

`n_s` は実行済みかつ再生成A正答のpair数で、Both・Question only・CoT onlyの共通分母です。
restorationはBがAと異なる `n_B` subsetだけを使います。出力は
`model_scale_records.jsonl`、`model_scale_summary.json`、
`table9_model_scale.csv`、`table9_model_scale.md`、`table9_model_scale.tex`、
`run.json` です。Qwen2.5-72Bはpublished `n_B=10` のためdirectionalな比較として残します。
詳細は [`docs/model-scale-cot-swap.md`](docs/model-scale-cot-swap.md) にあります。

## テスト

```bash
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_paper_experiment_catalog.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_targeting_fidelity_audit.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_kl_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_layerwise_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_fixed_window_answer_patching.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_coordinate_controls.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_position_controls.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_patch_text_combination.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_cot_swap.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_answer_line_deletion.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_clean_prefix_scan.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests/test_one_token_prefix_replacement.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_build_one_token_tables_*.py
uv run --project projects/typo-cot pytest projects/typo-cot/tests/test_edit_count_sensitivity.py
uv run --project projects/typo-cot --extra lrp pytest projects/typo-cot/tests
```

契約テストは、最終PDF fingerprint、完全なoperation一覧、操作内容が分かる一意なslug、
CLI JSON schema、documentation coverageを検証します。full suiteはmodel weightを
downloadせずに、pair生成、model/dataset/prompt、answer extraction、AttnLRP adapterも
exerciseします。

## リポジトリの公開範囲

ここで追跡するのは、公開実験カタログ、実装済みrunner、runtime依存、テストだけです。
中間のExp1--20 script、machine固有設定、archived outputは論文再現インターフェースでは
ないため除外しています。未実装の `typo-warning-prompt`、`input-corrector-audit`、
`restoration-order-accuracy` はカタログにのみ載り、各runnerの機能PRで追加します。
