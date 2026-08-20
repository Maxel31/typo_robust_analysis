# typo頑健化学習

このprojectは、投稿済みActivation Patching論文の固定環境を変更せず、typo頑健なadapterを
学習・評価します。本手法は今後の研究です。実装は機能単位のPRでreviewし、生成data、
checkpoint、実験結果は、固定評価protocolが主張を許すまでlocal artifactとして保持します。

科学的な実装順は次のとおり固定します。

```text
training/evaluation data -> 汎用文章上の因果layer localization
                         -> adapter training -> held-out evaluation
```

確証用targetは、汎用文章へのjoint Activation Patchingで選ぶmodel固有residual-stream windowです。
選択にはmulti-token KL restorationだけを使い、downstream answer、clean-harm score、neuron/head
screening、学習結果からtargetを変更しません。neuron/head localizationは提案手法ではなく、
探索的な負の結果として保持します。

## 環境

commandはrepository rootから実行します。学習projectは独自のlockfileを持ち、PEFTなどの
学習専用dependencyを追加します。`projects/typo-cot/uv.lock` と論文再現環境は更新しません。

```bash
TRAIN_PROJECT=projects/typo-robust-training
TRAIN_ROOT=projects/typo-robust-training/results
GPU_SELECT=5
GPU_VALIDATE=6
GPU_ID=5  # exploratory commands below
WANDB_PROJECT=typo-robustness-training
# Before training, provide WANDB_API_KEY through a secret manager or the environment.
# Optional: export WANDB_ENTITY=<team-or-user-entity>

uv sync --project "${TRAIN_PROJECT}" --locked
```

公開commandは科学的操作ごとに1つです。現在の実験では物理GPU 5と6を使用します。
公開cloneでは利用可能なGPUを指定できます。

## 1. leakageを防いだ学習・評価dataを構築する

初回構築前に、GitHub Typo Corpus v1.0.0を元repositoryの条件に従って取得します。
本repositoryからcorpus自体は再配布しません。承認ファイルは、利用を許可する
repository URLから確認済みlicenseへのJSON objectとし、未記載repositoryは除外します。
Dolmaは固定したdataset revisionのURL inventoryからstreamします。任意のcacheとして、
利用条件に従って取得したJSONLまたはJSONL.GZ sampleも指定できます。

```bash
export TYPO_GITHUB_CORPUS_PATH=/absolute/path/to/github-typo-corpus.v1.0.0.jsonl.gz
export TYPO_GITHUB_APPROVED_REPOSITORIES=/absolute/path/to/github-typo-approved-repositories.json
# Optional: export TYPO_DOLMA_CORPUS_PATH=/absolute/path/to/dolma-v1_5-sample.jsonl.gz
```

builderはnatural inputと、local Dolma archiveまたは固定URL inventoryのSHA-256に加え、
選択したshard URLを記録します。必須ファイル欠落、未承認のnatural-typo repository、
不正record、source revisionのずれがあれば、mixtureを暗黙に変えずrunを明示的に失敗させます。

```bash
uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-robustness-training-data \
  --config "${TRAIN_PROJECT}/configs/gemma4b-sanity.yaml" \
  --output-dir "${TRAIN_ROOT}/data/gemma4b-sanity"
```

既定のsanity mixtureはnon-padding student token比で、FineWeb-Edu 85%、公開されている
GSM8K/MMLU/ARC-Challengeのtrainまたはdevelopment data 10%、GitHub Typo Corpusの
training repository由来natural pairを最大5%とします。FineWeb-Eduはdocument group、
natural typoはrepositoryを単位としてsampling前に分割します。Dolma、MMLU-Pro、
MATH-500、CommonsenseQA、held-out natural repository、adjacent transpositionはv1手法の
tuningに一切使いません。

長いFineWeb-Edu/Dolma documentでは、まずcontent hashで固定される8,192文字のwindowを
1つ選び、次に固定tokenizerの512-token上限へ収まる単語境界までのprefixを保持します。
元の長さ、window境界、保持文字数、正確なtoken数を記録します。これによりtypo対象を
modelが実際に見るsequence内へ限定し、評価結果でwindowを選ばずdeduplication memoryを
制限します。

training recordはclean textを保持し、substitution、deletion、insertion、duplication、
keyboard-neighbor typoを学習時に決定的に生成します。v1では5種類のtraining operationを
一様にsampleし、一般置換の文字はtraining repositoryだけから得たnatural typo統計に
従います。tune、PR前gate、最終testのtypo pairは
固定してcontent hashを記録します。最終testのidentityは、全手法・hyperparameter・停止規則と
PR前gateを通過したcheckpointが固定されるまで封印します。

出力は次のとおりです。

- `training_sources.jsonl`: 順序付きclean training recordとsource metadata。
- `typo_statistics.json`: training repositoryのnatural pairだけから得た文字編集統計。
- `diagnostic_manifest.jsonl`: 固定したGSM8K/MMLU/ARC train/devのclean-typo
  localization pair。
- `tune_manifest.jsonl`: iteration専用の固定評価pair。
- `pre_pr_gate_manifest.jsonl`: 一度だけ使う固定PR前gate pair。
- `final_test_manifest.jsonl`: 封印された最終論文test identity。
- `evaluation_manifest.json`: splitの役割、hash、typo operation inventory、dataset revision。
- `decontamination_report.json`: exact/near-duplicateとtask denylistの監査。
- `run.json`: 引数、環境、件数、hash、完了状態。

corpus text、生成pair、checkpoint、run outputは`results/`以下のlocal artifactであり、commit
しません。

## 2. 確証用の因果windowを凍結・選択する

selection用200件とvalidation用200件のFineWeb-Edu pairを、学習data・全評価tierとID非重複で
先に凍結します。各documentへ、論文と同じkeyboard-neighbor substitution、deletion、duplication
から1 typoを決定的に適用します。幅`max(1, floor(L / 6 + 0.5))`の全連続windowについて、
編集語末tokenのcomplete decoder-block residual outputをjoint patchします。token 2--16の
pair単位multi-token KL restorationのmedianが最大のwindowを選び、完全同値なら浅い方を選びます。
独立validationのbootstrap 95%信頼区間下限が0より大きいことを必須とします。

```bash
LOCALIZATION_ROOT="${TRAIN_ROOT}/localization/generic-joint-window-v1"

uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot freeze-generic-localization-pairs \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --exclude-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --output-dir "${LOCALIZATION_ROOT}/pairs"

CUDA_VISIBLE_DEVICES="${GPU_SELECT}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-generic-joint-patch-window \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --selection-manifest "${LOCALIZATION_ROOT}/pairs/selection_manifest.jsonl" \
  --gpu-id "${GPU_SELECT}" \
  --output-dir "${LOCALIZATION_ROOT}/selection" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_VALIDATE}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot validate-generic-joint-patch-window \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-generic-joint-window.yaml" \
  --validation-manifest "${LOCALIZATION_ROOT}/pairs/validation_manifest.jsonl" \
  --window-selection "${LOCALIZATION_ROOT}/selection/window_selection.json" \
  --gpu-id "${GPU_VALIDATE}" \
  --output-dir "${LOCALIZATION_ROOT}/validation" \
  --resume
```

pair identity、除外集合、実現typo、source/model revision、pair別KL分母、scan、出力hashを
run manifestへ結合します。独立validationを通過しないmodelではlocalized-state学習を行いません。
validation不合格時も監査artifactは保存しますが、commandは非0で終了します。

### reasoning taskを使った旧探索selector

component localizationの負の結果を再現するために旧selectorも残しますが、確証用targetの選択には
使用しません。

```bash
CUDA_VISIBLE_DEVICES="${GPU_SELECT}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-distillation-layers \
  --config "${TRAIN_PROJECT}/configs/gemma4b-layer-selection.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --tasks gsm8k mmlu arc \
  --gpu-id "${GPU_SELECT}" \
  --output-dir "${TRAIN_ROOT}/localization/layers" \
  --resume
```

このcommandはreasoning診断dataの複合scoreを用いた過去のwindowを記録します。answer/harm項は、
上記の確証用generic-text selectorでは使用しません。

## 3. 探索的なneuron/head因果分析を再現する

このcommandは、現在のresidual-window手法より前に行ったcomponent-level studyを再現します。
これはablation兼negative-result auditであり、出力を確証用の学習targetの選択・変更には使いません。

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot localize-robustness-components \
  --config "${TRAIN_PROJECT}/configs/gemma4b-component-localization.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --components mlp-neuron attention-head \
  --causal-readouts answer multitoken-kl \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/components" \
  --resume
```

activation差とgradient attributionは候補のshortlistにだけ使います。
`component_selection.json`に入るのは、clean-to-typo causal patchが少なくとも2 taskで
有益な方向を示し、固定したclean-harm規則を通過した候補だけです。この過去のselection artifactは
分析用に保持し、提案adapterはSection 2で独立検証したgeneric-text residual windowを使用します。

## 4. baselineを学習し、過去のCycle 1 ablationを再現する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-noisy-language-model \
  --config "${TRAIN_PROJECT}/configs/baselines/noisy-language-model.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/noisy-language-model/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-output-matching \
  --config "${TRAIN_PROJECT}/configs/baselines/output-matching.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/output-matching/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/baselines/global-state-alignment.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/global-state-alignment/seed-42" \
  --resume

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/ablations/gemma4b-component-state-cycle1.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --component-selection "${TRAIN_ROOT}/localization/components/component_selection.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/localized-state-distillation/seed-42" \
  --resume
```

最後のcommandは、失敗したcomponent-level Cycle 1 ablationを再現する目的だけで保持します。
そのrelative-MSE objectiveとcomponent-selected LoRAは確証用の提案手法ではありません。
Section 2のartifactを使う有界なresidual-window objectiveは別の機能単位変更として公開し、
過去の失敗したtargetが現在のtargetへ暗黙に混入しないようにします。

上のcommandはversion 1 pilotの再現用であり、確証用比較として報告しません。seed 42、43、44の
matched output-only baselineとcontrolは、後続の有界residual-window学習機能で導入します。
すべてのteacher/student条件で、Teacherはclean入力を受け取りfreezeしたままとし、
宣言したStudentのLoRA parameterだけを変更できます。

公開する全学習commandではonline W&B trackingを必須にします。API keyは
`WANDB_API_KEY`からだけ渡し、`WANDB_ENTITY`は任意です。完了したoptimizer stepごとに、
集約total/component loss、learning rate、gradient norm、token throughput、現在のGPU memory、
学習開始以降のGPU memory peakをuploadします。corpus text、prompt、record ID、
API key、checkpoint内容は送信しません。
`wandb_run.json`には秘密情報を含まないrun identity、科学的binding/presentation、URL、
status、resume境界だけを保存し、`--resume`時は同一W&B runへ継続してloss curveを
重複させません。

W&B名には、略称ではなく科学的な役割を直接表示します。この機能で公開するadapter
configはversion 1のCycle 1再現一式なので、過去のrunであることも名前に明示します。

| W&Bに表示する役割 | 操作 | 意味 |
|---|---|---|
| `Historical baseline` | `Noisy-language-model training` | Cycle 1の通常のnoisy-text causal language-model baseline |
| `Historical pilot` | `Output/answer/clean-loss training` | Cycle 1の複数lossによるoutput-matching pilot |
| `Historical control` | `Global relative-MSE state alignment` | Cycle 1の全layer・全token state control |
| `Historical ablation` | `Component-level relative-MSE state distillation` | 失敗したCycle 1 neuron/head実験。確証手法ではない |

後半にはstate対象層、model、optimizer-step予算、seedを記録します。version 1のrunはすべて
`Historical Cycle 1`という別groupに分けます。有界residual-window比較のconfigとW&B mappingは
その学習機能と同じ変更で導入するため、未実装の確証用runと取り違えません。

### 確証用Cycle 3の学習と対照条件

Cycle 3では、frozen self-teacher、学習stream、all-linear LoRA容量、optimizer、厳密な
clean:noisy交互列、10M student-token予算を固定します。提案条件と出力分布整合の違いは、
独立に選択した因果windowへ有界residual-state cosine lossを追加する点だけです。
ランダム窓対照と全層対照では、state lossを測るlayer範囲だけを変更します。

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-random-window-state-distillation \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-random-window-10m.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${TRAIN_ROOT}/evaluation-data/robustness-v1" \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/random-window-state-distillation/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/cycle3/gemma4b-all-layer-state-10m.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --monitor-data "${TRAIN_ROOT}/evaluation-data/robustness-v1" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${TRAIN_ROOT}/training/cycle3/all-layer-state-distillation/seed-42"
```

利用可能なGPUが1枚の場合は2条件を直列に実行します。同じoutput directoryに互換性のある
exact checkpointがある場合だけ`--resume`を追加します。W&Bでは
`Random-window control`または`All-layer control`から始まり、
操作、層範囲、モデル、token budget、seedを含む説明的な名前を使用します。

## 5. 独立した評価studyを凍結する

adapterを比較する前に、clean/typo textの実現値を固定します。source configには、
除外用dataをbuildした際とbyte単位で同一のv3 configを指定します。これにより、
各評価項目は除外すべきtraining、diagnostic、tune IDへhashで結合されます。
この段階ではmodelを実行せず、model出力も参照しません。

```bash
EVALUATION_DATA="${TRAIN_ROOT}/evaluation-data/robustness-v1"
SOURCE_CONFIG="${TRAIN_PROJECT}/configs/cycle3/gemma4b-data-64m.yaml"

uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot freeze-robustness-evaluation \
  --protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --source-config "${SOURCE_CONFIG}" \
  --exclude-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --output-dir "${EVALUATION_DATA}"
```

このcommandはhash-boundな`tune`、一度だけ開封できる`pre_pr_gate`と`final_test`の
task/corpus manifestを書き出します。typoの実文字列はBase、出力分布整合baseline、
すべての提案adapterで共通です。task ID、corpus group、natural-typo repository、
訂正語はtraining/tune/sealed role間で交差しません。commit済みprotocolとgate定義は
[`../typo-cot/docs/robustness_evaluation_protocol_v1.md`](../typo-cot/docs/robustness_evaluation_protocol_v1.md)
に記載しています。

## 6. held-out頑健性を評価する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --evaluation-protocol "${TRAIN_PROJECT}/configs/robustness-evaluation-v1.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-cycle3-64m" \
  --evaluation-data "${EVALUATION_DATA}" \
  --evaluation-role tune \
  --layer-selection "${TRAIN_ROOT}/localization/generic-joint-window-v1/selection/window_selection.json" \
  --window-validation "${TRAIN_ROOT}/localization/generic-joint-window-v1/validation/window_validation.json" \
  --checkpoint "${TRAIN_ROOT}/training/cycle3/causal-window-state-distillation/seed-42/adapter" \
  --splits same-task unseen-task unseen-content unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/tune/targeted-seed-42"
```

`base`は常に同じpair上で自動評価します。複数の完了済みadapterを比較する場合は
`--checkpoint`を繰り返します。各pathには学習commandが書き出したhash-boundな
`training_runtime.json`が必要です。手法を変更している間は`--evaluation-role tune`だけを
使用します。全hyperparameterとcheckpointを固定した後に限り、同じcommandを
`--evaluation-role pre-pr-gate --confirm-sealed-role`として一度だけ実行します。
`final-test`は、合格したPR前checkpointを固定した後にだけ開封します。封印roleへのaccessは
immutableなdata artifactの隣へ記録され、暗黙に繰り返せません。

generic-text windowには、合格した独立validation artifactも必須です。評価器は両fileを
run identityへ結合し、過去のreasoning-task selectorへ暗黙にfallbackしません。

上のcommandは新規評価を開始します。その評価output directoryに既存の`run.json`と
pair checkpointが作成された後に再開する場合だけ、`--resume`を追加します。

reportにはclean/typo accuracy、wrong-to-right、right-to-wrong、net accuracy、clean harm、
multi-token KL、追加paired-patch gain、tokenization strata、unseen task、unseen operation、
natural typoを含めます。固定PR前gateは、primary random-2 typo accuracyがBase比+2 point以上で
95%信頼区間の下限が0より大きいこと、clean macroの低下が1 point以内で単一taskを3 point以上
壊さないこと、held-out clean perplexity比が1.02以下であること、natural typoを実質的に
悪化させないことを要求します。mechanistic patch auditは報告しますがblocking gateでは
ありません。確証的な主張にはさらに3つの学習seedを要求します。失敗した試行もlocalに保持し、
最終PRで要約します。

完全な固定protocolは
[`../typo-cot/docs/robustness_training_plan_v1.md`](../typo-cot/docs/robustness_training_plan_v1.md)
にあります。

## 7. 並行 SAE 診断トラックを実行する（GPU 0 のみ）

SAE は診断・将来研究用です。GPU 5/6 で保護している output matching と causal-window run を
変更・中断しません。凍結済み頑健性評価と GPU または人手が競合した場合は、SAE の新規投入を
止めて凍結評価を優先します。

最初に、残余の clean FineWeb-Edu stream を事前登録したsource budgetまで拡張します。builderには
凍結済み評価とlocalizationの全roleを渡す必要があり、ID・contentの完全一致に加えて、固定した
character 5-gram近重複検査を行います。このデータ準備commandはGPUを使用しません。

```bash
GPU_ID="0"
SAE_ROOT="/diskthalys/ssd14tc/sfukuhata/typo_sae_artifacts/gemma4b-v1"
TRAINING_DATA="/tmp/typo-rebuttal-manifest.vi6lNI/repo/projects/typo-robust-training/results/data/gemma4b-cycle3-64m/training_sources.jsonl"
EVALUATION_DATA="/tmp/typo-rebuttal-manifest.vi6lNI/repo/projects/typo-robust-training/results/evaluation-data/robustness-v1"
LOCALIZATION_DATA="/tmp/typo-rebuttal-manifest.vi6lNI/repo/projects/typo-robust-training/results/localization/generic-joint-window-v1/pairs"
SUPPLEMENT_DATA="${SAE_ROOT}/clean-corpus/sae_clean_supplement.jsonl"
WANDB_PROJECT="typo-robustness-sae"

uv run --project "${TRAIN_PROJECT}" --locked typo-cot build-sae-clean-corpus \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml" \
  --existing-data "${TRAINING_DATA}" \
  --exclude-data "${EVALUATION_DATA}" \
  --exclude-data "${LOCALIZATION_DATA}" \
  --training-budget minimum \
  --output-dir "${SAE_ROOT}/clean-corpus"
```

続いて、同じ統合clean streamを使い、事前登録した3点のL1係数を較正します。

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot calibrate-sparse-autoencoder-l1 \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml" \
  --training-data "${TRAINING_DATA}" \
  --training-data "${SUPPLEMENT_DATA}" \
  --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${SAE_ROOT}/l1-calibration"
```

続いて層 5 の 2 初期値 seed と層 20 の SAE を学習します。この command は typo/task record を
拒否し、最初の model forward より前に使用 record ID の SHA-256 を保存します。unique clean
source token を 100M 以上にするため追加データが必要な場合は、同じ漏洩検査を通した FineWeb-Edu
manifestを `--training-data` の繰り返しで追加します。

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-sparse-autoencoders \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml" \
  --training-data "${TRAINING_DATA}" \
  --training-data "${SUPPLEMENT_DATA}" \
  --l1-selection "${SAE_ROOT}/l1-calibration/l1_selection.json" \
  --gpu-id "${GPU_ID}" \
  --wandb-project "${WANDB_PROJECT}" \
  --output-dir "${SAE_ROOT}/training"
```

凍結済みの10M-token activation subsampleは4層のbfloat16 residual streamを保存するため、
約205 GBのディスクを使用します。また、1M-token shuffle bufferの結合・並べ替え時には、
ホストRAMを一時的に41 GB超使用し得ます。実行前に`SAE_ROOT`へ220 GB以上の空き容量と、
ホストに48 GB以上の利用可能RAMがあることを確認してください。現在指定している共有volumeは
この条件を満たしています。

最後に held-in clean text で発火率、再構成誤差 scale、WP-2 検収値を計算します。task accuracy は
測らず、評価 tier も開封しません。

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot validate-sparse-autoencoders \
  --config "${TRAIN_PROJECT}/configs/sae/gemma4b-sae-v1.yaml" \
  --registry "${TRAIN_PROJECT}/configs/sae/registry-v1.yaml" \
  --validation-data "${TRAINING_DATA}" \
  --validation-data "${SUPPLEMENT_DATA}" \
  --checkpoint-dir "${SAE_ROOT}/training" \
  --gpu-id "${GPU_ID}" \
  --output-dir "${SAE_ROOT}/validation"
```

初回WP-2 validationは、output作成やGPU/runtime初期化より前にproject rootへ排他的な予約fileを
作成し、その後に変更不能な失敗bundle lineageを`${SAE_ROOT}/wp2_attempts.json`へ記録します。
新たにレビューする初回v1事前登録は絶対pathの`wp2_project_root`と、そのdirectoryの
`wp2_project_root_identity`（`device`/`inode`）を明記し、そのidentityを持つdirectoryを実行前に
作成済みにする必要があります。このidentityは絶対pathと同じmachine-localな実行契約です。
既存の初回実行に使用したrepository内v1事前登録はbyte単位で不変に
保ち、救済lineageの証拠としては読み込めますが、新たな初回validationの開始には使えません。予約file
本体に加えて親directoryもfsyncするため、crash後もそのattemptは消費済みとなり、validationの暗黙
resumeは行いません。

救済runは、元のconfig、事前登録、training、validation、acceptance、ledgerの各hashと、単一の
改訂config / 事前登録hashを結ぶauthorizationが別レビューで承認されるまで禁止です。改訂事前登録は
同じ絶対pathの`wp2_project_root`とschema v2の`wp2_retry` lineage markerを持ち、
`initial_attempt_ledger_path`は厳密に`wp2_project_root/wp2_attempts.json`でなければなりません。
trainingとvalidationの救済modeはCLI optionではなく、このレビュー済みmarkerから自動かつ強制的に
決まり、authorizationはv2事前登録全体のSHA-256を逆向きにbindします。このためCLI引数の省略で
初回validation経路へfallbackできません。初回validation outputはproject root外でも構いません。
ledgerがその絶対pathを、authorizationが全artifact hashをbindするため、outputの移動や複製で新しい
救済budgetを作ることはできません。

救済trainingはruntime/model初期化より前に、排他的なfilesystem claimとして
`${SAE_ROOT}/wp2_retry_claim.json`を作成します。`--resume`時はoutput artifactや
`source_registry.json`を作成・更新する前にclaimとの完全一致を確認します。claimは順序付きmanifestの
path/raw hash、reserved/eligibleとして実際にmemoryへloadされたsource値と順序、load済みlayer→L1対応、
レビュー対象implementation closure、実効W&B project/entityをbindします。さらにproject root上の
nonblockingなprocess-lifetime lockにより、fresh/resumeを問わずruntimeへ到達できる救済trainingは1つだけです。
排他的なclaim/reservation/completion recordはcanonicalな既存親directory pathを`O_DIRECTORY|O_NOFOLLOW`で開き、
literalな最終basenameを`O_EXCL|O_NOFOLLOW`で作成します。このため最終componentのsymlinkでbudget slotを外部へ転送・保存できません。
レビュー済みpreregistration bytesがproject rootの`st_dev/st_ino`をinvocation横断で固定し、leaseと全authority read/writeで
open済みdirectoryのidentity一致を検証します。改名したproject rootをsymlinkまたは新しい通常directoryで置換した場合もfail closedします。
training/validationのoutput pathも最終symlinkを拒否します。
既存のclaim/reservation/completion authorityも同じ固定済み親に対するnonblocking `O_NOFOLLOW` descriptorから読み、
同一user所有・single-linkの通常fileだけを受理します。payloadが同一でもaliasは受理しません。

claim後かつtraining checkpoint作成前にprocessが停止した場合、同一claimの`--resume`は、outputが
未作成/空、または完全一致するcanonical source registry（および未完了のatomic checkpoint一時file、
または正のASCII PID標準表記名を持ち、期待するcanonical registryのbyte prefixだけを含む
同一user所有・single-linkの通常file・非symlinkのsource-registry一時file）のみを
含む場合に限り許可します。source-registry一時fileは非権威的で、回復時にregistryとしてloadしません。
それ以外のorphan artifactはfail closedです。runtime・model・optimizer初期化後、
W&B、activation収集、optimizer stepより前にcursor zeroのcheckpointをatomicに書いてfsyncします。W&Bも
claim由来のrun ID intentを`wandb.init`より前に永続化するため、再開時に同じidentityを使います。通常の
非救済runは従来どおり、既存checkpointがなければresumeできません。

救済validationもoutputや
GPU/runtime workより前に唯一の枠を予約し、claimが記録したtraining `run.json` SHAと完全一致する
checkpointだけを受理します。このSHAはmodel load直前とcompletion記録直前に再検証します。completionは
attempt 2、pass/fail、親ledger、training run、reservation、validation、acceptanceのhashを監査情報として
保存します。出力親directoryを変更してもproject-globalな予約はリセットされません。現時点では科学的な
救済authorization、v2救済registry、改訂値をrepositoryへ追加しておらず、このlineage機構そのものは
再学習を許可しません。

`--resume`は対象 command 自身のhash-bound checkpointが既にある場合だけ追加します。ただし上記の
レビュー済み救済claim-only recoveryだけは例外です。固定した
手法とgateは [`docs/sae_track_plan_v1.ja.md`](docs/sae_track_plan_v1.ja.md) にあります。
