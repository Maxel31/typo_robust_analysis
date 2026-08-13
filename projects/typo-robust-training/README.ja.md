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

## 4. baselineと提案adapterを別々に学習する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-noisy-language-model \
  --config "${TRAIN_PROJECT}/configs/baselines/noisy-language-model.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/noisy-language-model/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-output-matching \
  --config "${TRAIN_PROJECT}/configs/baselines/output-matching.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/output-matching/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-global-state-alignment \
  --config "${TRAIN_PROJECT}/configs/baselines/global-state-alignment.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/global-state-alignment/seed-42"

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot train-localized-state-distillation \
  --config "${TRAIN_PROJECT}/configs/gemma4b-targeted-lora.yaml" \
  --training-data "${TRAIN_ROOT}/data/gemma4b-sanity" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --component-selection "${TRAIN_ROOT}/localization/components/component_selection.json" \
  --seed 42 --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/training/localized-state-distillation/seed-42"
```

全条件をseed 42、43、44で繰り返します。Teacherはclean入力を受け取り、freezeしたままで、
activation patchは行いません。Studentはtypo入力を受け取り、宣言したLoRA parameterだけを
変更できます。全条件でtoken accounting、checkpoint/resume、clean-preservation評価を
共通化します。

## 5. held-out頑健性を評価する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot evaluate-typo-robustness \
  --config "${TRAIN_PROJECT}/configs/gemma4b-evaluation.yaml" \
  --data-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/evaluation_manifest.json" \
  --base-model google/gemma-3-4b-it \
  --checkpoints "${TRAIN_ROOT}/training" \
  --splits same-task unseen-task unseen-typo \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/evaluation/gemma4b"
```

reportにはclean/typo accuracy、wrong-to-right、right-to-wrong、net accuracy、clean harm、
multi-token KL、追加paired-patch gain、tokenization strata、unseen task、unseen operation、
natural typoを含めます。PR前gateはtypo accuracy +3 point以上、clean accuracy低下1 point以内、
wrong-to-rightがright-to-wrongより大きいこと、unseen transferが正であること、seed 42/43/44の
少なくとも2つで同方向であることです。失敗した試行もlocalに保持し、最終PRで要約します。

完全な固定protocolは
[`../typo-cot/docs/robustness_training_plan_v1.md`](../typo-cot/docs/robustness_training_plan_v1.md)
にあります。
