# typo頑健化学習

このprojectは、投稿済みActivation Patching論文の固定環境を変更せず、typo頑健なadapterを
学習・評価します。本手法は今後の研究です。固定したPR前gateでclean性能を維持しながら
held-out typo頑健性が向上するまで、実装branchはlocalに保持し、学習PRを作成しません。

科学的な実装順は次のとおり固定します。

```text
training/evaluation data -> layer localization -> neuron/head localization
                         -> adapter training -> held-out evaluation
```

neuron screeningでlayer localizationを置き換えません。診断dataで選んだlayer window内の
componentだけを候補にし、少なくとも2つの診断taskでanswer-levelまたはmulti-token KLの
causal patchingを通過させます。state-alignment lossは選択componentだけで計算し、parameterは
そのcomponentを含むlayerのLoRA adapterを通じて更新します。

## 環境

commandはrepository rootから実行します。学習projectは独自のlockfileを持ち、PEFTなどの
学習専用dependencyを追加します。`projects/typo-cot/uv.lock` と論文再現環境は更新しません。

```bash
TRAIN_PROJECT=projects/typo-robust-training
TRAIN_ROOT=projects/typo-robust-training/results
GPU_ID=0

uv sync --project "${TRAIN_PROJECT}" --locked
```

公開commandは科学的操作ごとに1つです。このrepositoryでの実装検証では`GPU_ID=3`として
物理GPU 3だけを使います。公開cloneでは利用可能な物理GPUを1枚だけ指定できます。

## 1. leakageを防いだ学習・評価dataを構築する

初回構築前に、GitHub Typo Corpus v1.0.0とDolma v1.5 sampleを、それぞれの
利用条件および元repositoryの条件に従って取得します。本repositoryからcorpus
自体は再配布しません。GitHub承認ファイルは、利用を許可するrepository URLから
確認済みlicenseへのJSON objectとし、未記載repositoryは除外します。Dolmaは
JSONLまたはJSONL.GZ形式で用意します。

```bash
export TYPO_GITHUB_CORPUS_PATH=/absolute/path/to/github-typo-corpus.v1.0.0.jsonl.gz
export TYPO_GITHUB_APPROVED_REPOSITORIES=/absolute/path/to/github-typo-approved-repositories.json
export TYPO_DOLMA_CORPUS_PATH=/absolute/path/to/dolma-v1_5-sample.jsonl.gz
```

builderは3つのlocal inputすべてのSHA-256を記録します。ファイル欠落、未承認の
natural-typo repository、不正record、source revisionのずれがあれば、mixtureを
暗黙に変えずrunを明示的に失敗させます。

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

training recordはclean textを保持し、substitution、deletion、insertion、duplication、
keyboard-neighbor typoを学習時に決定的に生成します。v1では5種類のtraining operationを
一様にsampleし、一般置換の文字はtraining repositoryだけから得たnatural typo統計に
従います。tune、PR前gate、最終testのtypo pairは
固定してcontent hashを記録します。最終testのidentityは、全手法・hyperparameter・停止規則と
PR前gateを通過したcheckpointが固定されるまで封印します。

出力は次のとおりです。

- `training_sources.jsonl`: 順序付きclean training recordとsource metadata。
- `typo_statistics.json`: training repositoryのnatural pairだけから得た文字編集統計。
- `diagnostic_manifest.jsonl`: GSM8K/MMLU/ARC train/devのlocalization record。
- `tune_manifest.jsonl`: iteration専用の固定評価pair。
- `pre_pr_gate_manifest.jsonl`: 一度だけ使う固定PR前gate pair。
- `final_test_manifest.jsonl`: 封印された最終論文test identity。
- `evaluation_manifest.json`: splitの役割、hash、typo operation inventory、dataset revision。
- `decontamination_report.json`: exact/near-duplicateとtask denylistの監査。
- `run.json`: 引数、環境、件数、hash、完了状態。

corpus text、生成pair、checkpoint、run outputは`results/`以下のlocal artifactであり、commit
しません。

## 2. layer windowを選択する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot select-distillation-layers \
  --config "${TRAIN_PROJECT}/configs/gemma4b-layer-selection.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --tasks gsm8k mmlu arc \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/layers"
```

編集語末位置で全layerを走査し、診断data上のmulti-token KL restoration、answer restoration、
clean harmから連続windowを1つ固定します。`[0,6)`は論文に由来するGemmaの候補であり、
強制される答えではありません。

## 3. neuronとattention headを因果的に局所化する

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run --project "${TRAIN_PROJECT}" --locked \
  typo-cot localize-robustness-components \
  --config "${TRAIN_PROJECT}/configs/gemma4b-component-localization.yaml" \
  --diagnostic-manifest "${TRAIN_ROOT}/data/gemma4b-sanity/diagnostic_manifest.jsonl" \
  --layer-selection "${TRAIN_ROOT}/localization/layers/layer_selection.json" \
  --components mlp-neuron attention-head \
  --causal-readouts answer multitoken-kl \
  --gpu-id "${GPU_ID}" \
  --output-dir "${TRAIN_ROOT}/localization/components"
```

activation差とgradient attributionは候補のshortlistにだけ使います。
`component_selection.json`に入るのは、clean-to-typo causal patchが少なくとも2 taskで
有益な方向を示し、固定したclean-harm規則を通過した候補だけです。

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
