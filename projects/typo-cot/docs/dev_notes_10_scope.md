# 実験10 スコープ拡張: Qwen2.5-7B-Instruct + MATH-500 後続投入 dev notes

作成: 2026-07-16 (exp-07-correctors worktree agent)

## 概要

ARR August 2026 resubmission 向け拡張ジョブ。2つの軸:
1. Qwen2.5-7B-Instruct の5ベンチ(gsm8k, mmlu, mmlu_pro, arc, commonsense_qa)摂動側生成
2. MATH-500 の6モデル(M5+Qwen)生成完了済み結果を後続実験キューに投入

## Qwen2.5-7B-Instruct 摂動基盤

### R_Q (AttnLRP 重要度スコア)
アーカイブに6ベンチ全て完備:
- gsm8k: 1319件, 2638 imp files
- mmlu: 5700件, 11400 imp files
- mmlu_pro: 1400件, 2800 imp files
- arc: 1172件, 2344 imp files
- commonsense_qa: 1221件, 2442 imp files
- math: 1000件 (500 baseline + 500 cot)

パス: `/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_{bench}/importance_scores/`

### 摂動データセット (LXT-4 + Random-4)
5ベンチ×2条件 = 10件をCPUで作成完了 (PYTHONHASHSEED=42):
- scripts/exp10_qwen_perturbed/create_datasets.py
- 出力: datasets/perturbed/Qwen2.5-7B-Instruct_{bench}_k4_{with_choices,random_with_choices}/
- MATH-500 分は exp10_math500 で作成済み (計12件)

### 摂動側生成キュー
- scripts/exp10_qwen_perturbed/run_queue.sh (driver A=importance, driver B=random)
- GPU 3/4/5/6, run_with_gpu.sh flock 排他
- シャード境界: gsm8k 100刻み(14), mmlu 200刻み(29), mmlu_pro 100刻み(14), arc 100刻み(12), commonsense_qa 100刻み(13)
- 進捗: logs/exp10_qwen_perturbed/progress_{A,B}.json

起動コマンド:
```bash
cd /diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot
setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh A >> logs/exp10_qwen_perturbed/driverA.log 2>&1 < /dev/null &
setsid nohup bash scripts/exp10_qwen_perturbed/run_queue.sh B >> logs/exp10_qwen_perturbed/driverB.log 2>&1 < /dev/null &
```

## MATH-500 生成結果 (完了済み)

exp-10-scope worktree で6モデルの clean+摂動が完了:
- outputs/baseline/{model}_math/summary.json: 全6モデル完了
- outputs/perturbed/{model}_math_k4_{importance,random}/summary.json: 全12設定完了
- datasets/perturbed/{model}_math_k4_{with_choices,random_with_choices}/: 全12件完了

モデル一覧: gemma-3-1b-it, gemma-3-4b-it, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct

## 後続実験への拡張 TSV 行

以下の TSV 行を各 worktree のキューファイルに追記する。
MATH-500 の baseline/perturbed は exp-10-scope worktree 内のパスを使用。

### 実験1+3 (exp-01-03-transplant): shards_active.tsv 追記行

フォーマット: `# name	model	benchmark	baseline_dir	perturbed_dir	start	n`

MATH-500 (全6モデル、各500件なのでシャード分割不要):
```tsv
gemma-3-1b-it_math_k4_importance	google/gemma-3-1b-it	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/gemma-3-1b-it_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/gemma-3-1b-it_math_k4_importance	-	-
gemma-3-1b-it_math_k4_random	google/gemma-3-1b-it	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/gemma-3-1b-it_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/gemma-3-1b-it_math_k4_random	-	-
gemma-3-4b-it_math_k4_importance	google/gemma-3-4b-it	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/gemma-3-4b-it_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/gemma-3-4b-it_math_k4_importance	-	-
gemma-3-4b-it_math_k4_random	google/gemma-3-4b-it	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/gemma-3-4b-it_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/gemma-3-4b-it_math_k4_random	-	-
Llama-3.2-1B-Instruct_math_k4_importance	meta-llama/Llama-3.2-1B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Llama-3.2-1B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Llama-3.2-1B-Instruct_math_k4_importance	-	-
Llama-3.2-1B-Instruct_math_k4_random	meta-llama/Llama-3.2-1B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Llama-3.2-1B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Llama-3.2-1B-Instruct_math_k4_random	-	-
Llama-3.2-3B-Instruct_math_k4_importance	meta-llama/Llama-3.2-3B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Llama-3.2-3B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Llama-3.2-3B-Instruct_math_k4_importance	-	-
Llama-3.2-3B-Instruct_math_k4_random	meta-llama/Llama-3.2-3B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Llama-3.2-3B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Llama-3.2-3B-Instruct_math_k4_random	-	-
Mistral-7B-Instruct-v0.3_math_k4_importance	mistralai/Mistral-7B-Instruct-v0.3	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Mistral-7B-Instruct-v0.3_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Mistral-7B-Instruct-v0.3_math_k4_importance	-	-
Mistral-7B-Instruct-v0.3_math_k4_random	mistralai/Mistral-7B-Instruct-v0.3	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Mistral-7B-Instruct-v0.3_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Mistral-7B-Instruct-v0.3_math_k4_random	-	-
Qwen2.5-7B-Instruct_math_k4_importance	Qwen/Qwen2.5-7B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Qwen2.5-7B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Qwen2.5-7B-Instruct_math_k4_importance	-	-
Qwen2.5-7B-Instruct_math_k4_random	Qwen/Qwen2.5-7B-Instruct	math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/baseline/Qwen2.5-7B-Instruct_math	/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed/Qwen2.5-7B-Instruct_math_k4_random	-	-
```

### 実験4 (exp-04-fixed-target): ORDER リストへの追加タプル

フォーマット: `(model_short, benchmark)` を ORDER リストに追加。
MATH-500 の追加 (6モデル、LXT-4のみ):
```python
# MATH-500 追加 (exp-10-scope baseline/perturbed を参照する設定要)
("gemma-3-1b-it", "math"),
("gemma-3-4b-it", "math"),
("Llama-3.2-1B-Instruct", "math"),
("Llama-3.2-3B-Instruct", "math"),
("Mistral-7B-Instruct-v0.3", "math"),
("Qwen2.5-7B-Instruct", "math"),
```

注意: exp-04 の run_queue.py は ARCHIVE パスを前提にしている (`ARCHIVE / "baseline"` / `ARCHIVE / "perturbed"`)。
MATH-500 の baseline/perturbed は exp-10-scope worktree にあるため、パス解決ロジックの修正が必要。
→ ARCHIVE に存在しない場合は exp-10-scope パスにフォールバックする条件分岐を追加するか、
  MATH-500 結果をアーカイブにコピー/symlink する。

### 実験5 (exp-05-matched-rnd): matched-rnd データセット構築 + 生成

MATH-500 の matched-rnd 生成には:
1. datasets/perturbed/{model}_math_k4_with_choices/ から LXT-4 摂動位置を読む
2. matched-rnd データセットを構築 (同じ位置、ランダムな typo)
3. 摂動側生成

対象: 6モデル × math × matched-rnd = 6設定
(exp-05 worktree が存在しないため、worktree 作成から必要)

### 実験9 (exp-09-inner-repair): shards_active.tsv 追記行

フォーマット: `# name	model	benchmark	condition	start	n`

MATH-500 (全6モデル、LXT-4のみ。500件なのでシャード分割不要):
```tsv
gemma-3-1b-it_math_lxt4	gemma-3-1b-it	math	lxt4	-	-
gemma-3-4b-it_math_lxt4	gemma-3-4b-it	math	lxt4	-	-
Llama-3.2-1B-Instruct_math_lxt4	Llama-3.2-1B-Instruct	math	lxt4	-	-
Llama-3.2-3B-Instruct_math_lxt4	Llama-3.2-3B-Instruct	math	lxt4	-	-
Mistral-7B-Instruct-v0.3_math_lxt4	Mistral-7B-Instruct-v0.3	math	lxt4	-	-
Qwen2.5-7B-Instruct_math_lxt4	Qwen2.5-7B-Instruct	math	lxt4	-	-
```

## Qwen2.5-7B-Instruct の5ベンチ TSV 行 (摂動側生成完了後)

Qwen の摂動側生成が完了したら、以下の TSV 行を追記。

### 実験1+3 (exp-01-03-transplant): shards_active.tsv

mmlu は5700件のためシャード分割 (2850件×2):
```tsv
Qwen2.5-7B-Instruct_gsm8k_k4_importance	Qwen/Qwen2.5-7B-Instruct	gsm8k	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_gsm8k	<exp10_perturbed>/Qwen2.5-7B-Instruct_gsm8k_k4_importance	-	-
Qwen2.5-7B-Instruct_gsm8k_k4_random	Qwen/Qwen2.5-7B-Instruct	gsm8k	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_gsm8k	<exp10_perturbed>/Qwen2.5-7B-Instruct_gsm8k_k4_random	-	-
Qwen2.5-7B-Instruct_mmlu_k4_importance__p0	Qwen/Qwen2.5-7B-Instruct	mmlu	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_k4_importance	0	2850
Qwen2.5-7B-Instruct_mmlu_k4_importance__p1	Qwen/Qwen2.5-7B-Instruct	mmlu	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_k4_importance	2850	-
Qwen2.5-7B-Instruct_mmlu_k4_random__p0	Qwen/Qwen2.5-7B-Instruct	mmlu	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_k4_random	0	2850
Qwen2.5-7B-Instruct_mmlu_k4_random__p1	Qwen/Qwen2.5-7B-Instruct	mmlu	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_k4_random	2850	-
Qwen2.5-7B-Instruct_mmlu_pro_k4_importance	Qwen/Qwen2.5-7B-Instruct	mmlu_pro	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu_pro	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_pro_k4_importance	-	-
Qwen2.5-7B-Instruct_mmlu_pro_k4_random	Qwen/Qwen2.5-7B-Instruct	mmlu_pro	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_mmlu_pro	<exp10_perturbed>/Qwen2.5-7B-Instruct_mmlu_pro_k4_random	-	-
Qwen2.5-7B-Instruct_arc_k4_importance	Qwen/Qwen2.5-7B-Instruct	arc	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_arc	<exp10_perturbed>/Qwen2.5-7B-Instruct_arc_k4_importance	-	-
Qwen2.5-7B-Instruct_arc_k4_random	Qwen/Qwen2.5-7B-Instruct	arc	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_arc	<exp10_perturbed>/Qwen2.5-7B-Instruct_arc_k4_random	-	-
Qwen2.5-7B-Instruct_commonsense_qa_k4_importance	Qwen/Qwen2.5-7B-Instruct	commonsense_qa	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_commonsense_qa	<exp10_perturbed>/Qwen2.5-7B-Instruct_commonsense_qa_k4_importance	-	-
Qwen2.5-7B-Instruct_commonsense_qa_k4_random	Qwen/Qwen2.5-7B-Instruct	commonsense_qa	/home/sfukuhata/dev/kanolab/archive/2025/JSAI2026/outputs/baseline/Qwen2.5-7B-Instruct_commonsense_qa	<exp10_perturbed>/Qwen2.5-7B-Instruct_commonsense_qa_k4_random	-	-
```

`<exp10_perturbed>` = `/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot/outputs/perturbed`

### 実験4 (exp-04-fixed-target): ORDER 追加

Qwen 5ベンチ (LXT-4のみ):
```python
("Qwen2.5-7B-Instruct", "gsm8k"),
("Qwen2.5-7B-Instruct", "mmlu"),
("Qwen2.5-7B-Instruct", "mmlu_pro"),
("Qwen2.5-7B-Instruct", "arc"),
("Qwen2.5-7B-Instruct", "commonsense_qa"),
```

### 実験9 (exp-09-inner-repair): shards_active.tsv

Qwen 5ベンチ (LXT-4のみ、mmlu はシャード分割):
```tsv
Qwen2.5-7B-Instruct_gsm8k_lxt4	Qwen2.5-7B-Instruct	gsm8k	lxt4	-	-
Qwen2.5-7B-Instruct_mmlu_lxt4_s0_n2850	Qwen2.5-7B-Instruct	mmlu	lxt4	0	2850
Qwen2.5-7B-Instruct_mmlu_lxt4_s2850	Qwen2.5-7B-Instruct	mmlu	lxt4	2850	-
Qwen2.5-7B-Instruct_mmlu_pro_lxt4	Qwen2.5-7B-Instruct	mmlu_pro	lxt4	-	-
Qwen2.5-7B-Instruct_arc_lxt4	Qwen2.5-7B-Instruct	arc	lxt4	-	-
Qwen2.5-7B-Instruct_commonsense_qa_lxt4	Qwen2.5-7B-Instruct	commonsense_qa	lxt4	-	-
```

## Open Questions

1. **exp-04 ARCHIVE パス問題**: run_queue.py は `ARCHIVE / "baseline"` を前提としており、
   MATH-500 の baseline は exp-10-scope worktree にある。アーカイブに symlink するか、
   run_queue.py にフォールバックパスを追加するか? → ユーザー判断待ち。

2. **exp-05 worktree 不在**: matched-rnd 実験の worktree が存在しない。
   worktree 作成と matched-rnd データセット構築スクリプトの確認が必要。

3. **Qwen 5ベンチの perturbed_dir**: Qwen の摂動側生成結果は exp-10-scope worktree に出力される。
   他の worktree からの参照時に絶対パスで指定しているが、worktree 間の参照が正しく
   動作するか (特に importance_scores の .pt ファイル参照) の確認が必要。

4. **Qwen mmlu シャード境界**: 既存の5モデルは mmlu を 2分割(0-1425, 1425-)しているが、
   Qwen では 5700 件を 2850 ずつにしている。既存モデルと揃えて 1425 で 4分割する方が
   一貫性があるかもしれない。

5. **Qwen 摂動側生成の所要時間見積もり**: 7Bモデル × AttnLRP × batch_size=1 で
   全5ベンチ ≈ 11,812件。mmlu が 5700件と支配的。R1キューとGPU競合するため、
   完了まで数日かかる可能性がある。
