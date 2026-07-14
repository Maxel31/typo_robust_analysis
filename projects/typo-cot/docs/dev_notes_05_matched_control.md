# 実験5(双子語統制) 開発メモ

ブランチ: `exp/05-matched-control`。共有ドキュメント (experiment_plan.md / work_items.md /
README.md) は編集しない方針のため、実験5の実装メモはこのファイルに置く。

## 実装物

| ファイル | 役割 |
|---|---|
| `src/typo_cot/perturbation/matched_sampler.py` | 5変数の層化マッチング (FeatureExtractor / MatchedTwinSampler / compute_smd_table) |
| `src/typo_cot/perturbation/matched_dataset.py` | Matched-Rnd-4 データセット作成 (MatchedTwinDatasetCreator) |
| `src/typo_cot/perturbation/dataset.py` | `target_word_list` 引数を追加 (外部選定標的の注入)。適用ループを `_apply_candidate_perturbations` に抽出 (挙動不変) |
| `src/typo_cot/analysis/matched_control.py` | 対応のある McNemar exact + リスク差 Wald 95% CI |
| `scripts/exp5/make_matched_twin_dataset.py` | データセット作成 CLI (SMD 表・緩和率を matched_stats.json に出力) |
| `scripts/exp5/compare_with_rebuttal.py` | rebuttal 版 (品詞・文字長のみ) との選択差分の確認 |
| `scripts/exp5/analyze_matched_control.py` | clean/LXT-4/Matched-Rnd-4 の results.json から McNemar 表を生成 |

テスト: `tests/test_matched_sampler.py` / `tests/test_matched_dataset.py` /
`tests/test_matched_control.py` (すべて GPU 不要、フェイクトークナイザ・注入関数で完結)。

## マッチング仕様 (実装での確定事項)

- 標的 = 重要度降順 top-k (LXT-4 と同一)。プール = 同一質問内の残りトークン
  (rebuttal 版と同じく、トークン数 ≤ k のときは全トークンにフォールバック)。
- 特徴量 5 種:
  1. 内容語/機能語 (spaCy en_core_web_sm の POS。ロード不能時は機能語リスト)
  2. 文字長 (サブワードマーカー除去後)
  3. Zipf 頻度ビン (wordfreq, 0.5 刻み)
  4. 分割増分 = 標的と同じ摂動タイプを候補に仮適用して再トークナイズしたピース数差。
     標的の摂動タイプは dataset.py と同一の token_seed 式
     `hash((seed, sample_id, token))` の抽選で決まる (PYTHONHASHSEED=42 必須)
  5. (第2優先) 質問文埋め込み (all-MiniLM-L6-v2, CPU) との cos 類似の 0.1 幅ビン
- 緩和ラダー (レベルを MatchRecord に記録、緩和率は SMD 表に出力):
  - L0 exact: クラス一致 ∧ |Δ長|≤1 ∧ 頻度ビン一致 ∧ 増分一致 ∧ 中心性ビン一致
  - L1 no_centrality: L0 から中心性を除外
  - L2 caliper: クラス一致 ∧ |Δ長|≤2 ∧ |Δ頻度ビン|≤0.5 ∧ |Δ増分|≤1
  - L3 class_len: クラス一致のみ (距離最小選好) = rebuttal 相当
  - L4 any: 制約なし (距離最小選好)
- レベル内は特徴距離最小 → 乱数タイブレーク (`hash((seed, sample_id, "matched_selection"))`)。
  非復元抽出。
- 摂動注入は既存の適用ループそのもの (per-token seed / offset 調整 / perturb() 失敗時の
  次候補フォールバック)。双子語で perturb() が失敗した場合は残プールのシャッフル
  バックアップで k=4 を充填し、その混入率を `applied_from_matched_rate` として記録。

## 実行手順 (23設定の新規生成)

```bash
# 1) データセット作成 (CPU)
PYTHONHASHSEED=42 uv run python scripts/exp5/make_matched_twin_dataset.py \
  --baseline_dir <archive>/outputs/baseline/{model}_{bench} -k 4 \
  --output_dir data/exp5/matched_rnd

# 2) 生成 (GPU; LRP 不要なので生成専用ランナーを再利用)
bash <repo>/tmp/gpu-locks/run_with_gpu.sh bash -c \
  'uv run python scripts/rebuttal/run_generation_only.py \
     --model {hf_model} --benchmark {bench} \
     --perturbed_data data/exp5/matched_rnd/{model}_{bench}_k4_matched_rnd/perturbed_dataset.json \
     --gpu_id "$CUDA_VISIBLE_DEVICES" --batch_size 4 --output_dir results/exp5/perturbed'

# 3) 統計
uv run python scripts/exp5/analyze_matched_control.py \
  --clean_results <archive>/outputs/baseline/{model}_{bench}/results.json \
  --lxt_results   <archive>/outputs/perturbed/{model}_{bench}_k4_importance/results.json \
  --matched_results results/exp5/perturbed/{model}_{bench}_k4_matched_rnd/results.json
```

注意: `run_generation_only.py` の `--gpu_id` は内部で CUDA_VISIBLE_DEVICES を
上書きするため、run_with_gpu.sh 経由では必ず `"$CUDA_VISIBLE_DEVICES"` を渡すこと。

- Gemma-3-4B-it × GSM8K/MMLU は rebuttal ログ流用のため生成不要 (仕様)。
  ただし rebuttal 版はマッチ変数が 2 変数のため、5 変数版で生成し直すかは
  ユーザー判断 (open question)。
- GLMM (誤答 ~ 条件 + (1|item) + (1|設定)) は全設定の生成完了後に別途実装する。

## GPU スモーク結果 (2026-07-14, 合格)

環境修正を exp/04-fixed-target から cherry-pick 済み: e8e2139 (transformers<5 pin) /
9500b0d+b37f59d (setup_device が外部 CUDA_VISIBLE_DEVICES を尊重) / 1adaaaf
(torch==2.9.1 pin; uv.lock は取り込み側で再解決)。torch 2.9.1+cu128 /
transformers 4.57.6、既存テスト 43 passed / 6 skipped。

Gemma-3-4B-it × Matched-Rnd-4 (`--limit 24`, batch_size 4, run_with_gpu.sh 経由):

| bench | n | accuracy | 出力 |
|---|---|---|---|
| GSM8K | 24 | 0.667 (16/24) | `results/smoke/exp5_perturbed/gemma-3-4b-it_gsm8k_k4_matched_rnd/` |
| MMLU  | 24 | 0.458 (11/24) | `results/smoke/exp5_perturbed/gemma-3-4b-it_mmlu_k4_matched_rnd/` |

- 双方とも 1 分未満で完走 (モデルロード込み ~80s)。ヘルパーが GPU 3/4 を別々に取得し、
  wrapper は「外部設定の CUDA_VISIBLE_DEVICES=3 を優先します」をログ出力 (修正が有効)。
- スキーマ互換: results.json のエントリキー・perturbed_tokens キーがアーカイブの
  `outputs/perturbed/gemma-3-4b-it_gsm8k_k4_importance/results.json` と完全一致。
- 目視 10 件 (GSM8K 5 + MMLU 5): 摂動注入トークンは厳密 top-4 重要度標的と全件
  非重複 (`results/smoke/exp5_visual_check_strict_top4.json`)。なおアーカイブの
  LXT-4 perturbed_tokens は perturb() 失敗フォールバックで厳密 top-4 から
  ずれることがあり、その比較では 2/10 が見かけ上重複する
  (`results/smoke/exp5_visual_check_targets.json`) が、サンプラの標的定義
  (厳密 top-k) とは非重複であり rebuttal 手続きと整合。
- 総合判定 PASS: `results/smoke/exp5_gpu_smoke_summary.json`

## 本番ラン (2026-07-14 開始)

ユーザー承認: 25 設定の本番実行、および rebuttal 期 2 設定
(gemma-3-4b-it × gsm8k/mmlu) も 5 変数版で統一再生成する。

- データセット構築 (CPU): `bash scripts/exp5/run_all_matched_datasets.sh`
  を nohup 実行 (ログ `logs/exp5/build_all.log`、設定別ログ `logs/exp5/make_*.log`)。
  gemma-3-4b-it × gsm8k/mmlu の 2 設定はスモーク時に構築済みの 5 変数フル版
  (embedding_enabled=true、全サンプル) をそのまま使用。
- 生成キュー (GPU): `nohup bash scripts/exp5/run_generation_queue.sh > logs/exp5/queue.log 2>&1 &`
  - 25 設定 (`scripts/exp5/settings_25.txt` の行順)。1 設定 = 1 シャード =
    run_with_gpu.sh 1 呼び出し (GPU_LOCK_TIMEOUT=86400)。シャード間でロック解放。
  - 完了スキップ: `results/exp5/perturbed/<name>/summary.json` の存在。
  - 進捗 JSON: `logs/exp5/queue_progress.json`。監視コマンド:
    `uv run --no-sync python scripts/exp5/queue_progress.py --print`
  - 依存追加: `uv sync --extra lrp --extra matched` 済み (torch 2.9.1+cu128 /
    transformers 4.57.6 は不変)。以後は全て `uv run --no-sync`。
- 出力パス (25 設定共通形式):
  - データセット: `data/exp5/matched_rnd/{model}_{bench}_k4_matched_rnd/`
    (perturbed_dataset.json / config.json / matched_stats.json)
  - 生成結果: `results/exp5/perturbed/{model}_{bench}_k4_matched_rnd/`
    (results.json / summary.json / config.json)
  - model ∈ {Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, gemma-3-1b-it,
    gemma-3-4b-it, Mistral-7B-Instruct-v0.3},
    bench ∈ {gsm8k, mmlu, mmlu_pro, arc, commonsense_qa}
  - 生成結果の bulk は git にコミットしない (repo ルート .gitignore の
    `projects/*/results/*` / `projects/*/data/*` により無視)。正典記録は
    Step 0 master table への取り込みとする。
- 最初のシャード検証 (gemma-3-4b-it_gsm8k, 2026-07-14 20:28 完了, 38 分):
  - accuracy 0.8089 (1067/1319)。rebuttal 2 変数版 matched_random 0.8120
    (1071/1319) と -0.3pt 差で整合。スモーク (n=24) 0.667 とも矛盾なし。
  - results.json のエントリキー・perturbed_tokens キーはアーカイブ
    `outputs/rebuttal/perturbed/gemma-3-4b-it_gsm8k_k4_matched_random` と完全一致。
    sample_id 順序も一致、全 1319 サンプルで摂動 4 トークン。
  - アーカイブ LXT-4 の適用トークンとの重複 2.1% (111/5276) — LXT 側の
    perturb() 失敗フォールバック由来で、厳密 top-4 標的とは非重複 (スモークと同判定)。

## Step 0 (master table) との接続

データアクセスは `load_correct_map()` (results.json -> sample_id -> is_correct) と
`PerturbedDatasetCreator` の baseline 読み込みに隔離してある。master table が
入ったら、`matched_stats.json` の per_target を `matched_rnd4_targets` 列に、
`analyze_matched_control.py` の入力を master table の flip/condition 列に
差し替えるだけでよい。
