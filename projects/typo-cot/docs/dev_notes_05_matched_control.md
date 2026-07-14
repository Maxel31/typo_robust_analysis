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

## Step 0 (master table) との接続

データアクセスは `load_correct_map()` (results.json -> sample_id -> is_correct) と
`PerturbedDatasetCreator` の baseline 読み込みに隔離してある。master table が
入ったら、`matched_stats.json` の per_target を `matched_rnd4_targets` 列に、
`analyze_matched_control.py` の入力を master table の flip/condition 列に
差し替えるだけでよい。
