# 実験10④ dev notes — 自然 typo 分布での A/B 検証

担当: exp/10-scope ブランチ (MATH-500 の dev_notes_exp10_math500.md、R1蒸留の
dev_notes_exp10_scope.md とは別トラック。新規ファイルのみ追加で衝突回避)。

## 目的 (計画書 §4 実験10④)

「LXT-4 の合成 typo(ランダム編集)は人工的」という潜在批判への予防実験。
**標的語は LXT-4 と同一に固定**し、typo の編集操作分布のみ
{合成一様分布 → GitHub Typo Corpus の経験分布} に差し替える A/B 設計
(分布効果と語選択効果の分離)。スコープ: Gemma-3-4B × B2 (GSM8K, MMLU)。

## データ取得 (2026-07-18)

- GitHub Typo Corpus v1.0.0 (Hagiwara & Mita 2020, mhagiwara/github-typo-corpus)。
  **公式 S3 バケット (github-typo-corpus.s3.amazonaws.com) は消失済み (NoSuchBucket)**。
  Wayback Machine のスナップショット `20200906123950` から取得:
  `https://web.archive.org/web/20200906123950id_/https://github-typo-corpus.s3.amazonaws.com/data/github-typo-corpus.v1.0.0.jsonl.gz`
  (43,769,081 bytes, 203,270 commits / 353,055 edits)。
  保存先: `data/external/` (gitignore 対象、`projects/*/data/*`)。
- HF の `chirunder/github_typo_corrections` は同コーパス由来だが lang/is_typo
  ラベル欠落のため不採用。

## 編集操作の経験分布 (scripts/exp10_natural_typo/estimate_distribution.py)

フィルタ: `is_typo=True` かつ src/tgt とも `lang=eng` かつ行長 ≤500 →
tgt(修正後)→src(修正前) が **単一文字の ASCII アルファベット編集**
(置換 / 挿入 / 削除 / 隣接転置) で説明できるもののみ採用。

- 対象編集 254,723 件中 **138,037 件を採用**
- 操作比率: **deletion 40.2% / substitution 23.2% / insertion 22.9% / transposition 13.7%**
  (合成 LXT-4 は proximity置換・重複挿入・削除の一様3種、転置なし → 分布が大きく異なる)
- 語内位置: internal 67.2% / last 23.2% / first 9.7%
- 挿入のうち重複打鍵 (直前文字と同じ) 23.3% (合成の double_typing は 100%)
- 保存: `configs/natural_typo_distribution.json` (置換の混同行列
  P(打鍵|意図)・挿入の P(挿入|直前) 条件付き分布と推定メタデータ込み、コミット対象)

## 実装 (TDD)

- `src/typo_cot/perturbation/natural_typo.py`
  - `extract_single_edit`: SequenceMatcher による単一編集抽出
    (転置が2差分に割れるケースの追加判定、同一文字連続への挿入は右正規化=重複打鍵扱い)
  - `NaturalTypoDistribution`: 分布の入れ物 + JSON 入出力
  - `NaturalTypoGenerator.perturb`: 操作→位置バケット→文字の順にサンプル。
    実行不能な操作 (1文字語の削除等) は実行可能集合内で再正規化。ケース保存。
  - `apply_natural_typos_to_targets`: A側の標的 (token_index) を固定し
    **右から順に適用** (A側 dataset.py の offset_adjustment を全トークンに足す
    位置ずれを踏襲しない)。トークン単位シードは既存規約と同じ
    `hash((seed, sample_id, token_str))` → **PYTHONHASHSEED=42 必須**。
- テスト: `tests/test_natural_typo.py` (25件)。RED→GREEN、既存
  `tests/test_perturbation.py` は不変 (44 passed / 16 skipped)。

## B側データセット (scripts/exp10_natural_typo/create_datasets.py)

標的の出所: アーカイブ A側
`outputs/perturbed/gemma-3-4b-it_{bench}_k4_importance/results.json` の
`perturbed_tokens`。オフセットはアーカイブ baseline の
`importance_scores/{sample_id}.pt` (`offset_mapping`, `question_char_start`)。

- gsm8k: 1319件、標的 5,276 中 5,272 適用 (k=4: 1315 / k=3: 4)
- mmlu: 2850件、標的 11,385 中 11,255 適用 (k=4: 2716 / k=3: 128 ほか)
- 適用できない標的 = 真のトークンが記号のみ (`_`, `-`, `'` 等) の 134 件 (1.2%)。
  **A側では dataset.py の offset_adjustment 位置ずれによりこれらの「標的」の
  実編集が隣接語にずれて適用されていた** (例: mmlu_abstract_algebra_0094 の
  `'_' -> 'S'`)。B側は真のトークンを標的とする実装のため、記号のみの標的は
  スキップし k=3 とした (メタデータ ab_design.warnings に記録)。
- 実現操作分布 (B側): deletion 40.5% / substitution 23.0% / insertion 23.1% /
  transposition 13.4% (経験分布とよく一致)
- 出力: `datasets/perturbed/gemma-3-4b-it_{bench}_k4_natural_with_choices/`

## 生成 (scripts/exp10_natural_typo/run_generation.py + run_queue.sh)

- **A側は再生成せずアーカイブの LXT-4 生成ログを流用** (指示どおり)。
- B側のみ生成。評価軸が flip率・精度 (生成テキストのみで計算可能) のため、
  **AttnLRP を省略した生成専用スクリプト**に分離 (run_inference.py は不変更)。
  モデルロードは run_inference.py と同一 (lxtラップ込み、bf16)、greedy
  (temperature=0.0)、max_new_tokens=512、batch_size=8、seed=42。
- GPU 3/4/5/6、ヘルパー run_with_gpu.sh 経由。driver G (gsm8k 3シャード) と
  driver M (mmlu 6シャード) を並列実行 (実測 GPU 5/6 を各1枚確保)。
- スループット実測 ~1.2s/sample (バッチ8) → 総計 ~2 GPU時間の見積り内。
- 出力: `outputs/perturbed/gemma-3-4b-it_{bench}_k4_natural/`
  (アーカイブ互換スキーマ。LRP 由来フィールドは空リスト)

## A/B 比較 (scripts/exp10_natural_typo/compare_ab.py)

指標: 精度 / Δ精度 / flip率 (baseline正解→摂動後不正解) / 回復率 / 回答変化率 /
flip集合の一致 (Jaccard, McNemar) / 実現操作分布。
内的軸相関 (LRP重要度 Jaccard) は B側 AttnLRP 省略のため対象外 (open question:
必要になれば run_inference.py で B側を再生成すれば同一データセットで計算可能)。

結果: `analysis/exp10_natural_typo/ab_comparison.{json,md}` (2026-07-18 完了)

| 指標 | gsm8k A(合成) | gsm8k B(自然) | mmlu A(合成) | mmlu B(自然) |
|---|---|---|---|---|
| 精度 (baseline) | 0.8347 | 0.8347 | 0.6323 | 0.6323 |
| 精度 (摂動後) | 0.7824 | 0.7824 | 0.5860 | 0.5937 |
| Δ精度 | **-0.0523** | **-0.0523** | **-0.0463** | **-0.0386** |
| flip率 (正→誤) | **0.1163** | **0.1090** | **0.2020** | **0.2003** |
| 回答変化率 | 0.2009 | 0.1827 | 0.2888 | 0.2989 |

- gsm8k: Aのみflip=65 / Bのみflip=57 / 両方=63、flip集合 Jaccard=0.34、
  **McNemar p=0.526** (n_correct=1101)
- mmlu: Aのみflip=172 / Bのみflip=169 / 両方=192、flip集合 Jaccard=0.36、
  **McNemar p=0.914** (n_correct=1802)

**結論: 標的語を固定したまま編集操作分布を合成→自然に差し替えても、
flip率・Δ精度は両ベンチで同水準 (McNemar 有意差なし)。効果量が typo 生成分布に
依存しないという予測を支持** (計画書の予測「効果量は同水準」どおり。
むしろ自然分布の方がわずかに弱い傾向: mmlu Δ精度 -4.6→-3.9pt、これも計画書の
「自然typoは効果量がやや減るが構造は保持」と整合)。flip集合の Jaccard ~0.35 は
「どのサンプルが flip するか」は編集の実現値に依存して入れ替わるが、
集計レベルの効果量は分布に頑健であることを示す。

## Open questions / 制約

- 公式配布 S3 の消失により、コーパスの再取得は Wayback Machine 依存
  (上記 URL とバイト数を記録済み。分布 JSON 自体はコミット済みで
  再推定なしに B側データセットは再現可能)。
- A側の 1.2% の標的位置ずれ (dataset.py の offset_adjustment) は既知の A側実装
  仕様として扱い、A側データは修正しない (アーカイブ確定データのため)。
- B側は生成のみで LRP なし → 内的軸相関は未計算 (「可能なら」項目)。

## 再現手順

```bash
cd projects/typo-cot
# 1. コーパス取得 (data/external/ へ) → 分布推定
uv run --no-sync python scripts/exp10_natural_typo/estimate_distribution.py
# 2. B側データセット作成 (CPU)
PYTHONHASHSEED=42 uv run --no-sync python scripts/exp10_natural_typo/create_datasets.py
# 3. B側生成 (GPU, ヘルパー経由)
setsid nohup bash scripts/exp10_natural_typo/run_queue.sh G >> logs/exp10_natural_typo/driverG.log 2>&1 &
setsid nohup bash scripts/exp10_natural_typo/run_queue.sh M >> logs/exp10_natural_typo/driverM.log 2>&1 &
# 4. A/B 比較表
uv run --no-sync python scripts/exp10_natural_typo/compare_ab.py
```
