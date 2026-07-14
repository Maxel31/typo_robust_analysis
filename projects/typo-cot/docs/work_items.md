# 作業項目一覧(work items)

[experiment_plan.md](experiment_plan.md) §5(実行計画)・§7(実装マッピング)に基づく作業分解(v2)。
実行順: Step 0 → 実験4 → 実験5 → 実験7 → 実験1 → 実験3 → 実験2 → 実験8 → 実験9 → 実験6 → 実験10。
モジュール名は §7.3 の提案名。migrated 済みモジュール(§7.4)は拡張のみ。

## Step 0: 資産棚卸しと凍結(GPU 不要・依存なし)

- [ ] `scripts/step0_build_master_table.py` — `configs/paths.yaml` のアーカイブ(archive_baseline 90 / archive_perturbed_outputs 322 / archive_perturbed_datasets 360 / archive_rebuttal_*)を `docs/v1_run_manifest.md`(25設定の正典)と突合し、§7.1 スキーマの統合 parquet(1行=1サンプル×1設定: キー・3条件生成・摂動メタ・帰属・flip/span判定・本文/付録ラベル)を構築。読み口は既存 `data/loader.py`・`analysis/analyzer.py`
- [ ] `configs/registry.yaml` — prompt 本文ハッシュ・seed・greedy 設定・答えトリガー文字列・モデル revision を凍結し、既存 `models/prompts.py`・`config.py` から読む形に接続(公開パッケージの一部)
- [ ] `visualization/*` — 図表出力の軸名を Surface / Attribution に統一(修正C)
- [ ] span 失敗全数表(全体 7.28%、最大 Mistral×GSM8K 31.16%)の再現+`main_or_appendix` ラベル確定

## 実験4: fixed-target 全設定展開(依存: Step 0)

- [ ] `scripts/rebuttal/run_fixed_target_attribution.py`(4設定実装済み)を昇格: `lrp/analyzer.py`+`models/wrapper.py`(teacher-forcing)を M5+Qwen × B5+MATH に一般化
- [ ] 統合テーブルの flip フラグで再計算対象を絞る(flip 事例のみ backward、コスト約1/4)
- [ ] fixed 版 Jaccard@{5,10,20}・ρ(J|R)_fixed(bootstrap 95%CI+Holm)・Δρ paired bootstrap・形式間メタ比較・GLMM 再推定(`analysis/stats.py`)
- [ ] fixed 版 R_C ランキングを統合テーブルに追記(実験2・6・9 の上流入力)。Fig.3 差替素材の出力

## 実験5: マッチド統制の拡張(依存: Step 0。backward 不要)

- [ ] `perturbation/matched_sampler.py` — `scripts/rebuttal/make_matched_random_dataset.py` を5変数層化マッチ(内容/機能語・文字長±1・Zipf 頻度ビン(wordfreq)・分割数増分・埋め込み中心性ビン)に拡張+SMD バランス表
- [ ] 摂動注入は既存 `perturbation/generator.py`、Matched-Rnd-4 の新規生成23設定は `scripts/run_inference.py` 系を再利用
- [ ] McNemar+リスク差 CI、GLMM(誤答 ~ 条件 + (1|item) + (1|設定))、Table 3 への1行追加+付録全表

## 実験7: 校正器3段ラダー(依存: Step 0(R_Q 参照)。backward 不要)

- [ ] `defense/correctors.py` — pyspellchecker(`scripts/rebuttal/make_spellfix_dataset.py`・`analyze_spellfix.py` が実装済み参照)/ニューラル文脈型(neuspell 系 or T5 系 GEC)/LLM 校正(Qwen2.5-7B、保守的プロンプト+温度0)の統一インターフェース
- [ ] 校正テキストジョブ15本+評価生成73ラン(`scripts/run_inference.py`・`evaluation/extractor.py` 再利用)
- [ ] token 復元率/byte-identical 率/誤修正率/失敗トークンの R_Q 偏在(Mann–Whitney)+byte-identical 検算+3面図
- [ ] 失敗語の LOO 再検証(実験6-(iv) の副産物で再集計、追加コスト≈0)

## 実験1: CoT 移植 2×2(依存: Step 0 のみ。実験4と独立、前倒し可。backward 不要)

- [ ] `intervention/cell_builder.py` — 生成ログから clean/摂動 CoT を取り出し答え句直前で切断、4セル(A/B/C/D)の teacher-forcing プロンプト構築。切断・答え句検出は `evaluation/extractor.py` 流用、生成は `models/wrapper.py` 拡張
- [ ] 7モデル×6ベンチ×摂動2条件(LXT-4+Random-4=修正A)で答えスパン生成(≦16 トークン)。セルBの再生成検証
- [ ] TE/DE/IE 分解・媒介割合・GLMM(flip ~ Q_p × C_p + (1|item))、除外規定(答え早期露出)とセルD の条件付き報告
- [ ] パターンX/Y 両分岐の結論文を事前執筆(実験8 の設計入力)

## 実験3: KL 発散プロファイル(依存: Step 0+実験1(セルC forward 共有)。precision@k は実験6-(iv) の LOO も参照)

- [ ] `intervention/kl_profiler.py` — セルC teacher-forcing にフックし位置別 KL・log-prob 低下・rank を逐次スカラー化(ロジット非保存)。`cell_builder` と一体実装
- [ ] 発散オンセット定義・KL 上位10 vs R_C 上位10 の precision@k(位置シャッフル帰無)・flip 群比較・Random-4 条件も適用

## 実験2: 重要 CoT トークン削除介入(依存: Step 0+実験4(fixed 版 R_C ランキング)+実験6-(iv)(LOO 腕))

- [ ] `intervention/cot_editor.py` — CoT 語タイプ単位の削除/「…」置換/同品詞・同頻度帯置換+再トークン化(POS タガー・wordfreq)。生成は実験1と同じ teacher-forcing 経路
- [ ] コア対比(top vs 一致ランダム、M5×B5)/完全グリッド(標的3×操作3×k3、Gemma-3-4B×B2)/回復曲線(M3×B2、p∈{0,25,50,75,100}+permutation 検定)
- [ ] top-LOO 削除腕(M3×B2、修正B)。内容語層と数値層の別枠報告、McNemar+用量反応単調性

## 実験8: activation patching(依存: 実験1(DE 規模・flip ペア)+実験3(S2 測定点))

- [ ] `intervention/patching.py` — forward hooks の 2-pass patching 実行器(残差ストリームのキャッシュ・上書き)+質問長差オフセット整列+層窓スイープランナー(TransformerLens 対応確認、未対応なら素 hooks/nnsight)
- [ ] セルC 構成(`cell_builder` 流用)で M3×B2、flip 事例300〜500/設定を LXT-4/Random-4 半々(修正A)、部位3×方向2×3層幅窓
- [ ] 指標: 元答えロジット回復率・flip 解消率・層プロファイル+S2(c1 分布への KL 回復)

## 実験9: 内部修復分析(依存: Step 0(摂動語スパン位置・R_Q・flip)。生成不要・backward 不要)

- [ ] `repair/lexicon_probe.py` — hidden states 付き forward(`models/wrapper.py` 拡張)、摂動語スパン末尾の層別 hidden 抽出、logit lens、層別 cos 類似=修復スコア
- [ ] M5+Qwen×B5、Random-4 語も追加(修正A)。ロジスティック回帰 flip ~ 修復スコア + 分割増分 + Zipf + R_Q + (1|item)(`analysis/metrics.py`+statsmodels)

## 実験6: 帰属手法比較+LOO 再構成(依存: 実験4(fixed-target プロトコル凍結))

- [ ] `attribution/alternatives.py` — G×I / IG(m=32、completeness チェック)/ occlusion / rollout を `lrp/analyzer.py` と同一出力形式で実装(G×I・IG は backward)
- [ ] `intervention/loo_scorer.py` — 語タイプ全出現削除→答えトリガー teacher-forcing→log-prob 低下のバッチスコアラ(修正Bの本体。実験2 LOO 腕・実験3 precision@k・実験7 失敗語再検証の上流)
- [ ] 3面評価: efficacy 順位(G×I は M5×B5、IG/occlusion/LOO は M3×B2)・集合 Jaccard(LOO 含む)・各手法 ρ(J|R)(fixed-target プロトコル下)

## 実験10: スコープ拡張(①②は Step 0 直後から並行可、④は主要実験の後)

- [ ] ① Qwen2.5-7B 追加 — `models/wrapper.py`・`models/prompts.py` にチャットテンプレート追加、既存パイプラインで全 Tier F(基盤生成は第1期に前倒し、実験4・1・3 が消費)
- [ ] ② MATH-500 追加 — `data/loader.py` にローダ追加、`lrp/analyzer.py` に CoT 長上限+チャンク化のメモリ管理(調整弁: AttnLRP 系後回し)
- [ ] ③ R1 蒸留系(7〜8B)を実験1・3 のみに追加(R_C 非計算は Limitations 明記)
- [ ] ④ `perturbation/natural_typo.py` — GitHub Typo Corpus の編集操作経験分布サンプラー(`perturbation/generator.py` のインターフェース準拠)、標的固定 A/B 設計、Gemma-3-4B×B2

## 並行トラック(人手、週次表 §5.2 の並行列)

- [ ] 統計整備 — `analysis/stats.py`: McNemar+リスク差 CI / GLMM(+(1|item)) / Holm / paired bootstrap の定型化(全実験共通、週1〜2)
- [ ] H1〜Hn レジストリ — 各実験の仮説・判定基準・分岐時解釈の一覧表を起草(週1〜2、実験1 のX/Y 分岐文を含む)
- [ ] 公開パッケージ — `configs/registry.yaml`+統合テーブル+匿名リポジトリ公開の整備(§6.6、週7〜)
- [ ] 修正Cの紙面設計 — 本文/付録の剪定確定(Step 0 でラベル付与)、Surface/Attribution 改名、Fig.3 差替とフレームワーク図 SVG(§6.1/6.10、週5〜)
