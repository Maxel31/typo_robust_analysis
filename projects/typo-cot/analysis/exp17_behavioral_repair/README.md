# 実験17: 行動修復マーカー分析 (H17) — DeepSeek-R1-Distill-Qwen-7B

R1 系モデルの `<think>` CoT 内に現れる**自己修正マーカー**(typo への気づき・言い換え)を検出し、
flip および R_Q(token importance_score)と突き合わせる。GPU 不要・読み取り専用
(WT10 = `exp-10-scope` ワークツリーの `outputs/baseline`, `outputs/perturbed` を参照)。

## マーカーの操作的定義 (exp17_analysis.py)
- **markA**: 摂動語の正しい綴りが CoT 中に出現(暗黙のうちに読み替えている)
- **markT**: typo 形がそのまま CoT に再現される
- **markC**: 同一トークンで markA と markT が共起(明示的な typo 気づき+訂正)
- **cue**: `CORRECTION_RE`("typo", "meant to say", "I think ... means" 等)にマッチする
  明示的な気づき言及が CoT 中のどこかにある
- **repair_explicit** (サンプルレベル) = cue OR (いずれかのトークンで markC)

## 判定: H17 は REFUTED(反証)

**鍵となる結果**: 明示的な自己修正マーカー(strict-cue)を持つサンプルは、持たないサンプルより
**flip が多い**(OR, 95% CI はすべて 1 を跨がない):
- math: OR=2.76 [1.76, 4.33]
- gsm8k: OR=2.96 [2.12, 4.14]
- mmlu: OR=1.98 [1.70, 2.30]

H17 が予測した方向(自己修正 → flip 減少 = 修復)とは**逆**。マーカーは修復のシグナルではなく
**「struggle(つまずき)」のシグナル**であり、モデルが typo に気づいて葛藤している箇所ほど
最終的に誤答へ転じやすい。

MATH での importance<random の逆転(flip rate: importance 0.185 < random 0.237, `## 0`)は
本実験の行動修復マーカーでは説明されない → 別トラック(構造/LaTeX 無効化, Track C)に持ち越し。

## 生成物
- `scripts/exp17_analysis.py` … マーカー検出・cross-tab・R_Q 五分位分析・FP 監査(自己完結スクリプト)
- `raw_output.txt` … 実行結果全文(`## 0`–`## 5` の表, `exp17_analysis.py` が生成)

## 再現
```
python scripts/exp17_analysis.py   # ../raw_output.txt に上書き出力 (GPU 不要)
```
注意: `WT10` パスは `exp-10-scope` ワークツリーの `outputs/` を読み取り専用参照するハードコード
(スクリプト先頭の定数)。当該ワークツリーが存在しない環境では実行不可(出力の `raw_output.txt` は
実行済みなので、通常はこの README と txt を読むだけで十分)。
