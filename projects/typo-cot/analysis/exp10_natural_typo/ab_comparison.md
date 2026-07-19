# 実験10④ A/B 比較: 合成typo (LXT-4) vs 自然typo (GitHub Typo Corpus 分布)

モデル: gemma-3-4b-it / 標的語は A/B で同一 (LXT-4 の標的トークンを固定) / k=4

| 指標 | gsm8k A(合成) | gsm8k B(自然) | mmlu A(合成) | mmlu B(自然) |
|---|---|---|---|---|
| 精度 (baseline) | 0.8347 | 0.8347 | 0.6323 | 0.6323 |
| 精度 (摂動後) | 0.7824 | 0.7824 | 0.586 | 0.5937 |
| Δ精度 | -0.0523 | -0.0523 | -0.0463 | -0.0386 |
| flip率 (正→誤) | 0.1163 | 0.109 | 0.202 | 0.2003 |
| 回答変化率 | 0.2009 | 0.1827 | 0.2888 | 0.2989 |

- **gsm8k**: flip一致 Jaccard=0.3405, Aのみflip=65, Bのみflip=57, 両方flip=63, McNemar p=0.526244 (n_correct=1101)
  - 操作分布 A: {'double_typing': 0.3357, 'omission': 0.3332, 'proximity': 0.3311} / B: {'deletion': 0.405, 'insertion': 0.231, 'substitution': 0.2305, 'transposition': 0.1335}
- **mmlu**: flip一致 Jaccard=0.3602, Aのみflip=172, Bのみflip=169, 両方flip=192, McNemar p=0.913753 (n_correct=1802)
  - 操作分布 A: {'double_typing': 0.3379, 'omission': 0.3207, 'proximity': 0.3414} / B: {'deletion': 0.4064, 'insertion': 0.2252, 'substitution': 0.2284, 'transposition': 0.1399}

注: 内的軸相関 (LRP重要度 Jaccard) は B側で AttnLRP を省略したため対象外。
