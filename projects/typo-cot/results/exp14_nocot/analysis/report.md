# 実験14: no-CoT ショートカット探針 — H14 判定

- 設定数: 72 (回帰採用 n=60)
- rank-corr(noCoT_flip, DE): rho=-0.035761920640213406, p=0.7861832196584656, n=60
- サンプル OR (Mantel-Haenszel): 8.849247323956646 / crude=10.119756594597105
- 事前登録判定: rank≥0.7=False, OR>3=True → **H14=不支持**
- **層別 rank-corr (探索的, 全設定 rho≈0 は Simpson 型)**:
    - MC課題のみ noCoT_flip~DE: rho=0.726 (p=0.000, n=40)
    - 生成課題のみ noCoT_flip~DE: rho=0.633 (n=20)
    - 全設定 noCoT_flip~IE: rho=0.578 / MC noCoT_flip~IE: rho=0.755 (noCoT_flip は DE 特異でなく typo 感受性全般を反映)
- 鋭い予測 (Gemma-1B×CSQA): 全設定 top25%=False [{'perturbation': 'importance', 'rank': 9, 'of': 60, 'nocot_flip_rate': 0.36312849162011174, 'percentile_top': 0.15}, {'perturbation': 'random', 'rank': 18, 'of': 60, 'nocot_flip_rate': 0.29608938547486036, 'percentile_top': 0.3}]
    - MC課題内: top25%=True [{'perturbation': 'importance', 'rank_mc': 2, 'of_mc': 40, 'percentile_top_mc': 0.05}, {'perturbation': 'random', 'rank_mc': 5, 'of_mc': 40, 'percentile_top_mc': 0.125}]

## noCoT_flip ランキング (回帰採用設定)

| # | model | benchmark | pert | noCoT_flip | DE | IE |
|---|-------|-----------|------|-----------|----|----|
| 1 | Mistral-7B-Instruct-v0.3 | math | importance | 0.600 | 0.052 | 0.621 |
| 2 | Mistral-7B-Instruct-v0.3 | math | random | 0.489 | 0.017 | 0.407 |
| 3 | gemma-3-1b-it | gsm8k | importance | 0.460 | 0.016 | 0.373 |
| 4 | gemma-3-1b-it | gsm8k | random | 0.460 | 0.030 | 0.331 |
| 5 | gemma-3-4b-it | math | importance | 0.431 | 0.051 | 0.092 |
| 6 | gemma-3-4b-it | math | random | 0.379 | 0.027 | 0.116 |
| 7 | Llama-3.2-3B-Instruct | math | importance | 0.378 | 0.093 | 0.395 |
| 8 | Llama-3.2-1B-Instruct | commonsense_qa | importance | 0.373 | 0.162 | 0.262 |
| 9 | gemma-3-1b-it | commonsense_qa | importance | 0.363 | 0.252 | 0.228 |
| 10 | Qwen2.5-7B-Instruct | math | importance | 0.355 | 0.000 | 0.163 |
| 11 | Llama-3.2-1B-Instruct | gsm8k | random | 0.339 | 0.009 | 0.286 |
| 12 | Llama-3.2-1B-Instruct | commonsense_qa | random | 0.332 | 0.153 | 0.194 |
| 13 | Mistral-7B-Instruct-v0.3 | gsm8k | importance | 0.331 | 0.018 | 0.255 |
| 14 | Llama-3.2-1B-Instruct | gsm8k | importance | 0.322 | 0.009 | 0.303 |
| 15 | gemma-3-4b-it | gsm8k | importance | 0.316 | 0.013 | 0.096 |
| 16 | Llama-3.2-3B-Instruct | commonsense_qa | importance | 0.305 | 0.078 | 0.210 |
| 17 | Llama-3.2-3B-Instruct | math | random | 0.297 | 0.080 | 0.480 |
| 18 | gemma-3-1b-it | commonsense_qa | random | 0.296 | 0.223 | 0.182 |
| 19 | Llama-3.2-3B-Instruct | gsm8k | importance | 0.291 | 0.013 | 0.191 |
| 20 | gemma-3-4b-it | gsm8k | random | 0.289 | 0.013 | 0.058 |
| 21 | Llama-3.2-1B-Instruct | arc | importance | 0.287 | 0.185 | 0.260 |
| 22 | Llama-3.2-3B-Instruct | gsm8k | random | 0.276 | 0.011 | 0.172 |
| 23 | Llama-3.2-1B-Instruct | mmlu | importance | 0.260 | 0.129 | 0.296 |
| 24 | Mistral-7B-Instruct-v0.3 | gsm8k | random | 0.260 | 0.012 | 0.211 |
| 25 | gemma-3-4b-it | mmlu_pro | importance | 0.258 | 0.048 | 0.211 |
| 26 | gemma-3-1b-it | mmlu | importance | 0.243 | 0.186 | 0.285 |
| 27 | Qwen2.5-7B-Instruct | gsm8k | importance | 0.231 | 0.005 | 0.032 |
| 28 | Mistral-7B-Instruct-v0.3 | mmlu_pro | importance | 0.231 | 0.092 | 0.265 |
| 29 | gemma-3-1b-it | arc | importance | 0.229 | 0.158 | 0.194 |
| 30 | Llama-3.2-3B-Instruct | mmlu_pro | importance | 0.228 | 0.060 | 0.315 |
| 31 | gemma-3-1b-it | mmlu_pro | importance | 0.227 | 0.158 | 0.349 |
| 32 | Llama-3.2-3B-Instruct | mmlu | importance | 0.219 | 0.040 | 0.211 |
| 33 | Llama-3.2-1B-Instruct | mmlu_pro | importance | 0.219 | 0.140 | 0.319 |
| 34 | gemma-3-4b-it | mmlu | importance | 0.210 | 0.059 | 0.157 |
| 35 | Qwen2.5-7B-Instruct | math | random | 0.206 | 0.000 | 0.022 |
| 36 | Mistral-7B-Instruct-v0.3 | mmlu | importance | 0.205 | 0.054 | 0.171 |
| 37 | gemma-3-4b-it | commonsense_qa | importance | 0.201 | 0.059 | 0.127 |
| 38 | gemma-3-1b-it | mmlu | random | 0.197 | 0.154 | 0.198 |
| 39 | Llama-3.2-3B-Instruct | arc | importance | 0.194 | 0.051 | 0.151 |
| 40 | Llama-3.2-1B-Instruct | arc | random | 0.193 | 0.109 | 0.137 |
| 41 | Llama-3.2-1B-Instruct | mmlu | random | 0.193 | 0.105 | 0.189 |
| 42 | Llama-3.2-3B-Instruct | commonsense_qa | random | 0.190 | 0.088 | 0.125 |
| 43 | Mistral-7B-Instruct-v0.3 | commonsense_qa | importance | 0.187 | 0.067 | 0.147 |
| 44 | gemma-3-1b-it | arc | random | 0.187 | 0.157 | 0.160 |
| 45 | gemma-3-1b-it | mmlu_pro | random | 0.182 | 0.111 | 0.275 |
| 46 | Llama-3.2-1B-Instruct | mmlu_pro | random | 0.172 | 0.137 | 0.220 |
| 47 | Mistral-7B-Instruct-v0.3 | arc | importance | 0.155 | 0.042 | 0.092 |
| 48 | Qwen2.5-7B-Instruct | gsm8k | random | 0.149 | 0.008 | 0.040 |
| 49 | Mistral-7B-Instruct-v0.3 | mmlu_pro | random | 0.141 | 0.057 | 0.196 |
| 50 | Llama-3.2-3B-Instruct | mmlu | random | 0.141 | 0.044 | 0.136 |
| 51 | Llama-3.2-3B-Instruct | mmlu_pro | random | 0.138 | 0.049 | 0.151 |
| 52 | gemma-3-4b-it | commonsense_qa | random | 0.138 | 0.048 | 0.092 |
| 53 | Llama-3.2-3B-Instruct | arc | random | 0.132 | 0.038 | 0.086 |
| 54 | gemma-3-4b-it | mmlu_pro | random | 0.125 | 0.028 | 0.137 |
| 55 | gemma-3-4b-it | arc | importance | 0.123 | 0.045 | 0.070 |
| 56 | gemma-3-4b-it | mmlu | random | 0.122 | 0.034 | 0.108 |
| 57 | Mistral-7B-Instruct-v0.3 | commonsense_qa | random | 0.122 | 0.070 | 0.110 |
| 58 | Mistral-7B-Instruct-v0.3 | mmlu | random | 0.100 | 0.034 | 0.108 |
| 59 | Mistral-7B-Instruct-v0.3 | arc | random | 0.084 | 0.031 | 0.063 |
| 60 | gemma-3-4b-it | arc | random | 0.068 | 0.027 | 0.046 |
