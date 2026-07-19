# A2 (i) リーク層別 — restore「自明コピー」批判への反証

セル C (typo質問+clean CoT強制) の restore を、強制 clean CoT prefix 内に最終答え文字列が現れるか (leak) で層別。**リークなし事例で restore が高いほど「答えは CoT に書かれておらず再導出された」= 自明コピー説の反証。**

## GSM8K

### gemma-3-4b-it (n_te_flipped=162)
- overall restore = 0.920
- **leak**:    n=162  restore=0.920 [0.88,0.96]
- **no-leak**: n=0  —
  - lastline:     n=162  restore=0.920 [0.88,0.96]
  - earlier-only: n=0  —
  - absent:       n=0  —

### Llama-3.2-3B-Instruct (n_te_flipped=331)
- overall restore = 0.958
- **leak**:    n=329  restore=0.964 [0.94,0.98]
- **no-leak**: n=2    restore=0.000 [0.00,0.00]
  - lastline:     n=329  restore=0.964 [0.94,0.98]
  - earlier-only: n=0  —
  - absent:       n=2    restore=0.000 [0.00,0.00]

### Mistral-7B-Instruct-v0.3 (n_te_flipped=270)
- overall restore = 0.944
- **leak**:    n=266  restore=0.947 [0.92,0.97]
- **no-leak**: n=4    restore=0.750 [0.25,1.00]
  - lastline:     n=264  restore=0.947 [0.92,0.97]
  - earlier-only: n=2    restore=1.000 [1.00,1.00]
  - absent:       n=4    restore=0.750 [0.25,1.00]

### GSM8K pooled over models
- overall restore = 0.945 (n=763)
- leak:    n=757  restore=0.948 [0.93,0.96]
- no-leak: n=6    restore=0.500 [0.17,0.83]

## MMLU

### gemma-3-4b-it (n_te_flipped=508)
- overall restore = 0.789
- **leak**:    n=329  restore=0.833 [0.79,0.87]
- **no-leak**: n=179  restore=0.709 [0.64,0.78]
  - leak rates: letter-marker=0.29 letter-anywhere=0.31 option-text=0.45
  - no-letter-marker:   n=361  restore=0.751 [0.71,0.80]
  - no-letter-anywhere: n=353  restore=0.748 [0.70,0.79]
  - no-option-text:     n=277  restore=0.762 [0.71,0.81]
  - no-any(generous):   n=177  restore=0.706 [0.64,0.77]

### Llama-3.2-3B-Instruct (n_te_flipped=663)
- overall restore = 0.824
- **leak**:    n=356  restore=0.846 [0.81,0.88]
- **no-leak**: n=307  restore=0.798 [0.75,0.84]
  - leak rates: letter-marker=0.09 letter-anywhere=0.10 option-text=0.49
  - no-letter-marker:   n=604  restore=0.810 [0.78,0.84]
  - no-letter-anywhere: n=598  restore=0.808 [0.78,0.84]
  - no-option-text:     n=338  restore=0.811 [0.77,0.86]
  - no-any(generous):   n=305  restore=0.797 [0.75,0.84]

### Mistral-7B-Instruct-v0.3 (n_te_flipped=543)
- overall restore = 0.812
- **leak**:    n=240  restore=0.850 [0.80,0.89]
- **no-leak**: n=303  restore=0.782 [0.74,0.83]
  - leak rates: letter-marker=0.05 letter-anywhere=0.07 option-text=0.42
  - no-letter-marker:   n=517  restore=0.810 [0.78,0.84]
  - no-letter-anywhere: n=506  restore=0.814 [0.78,0.85]
  - no-option-text:     n=317  restore=0.782 [0.74,0.83]
  - no-any(generous):   n=299  restore=0.786 [0.74,0.83]

### MMLU pooled over models
- overall restore = 0.810 (n=1714)
- leak:    n=925  restore=0.842 [0.82,0.87]
- no-leak: n=789  restore=0.772 [0.74,0.80]
- no-any(generous): n=781  restore=0.772 [0.74,0.80]
