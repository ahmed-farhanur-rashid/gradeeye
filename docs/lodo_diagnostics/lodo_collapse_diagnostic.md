# LODO Collapse Diagnostic

ConvNeXt-Tiny, 3 folds, single checkpoint per fold evaluated on test_full + test_matched.

## Class distributions (% per grade)

| Fold | Pool dist (train+val) | test_full dist | test_matched dist | matched→full sample pct (per class) |
|------|----------------------|----------------|-------------------|--------------------------------------|
| eyepacs | 0:52.2, 1:11.8, 2:24.9, 3:5.0, 4:6.1 | 0:73.7, 1:7.0, 2:14.8, 3:2.4, 4:2.2 | 0:73.6, 1:7.0, 2:15.0, 3:2.2, 4:2.2 | 0:0.3, 1:0.3, 2:0.3, 3:0.2, 4:0.3 |
| aptos | 0:73.4, 1:7.2, 2:14.9, 3:2.4, 4:2.2 | 0:49.3, 1:10.1, 2:27.3, 3:5.3, 4:8.1 | 0:49.3, 1:10.1, 2:27.3, 3:5.3, 4:7.9 | 0:6.2, 1:6.2, 2:6.2, 3:6.2, 4:6.1 |
| messidor2 | 0:72.7, 1:7.1, 2:15.3, 3:2.5, 4:2.4 | 0:58.3, 1:15.5, 2:19.9, 3:4.3, 4:2.0 | 0:58.1, 1:15.4, 2:19.8, 3:4.4, 4:2.2 | 0:13.0, 1:13.0, 2:13.0, 3:13.3, 4:14.3 |

## Metrics on held-out test

| Fold | split | n | QWK | Acc | MacroF1 | MacroRecall |
|------|-------|---|------|------|---------|-------------|
| eyepacs | test_full | 88702 | 0.2423 | 0.6983 | 0.2854 | 0.2729 |
| eyepacs | test_matched | 227 | 0.1204 | 0.6916 | 0.1981 | 0.2219 |
| aptos | test_full | 3662 | 0.5496 | 0.2731 | 0.2475 | 0.3375 |
| aptos | test_matched | 227 | 0.5037 | 0.2247 | 0.2075 | 0.3095 |
| messidor2 | test_full | 1744 | 0.6324 | 0.6055 | 0.4993 | 0.5634 |
| messidor2 | test_matched | 227 | 0.5722 | 0.5683 | 0.4329 | 0.5155 |

## Confusion matrices (rows=true, cols=pred)


### eyepacs / test_full  (n=88702)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 60664 | 4403 | 50 | 116 | 110 |
| **Mild** | 5679 | 507 | 5 | 7 | 7 |
| **Mod** | 10939 | 1866 | 66 | 215 | 67 |
| **Sev** | 1230 | 408 | 29 | 387 | 33 |
| **PDR** | 1059 | 211 | 29 | 301 | 314 |

### eyepacs / test_matched  (n=227)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 154 | 12 | 0 | 1 | 0 |
| **Mild** | 13 | 3 | 0 | 0 | 0 |
| **Mod** | 30 | 4 | 0 | 0 | 0 |
| **Sev** | 2 | 3 | 0 | 0 | 0 |
| **PDR** | 4 | 0 | 0 | 1 | 0 |

### aptos / test_full  (n=3662)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 629 | 483 | 498 | 188 | 7 |
| **Mild** | 1 | 3 | 118 | 229 | 19 |
| **Mod** | 1 | 0 | 76 | 791 | 131 |
| **Sev** | 0 | 0 | 1 | 148 | 44 |
| **PDR** | 1 | 0 | 7 | 143 | 144 |

### aptos / test_matched  (n=227)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 31 | 28 | 43 | 9 | 1 |
| **Mild** | 0 | 0 | 4 | 18 | 1 |
| **Mod** | 0 | 0 | 3 | 50 | 9 |
| **Sev** | 0 | 0 | 0 | 10 | 2 |
| **PDR** | 0 | 0 | 0 | 11 | 7 |

### messidor2 / test_full  (n=1744)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 735 | 122 | 147 | 1 | 12 |
| **Mild** | 150 | 69 | 47 | 1 | 3 |
| **Mod** | 55 | 42 | 182 | 48 | 20 |
| **Sev** | 0 | 0 | 19 | 45 | 11 |
| **PDR** | 0 | 0 | 4 | 6 | 25 |

### messidor2 / test_matched  (n=227)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|---|---|---|---|---|---|
| **NoDR** | 93 | 16 | 19 | 1 | 3 |
| **Mild** | 17 | 8 | 8 | 0 | 2 |
| **Mod** | 8 | 6 | 20 | 6 | 5 |
| **Sev** | 0 | 0 | 4 | 4 | 2 |
| **PDR** | 0 | 0 | 1 | 0 | 4 |

## Per-class recall (test_full vs test_matched)

| Fold | split | NoDR | Mild | Mod | Sev | PDR |
|------|-------|------|------|------|------|------|
| eyepacs | test_full | 0.928 | 0.082 | 0.005 | 0.185 | 0.164 |
| eyepacs | test_matched | 0.922 | 0.188 | 0.000 | 0.000 | 0.000 |
| aptos | test_full | 0.348 | 0.008 | 0.076 | 0.767 | 0.488 |
| aptos | test_matched | 0.277 | 0.000 | 0.048 | 0.833 | 0.389 |
| messidor2 | test_full | 0.723 | 0.256 | 0.524 | 0.600 | 0.714 |
| messidor2 | test_matched | 0.705 | 0.229 | 0.444 | 0.400 | 0.800 |