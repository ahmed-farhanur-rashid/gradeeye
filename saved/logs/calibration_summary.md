# Per-Threshold Calibration Diagnostic — All Variants

Honest 50/50 5-rep CV: temperature fitted on train half of held-out test, evaluated on test half.

(Re-runs the existing `src.eval.run_calibration_study` analysis on the 3-ch balanced baseline, additively extends it to 3-ch unbalanced and the four 4-ch auxiliary-channel variants.)


## Per-(variant, fold) mean-ECE

| Variant | Fold | N | Raw ECE | Per-Thresh ECE | Global-T ECE | Δ(Per-Raw) | Δ(Glob-Raw) |
|:--------|:-----|--:|--------:|---------------:|-------------:|-----------:|------------:|
| 3ch_balanced       | eyepacs    |  88702 | 0.2710 | 0.2274 | 0.3237 | -0.0437 | +0.0527 |
| 3ch_balanced       | aptos      |   3662 | 0.2172 | 0.2243 | 0.3329 | +0.0071 | +0.1157 |
| 3ch_balanced       | messidor2  |   1744 | 0.1509 | 0.1239 | 0.3066 | -0.0270 | +0.1557 |
| 3ch_balanced       | ddr        |  12522 | 0.1312 | 0.1169 | 0.3075 | -0.0143 | +0.1763 |
| 3ch_unbalanced     | eyepacs    |  88702 | 0.2608 | 0.2744 | 0.3325 | +0.0136 | +0.0717 |
| 3ch_unbalanced     | aptos      |   3662 | 0.1846 | 0.1877 | 0.3166 | +0.0031 | +0.1320 |
| 3ch_unbalanced     | messidor2  |   1744 | 0.2891 | 0.2378 | 0.3433 | -0.0513 | +0.0542 |
| 3ch_unbalanced     | ddr        |  12522 | 0.1418 | 0.1417 | 0.3055 | -0.0000 | +0.1637 |
| 4ch_soft           | eyepacs    |  88702 | 0.1777 | 0.1703 | 0.2678 | -0.0074 | +0.0901 |
| 4ch_soft           | aptos      |   3662 | 0.2247 | 0.2158 | 0.3525 | -0.0089 | +0.1278 |
| 4ch_soft           | messidor2  |   1744 | 0.1883 | 0.1479 | 0.3063 | -0.0404 | +0.1180 |
| 4ch_soft           | ddr        |  12522 | 0.1936 | 0.1833 | 0.3274 | -0.0103 | +0.1338 |
| 4ch_tversky        | eyepacs    |  88702 | 0.3034 | 0.2618 | 0.3475 | -0.0416 | +0.0441 |
| 4ch_tversky        | aptos      |   3662 | 0.1870 | 0.1943 | 0.3107 | +0.0072 | +0.1237 |
| 4ch_tversky        | messidor2  |   1744 | 0.1591 | 0.1442 | 0.3009 | -0.0149 | +0.1418 |
| 4ch_tversky        | ddr        |  12522 | 0.1193 | 0.1285 | 0.2958 | +0.0092 | +0.1766 |
| 4ch_morph          | eyepacs    |  88702 | 0.3852 | 0.2603 | 0.3815 | -0.1249 | -0.0036 |
| 4ch_morph          | aptos      |   3662 | 0.2292 | 0.2264 | 0.3451 | -0.0028 | +0.1159 |
| 4ch_morph          | messidor2  |   1744 | 0.1809 | 0.1494 | 0.3127 | -0.0314 | +0.1318 |
| 4ch_morph          | ddr        |  12522 | 0.1558 | 0.1547 | 0.3105 | -0.0011 | +0.1547 |

## Per-variant mean over folds

| Variant | Raw ECE | Per-Thresh ECE | Global-T ECE | Δ(Per-Raw) | Δ(Glob-Raw) | Per-Temperatures (k=0..3) |
|:--------|--------:|---------------:|-------------:|-----------:|------------:|:--------------------------|
| 3ch_balanced       | 0.1926 | 0.1731 | 0.3177 | -0.0195 | +0.1251 | [ 4.09, 0.79, 0.44, 0.91 ] |
| 3ch_unbalanced     | 0.2190 | 0.2104 | 0.3245 | -0.0087 | +0.1054 | [ 3.25, 2.65, 1.24, 3.66 ] |
| 4ch_soft           | 0.1961 | 0.1793 | 0.3135 | -0.0167 | +0.1174 | [ 2.71, 0.91, 0.63, 1.64 ] |
| 4ch_tversky        | 0.1922 | 0.1822 | 0.3137 | -0.0100 | +0.1215 | [ 5.00, 0.89, 1.08, 0.52 ] |
| 4ch_morph          | 0.2378 | 0.1977 | 0.3374 | -0.0401 | +0.0997 | [ 5.00, 0.69, 0.36, 5.00 ] |