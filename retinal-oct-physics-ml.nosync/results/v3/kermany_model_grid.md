
### all classes (CNV/DME/DRUSEN/NORMAL)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 all + NDT (297) | Gradient boosting | 0.692 ± 0.005 | 0.692 [0.663, 0.709] | 0.691 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.688 ± 0.004 | 0.688 [0.667, 0.715] | 0.687 |
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.687 ± 0.001 | 0.687 [0.663, 0.710] | 0.687 |
| v2 all + NDT (297) | Two-stage (GB) | 0.685 ± 0.006 | 0.685 [0.654, 0.701] | 0.684 |
| v2 band-summary + disease + NDT robust (157) | Two-stage (GB) | 0.683 ± 0.002 | 0.683 [0.662, 0.710] | 0.683 |
| v2 all (243) | Two-stage (GB) | 0.683 ± 0.004 | 0.683 [0.663, 0.710] | 0.683 |
| v2 band-summary + disease (111) | Gradient boosting | 0.683 ± 0.002 | 0.683 [0.655, 0.704] | 0.683 |
| v2 band-summary + disease + NDT (165) | Two-stage (GB) | 0.683 ± 0.000 | 0.683 [0.658, 0.705] | 0.682 |
| v2 all (243) | Gradient boosting | 0.678 ± 0.000 | 0.678 [0.655, 0.702] | 0.677 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.678 ± 0.000 | 0.678 [0.652, 0.701] | 0.678 |
| v2 all + NDT (297) | Logistic regression | 0.677 ± 0.000 | 0.677 [0.653, 0.697] | 0.676 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.676 ± 0.000 | 0.676 [0.653, 0.698] | 0.676 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.676 ± 0.001 | 0.676 [0.651, 0.695] | 0.675 |
| v2 band-summary + disease (111) | Logistic regression | 0.665 ± 0.001 | 0.665 [0.642, 0.688] | 0.664 |
| v2 band-summary + disease (111) | RF (paper config) | 0.653 ± 0.000 | 0.653 [0.627, 0.678] | 0.652 |
| v2 all + NDT (297) | RF (paper config) | 0.653 ± 0.003 | 0.653 [0.629, 0.684] | 0.653 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.651 ± 0.004 | 0.651 [0.628, 0.679] | 0.651 |
| v2 all (243) | Logistic regression | 0.650 ± 0.000 | 0.650 [0.627, 0.671] | 0.649 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.648 ± 0.004 | 0.648 [0.626, 0.679] | 0.648 |
| v2 all (243) | RF (paper config) | 0.643 ± 0.001 | 0.643 [0.615, 0.667] | 0.641 |
| NDT only (54) | Gradient boosting | 0.620 ± 0.002 | 0.620 [0.593, 0.643] | 0.619 |
| NDT only (54) | Two-stage (GB) | 0.615 ± 0.001 | 0.615 [0.590, 0.639] | 0.615 |
| NDT only (54) | RF (paper config) | 0.602 ± 0.003 | 0.602 [0.574, 0.625] | 0.602 |
| NDT robust (46) | Gradient boosting | 0.601 ± 0.000 | 0.601 [0.577, 0.626] | 0.601 |
| NDT robust (46) | Two-stage (GB) | 0.600 ± 0.005 | 0.599 [0.581, 0.628] | 0.599 |
| NDT robust (46) | RF (paper config) | 0.597 ± 0.001 | 0.597 [0.570, 0.621] | 0.597 |
| NDT only (54) | Logistic regression | 0.595 ± 0.004 | 0.595 [0.566, 0.616] | 0.596 |
| NDT robust (46) | Logistic regression | 0.575 ± 0.002 | 0.575 [0.548, 0.598] | 0.575 |

### disease-only (CNV/DME/DRUSEN)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 all + NDT (297) | Gradient boosting | 0.735 ± 0.003 | 0.735 [0.709, 0.766] | 0.735 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.735 ± 0.000 | 0.735 [0.706, 0.760] | 0.734 |
| v2 band-summary + disease (111) | Gradient boosting | 0.727 ± 0.000 | 0.727 [0.701, 0.754] | 0.727 |
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.723 ± 0.001 | 0.723 [0.695, 0.751] | 0.723 |
| v2 all (243) | Gradient boosting | 0.723 ± 0.002 | 0.723 [0.695, 0.750] | 0.723 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.715 ± 0.005 | 0.715 [0.691, 0.746] | 0.715 |
| v2 all + NDT (297) | Logistic regression | 0.710 ± 0.000 | 0.710 [0.682, 0.735] | 0.710 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.707 ± 0.005 | 0.707 [0.684, 0.740] | 0.707 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.702 ± 0.006 | 0.701 [0.680, 0.735] | 0.701 |
| v2 band-summary + disease (111) | RF (paper config) | 0.699 ± 0.001 | 0.699 [0.672, 0.731] | 0.698 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.697 ± 0.001 | 0.697 [0.668, 0.723] | 0.696 |
| v2 all + NDT (297) | RF (paper config) | 0.695 ± 0.009 | 0.695 [0.658, 0.713] | 0.694 |
| v2 all (243) | RF (paper config) | 0.693 ± 0.007 | 0.693 [0.672, 0.729] | 0.692 |
| v2 band-summary + disease (111) | Logistic regression | 0.689 ± 0.001 | 0.689 [0.659, 0.715] | 0.689 |
| v2 all (243) | Logistic regression | 0.688 ± 0.003 | 0.688 [0.665, 0.717] | 0.689 |
| NDT only (54) | Gradient boosting | 0.681 ± 0.001 | 0.681 [0.653, 0.707] | 0.681 |
| NDT only (54) | RF (paper config) | 0.665 ± 0.003 | 0.665 [0.639, 0.697] | 0.665 |
| NDT robust (46) | Gradient boosting | 0.660 ± 0.004 | 0.660 [0.638, 0.692] | 0.660 |
| NDT robust (46) | RF (paper config) | 0.655 ± 0.006 | 0.655 [0.633, 0.690] | 0.654 |
| NDT only (54) | Logistic regression | 0.646 ± 0.005 | 0.646 [0.611, 0.669] | 0.645 |
| NDT robust (46) | Logistic regression | 0.619 ± 0.005 | 0.619 [0.586, 0.642] | 0.618 |
