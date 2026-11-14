
### all classes (CNV/DME/DRUSEN/NORMAL)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 all (243) | Two-stage (GB) | 0.683 ± 0.004 | 0.683 [0.663, 0.710] | 0.683 |
| v2 band-summary + disease (111) | Gradient boosting | 0.683 ± 0.002 | 0.683 [0.655, 0.704] | 0.683 |
| v2 all (243) | Gradient boosting | 0.678 ± 0.000 | 0.678 [0.655, 0.702] | 0.677 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.678 ± 0.000 | 0.678 [0.652, 0.701] | 0.678 |
| v2 band-summary + disease (111) | Logistic regression | 0.665 ± 0.001 | 0.665 [0.642, 0.688] | 0.664 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.665 ± 0.005 | 0.664 [0.633, 0.683] | 0.662 |
| v2 texture, no entropy (200) | Two-stage (GB) | 0.661 ± 0.001 | 0.660 [0.634, 0.682] | 0.659 |
| v2 texture (band x sector) (220) | Two-stage (GB) | 0.659 ± 0.001 | 0.659 [0.633, 0.680] | 0.658 |
| v2 texture, no entropy (200) | Gradient boosting | 0.658 ± 0.007 | 0.658 [0.629, 0.674] | 0.657 |
| v2 band-summary + disease (111) | RF (paper config) | 0.653 ± 0.000 | 0.653 [0.627, 0.678] | 0.652 |
| v2 all (243) | Logistic regression | 0.650 ± 0.000 | 0.650 [0.627, 0.671] | 0.649 |
| v2 texture (band x sector) (220) | Logistic regression | 0.645 ± 0.003 | 0.645 [0.618, 0.664] | 0.644 |
| v2 all (243) | RF (paper config) | 0.643 ± 0.001 | 0.643 [0.615, 0.667] | 0.641 |
| v2 texture, no entropy (200) | Logistic regression | 0.636 ± 0.003 | 0.636 [0.611, 0.654] | 0.635 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.625 ± 0.006 | 0.625 [0.607, 0.655] | 0.622 |
| v2 texture, no entropy (200) | RF (paper config) | 0.622 ± 0.001 | 0.622 [0.595, 0.644] | 0.619 |
| v2 disease-specific (23) | Gradient boosting | 0.554 ± 0.002 | 0.554 [0.530, 0.577] | 0.555 |
| v2 disease-specific (23) | Two-stage (GB) | 0.546 ± 0.007 | 0.546 [0.526, 0.577] | 0.545 |
| v2 disease-specific (23) | RF (paper config) | 0.527 ± 0.001 | 0.527 [0.500, 0.551] | 0.527 |
| paper + entropy (450) | Two-stage (GB) | 0.501 ± 0.002 | 0.501 [0.473, 0.522] | 0.501 |
| paper + entropy (450) | Gradient boosting | 0.499 ± 0.007 | 0.499 [0.468, 0.515] | 0.500 |
| v2 disease-specific (23) | Logistic regression | 0.494 ± 0.001 | 0.494 [0.467, 0.520] | 0.494 |
| paper (300 grid H+S) (300) | Two-stage (GB) | 0.478 ± 0.002 | 0.479 [0.457, 0.504] | 0.476 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.473 ± 0.003 | 0.473 [0.445, 0.493] | 0.473 |
| paper + entropy (450) | RF (paper config) | 0.452 ± 0.005 | 0.452 [0.432, 0.483] | 0.452 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.449 ± 0.001 | 0.449 [0.425, 0.477] | 0.449 |
| paper row-summary (40) (40) | RF (paper config) | 0.432 ± 0.002 | 0.432 [0.411, 0.458] | 0.431 |
| paper row-summary (40) (40) | Gradient boosting | 0.427 ± 0.004 | 0.427 [0.400, 0.445] | 0.427 |
| paper row-summary (40) (40) | Two-stage (GB) | 0.424 ± 0.004 | 0.424 [0.405, 0.451] | 0.422 |
| paper + entropy (450) | Logistic regression | 0.414 ± 0.000 | 0.414 [0.390, 0.436] | 0.414 |
| paper (300 grid H+S) (300) | Logistic regression | 0.397 ± 0.000 | 0.397 [0.376, 0.420] | 0.397 |
| paper row-summary (40) (40) | Logistic regression | 0.357 ± 0.007 | 0.357 [0.341, 0.386] | 0.355 |

### disease-only (CNV/DME/DRUSEN)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Gradient boosting | 0.727 ± 0.000 | 0.727 [0.701, 0.754] | 0.727 |
| v2 all (243) | Gradient boosting | 0.723 ± 0.002 | 0.723 [0.695, 0.750] | 0.723 |
| v2 texture, no entropy (200) | Gradient boosting | 0.702 ± 0.000 | 0.701 [0.673, 0.727] | 0.702 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.700 ± 0.002 | 0.700 [0.674, 0.727] | 0.700 |
| v2 band-summary + disease (111) | RF (paper config) | 0.699 ± 0.001 | 0.699 [0.672, 0.731] | 0.698 |
| v2 all (243) | RF (paper config) | 0.693 ± 0.007 | 0.693 [0.672, 0.729] | 0.692 |
| v2 band-summary + disease (111) | Logistic regression | 0.689 ± 0.001 | 0.689 [0.659, 0.715] | 0.689 |
| v2 all (243) | Logistic regression | 0.688 ± 0.003 | 0.688 [0.665, 0.717] | 0.689 |
| v2 texture (band x sector) (220) | Logistic regression | 0.666 ± 0.001 | 0.666 [0.637, 0.688] | 0.666 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.660 ± 0.000 | 0.660 [0.631, 0.689] | 0.660 |
| v2 texture, no entropy (200) | RF (paper config) | 0.657 ± 0.003 | 0.657 [0.625, 0.682] | 0.657 |
| v2 texture, no entropy (200) | Logistic regression | 0.654 ± 0.001 | 0.654 [0.625, 0.678] | 0.654 |
| v2 disease-specific (23) | RF (paper config) | 0.650 ± 0.005 | 0.650 [0.626, 0.684] | 0.648 |
| v2 disease-specific (23) | Gradient boosting | 0.642 ± 0.005 | 0.642 [0.617, 0.674] | 0.642 |
| v2 disease-specific (23) | Logistic regression | 0.599 ± 0.002 | 0.599 [0.567, 0.626] | 0.596 |
| paper + entropy (450) | Gradient boosting | 0.597 ± 0.001 | 0.597 [0.571, 0.625] | 0.597 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.596 ± 0.001 | 0.596 [0.571, 0.625] | 0.596 |
| paper + entropy (450) | RF (paper config) | 0.561 ± 0.008 | 0.561 [0.526, 0.583] | 0.560 |
| paper row-summary (40) (40) | Gradient boosting | 0.554 ± 0.001 | 0.555 [0.525, 0.581] | 0.555 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.551 ± 0.006 | 0.551 [0.519, 0.573] | 0.551 |
| paper row-summary (40) (40) | RF (paper config) | 0.538 ± 0.008 | 0.538 [0.502, 0.558] | 0.537 |
| paper + entropy (450) | Logistic regression | 0.487 ± 0.001 | 0.487 [0.463, 0.515] | 0.487 |
| paper (300 grid H+S) (300) | Logistic regression | 0.464 ± 0.001 | 0.464 [0.440, 0.492] | 0.464 |
| paper row-summary (40) (40) | Logistic regression | 0.459 ± 0.001 | 0.459 [0.434, 0.486] | 0.456 |
