
### all classes (AMD/DME/ERM/NORMAL/RAO/RVO/VID)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.757 ± 0.001 | 0.689 [0.636, 0.735] | 0.632 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.748 ± 0.002 | 0.678 [0.633, 0.730] | 0.618 |
| v2 all + NDT (297) | Logistic regression | 0.767 ± 0.003 | 0.667 [0.614, 0.727] | 0.627 |
| v2 band-summary + disease (111) | Logistic regression | 0.726 ± 0.002 | 0.661 [0.603, 0.722] | 0.579 |
| v2 all (243) | Logistic regression | 0.740 ± 0.005 | 0.652 [0.599, 0.715] | 0.592 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.788 ± 0.003 | 0.650 [0.588, 0.705] | 0.662 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.784 ± 0.001 | 0.643 [0.576, 0.693] | 0.654 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.838 ± 0.005 | 0.642 [0.583, 0.701] | 0.693 |
| v2 band-summary + disease + NDT (165) | Two-stage (GB) | 0.828 ± 0.005 | 0.633 [0.572, 0.692] | 0.684 |
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.830 ± 0.001 | 0.632 [0.570, 0.687] | 0.686 |
| v2 all + NDT (297) | Gradient boosting | 0.832 ± 0.004 | 0.631 [0.562, 0.674] | 0.687 |
| NDT robust (46) | RF (paper config) | 0.729 ± 0.009 | 0.631 [0.586, 0.687] | 0.612 |
| v2 all + NDT (297) | RF (paper config) | 0.791 ± 0.007 | 0.626 [0.552, 0.665] | 0.647 |
| v2 band-summary + disease + NDT robust (157) | Two-stage (GB) | 0.822 ± 0.002 | 0.624 [0.577, 0.679] | 0.673 |
| v2 all + NDT (297) | Two-stage (GB) | 0.823 ± 0.004 | 0.623 [0.555, 0.667] | 0.679 |
| NDT only (54) | RF (paper config) | 0.730 ± 0.001 | 0.616 [0.565, 0.656] | 0.604 |
| v2 band-summary + disease (111) | RF (paper config) | 0.755 ± 0.004 | 0.609 [0.552, 0.668] | 0.608 |
| v2 band-summary + disease (111) | Gradient boosting | 0.816 ± 0.000 | 0.608 [0.548, 0.662] | 0.662 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.810 ± 0.000 | 0.602 [0.543, 0.658] | 0.650 |
| v2 all (243) | RF (paper config) | 0.772 ± 0.003 | 0.601 [0.532, 0.646] | 0.621 |
| NDT only (54) | Logistic regression | 0.635 ± 0.004 | 0.597 [0.551, 0.658] | 0.511 |
| v2 all (243) | Gradient boosting | 0.812 ± 0.003 | 0.592 [0.527, 0.638] | 0.652 |
| v2 all (243) | Two-stage (GB) | 0.810 ± 0.001 | 0.591 [0.533, 0.646] | 0.654 |
| NDT robust (46) | Logistic regression | 0.587 ± 0.004 | 0.589 [0.544, 0.649] | 0.483 |
| NDT robust (46) | Gradient boosting | 0.786 ± 0.006 | 0.588 [0.549, 0.635] | 0.627 |
| NDT only (54) | Gradient boosting | 0.789 ± 0.001 | 0.586 [0.539, 0.633] | 0.630 |
| NDT robust (46) | Two-stage (GB) | 0.779 ± 0.006 | 0.582 [0.541, 0.627] | 0.622 |
| NDT only (54) | Two-stage (GB) | 0.782 ± 0.004 | 0.580 [0.536, 0.620] | 0.621 |

### disease-only (AMD/DME/ERM/RAO/RVO/VID)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.763 ± 0.002 | 0.685 [0.638, 0.745] | 0.610 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.768 ± 0.001 | 0.685 [0.633, 0.741] | 0.614 |
| v2 all + NDT (297) | Logistic regression | 0.787 ± 0.001 | 0.665 [0.595, 0.724] | 0.619 |
| v2 band-summary + disease (111) | Logistic regression | 0.722 ± 0.000 | 0.648 [0.579, 0.706] | 0.551 |
| v2 all (243) | Logistic regression | 0.752 ± 0.006 | 0.642 [0.583, 0.717] | 0.575 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.809 ± 0.005 | 0.630 [0.570, 0.699] | 0.641 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.807 ± 0.005 | 0.630 [0.568, 0.703] | 0.640 |
| NDT robust (46) | RF (paper config) | 0.767 ± 0.001 | 0.620 [0.561, 0.677] | 0.604 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.844 ± 0.003 | 0.620 [0.553, 0.680] | 0.677 |
| NDT only (54) | RF (paper config) | 0.764 ± 0.004 | 0.614 [0.538, 0.645] | 0.593 |
| v2 all + NDT (297) | RF (paper config) | 0.814 ± 0.000 | 0.613 [0.542, 0.672] | 0.635 |
| NDT only (54) | Logistic regression | 0.648 ± 0.001 | 0.612 [0.541, 0.663] | 0.503 |
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.844 ± 0.001 | 0.605 [0.529, 0.661] | 0.663 |
| v2 band-summary + disease (111) | Gradient boosting | 0.839 ± 0.000 | 0.595 [0.523, 0.653] | 0.659 |
| NDT robust (46) | Logistic regression | 0.598 ± 0.003 | 0.595 [0.529, 0.650] | 0.472 |
| v2 all + NDT (297) | Gradient boosting | 0.840 ± 0.002 | 0.592 [0.525, 0.655] | 0.661 |
| v2 band-summary + disease (111) | RF (paper config) | 0.775 ± 0.007 | 0.582 [0.517, 0.648] | 0.589 |
| NDT robust (46) | Gradient boosting | 0.815 ± 0.005 | 0.581 [0.522, 0.632] | 0.631 |
| v2 all (243) | RF (paper config) | 0.792 ± 0.001 | 0.580 [0.512, 0.644] | 0.603 |
| NDT only (54) | Gradient boosting | 0.813 ± 0.003 | 0.568 [0.502, 0.630] | 0.618 |
| v2 all (243) | Gradient boosting | 0.827 ± 0.005 | 0.554 [0.490, 0.620] | 0.628 |
