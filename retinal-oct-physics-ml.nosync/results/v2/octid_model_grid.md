
### all classes (CSR/DR/MH/NORMAL)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Gradient boosting | 0.874 ± 0.013 | 0.848 [0.795, 0.864] | 0.848 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.867 ± 0.013 | 0.844 [0.794, 0.866] | 0.841 |
| v2 band-summary + disease (111) | Logistic regression | 0.861 ± 0.006 | 0.839 [0.795, 0.862] | 0.838 |
| v2 texture, no entropy (200) | Two-stage (GB) | 0.856 ± 0.012 | 0.837 [0.783, 0.850] | 0.834 |
| v2 all (243) | Two-stage (GB) | 0.854 ± 0.010 | 0.834 [0.785, 0.855] | 0.830 |
| v2 all (243) | Gradient boosting | 0.858 ± 0.009 | 0.831 [0.787, 0.856] | 0.830 |
| v2 texture (band x sector) (220) | Two-stage (GB) | 0.846 ± 0.008 | 0.826 [0.772, 0.844] | 0.822 |
| v2 texture, no entropy (200) | Gradient boosting | 0.847 ± 0.005 | 0.821 [0.775, 0.845] | 0.820 |
| v2 all (243) | RF (paper config) | 0.843 ± 0.004 | 0.821 [0.778, 0.846] | 0.817 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.846 ± 0.011 | 0.819 [0.765, 0.840] | 0.819 |
| v2 band-summary + disease (111) | RF (paper config) | 0.845 ± 0.005 | 0.818 [0.790, 0.857] | 0.816 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.834 ± 0.010 | 0.815 [0.767, 0.835] | 0.809 |
| v2 all (243) | Logistic regression | 0.842 ± 0.009 | 0.814 [0.765, 0.837] | 0.813 |
| v2 texture, no entropy (200) | RF (paper config) | 0.829 ± 0.007 | 0.810 [0.770, 0.842] | 0.805 |
| v2 texture (band x sector) (220) | Logistic regression | 0.819 ± 0.002 | 0.792 [0.756, 0.829] | 0.790 |
| v2 texture, no entropy (200) | Logistic regression | 0.814 ± 0.004 | 0.785 [0.739, 0.816] | 0.783 |
| v2 disease-specific (23) | Gradient boosting | 0.792 ± 0.013 | 0.755 [0.724, 0.802] | 0.755 |
| v2 disease-specific (23) | Two-stage (GB) | 0.778 ± 0.004 | 0.745 [0.701, 0.782] | 0.742 |
| paper + entropy (450) | Two-stage (GB) | 0.776 ± 0.008 | 0.722 [0.687, 0.765] | 0.723 |
| v2 disease-specific (23) | RF (paper config) | 0.752 ± 0.009 | 0.718 [0.684, 0.765] | 0.716 |
| v2 disease-specific (23) | Logistic regression | 0.745 ± 0.004 | 0.716 [0.682, 0.759] | 0.711 |
| paper + entropy (450) | Gradient boosting | 0.768 ± 0.002 | 0.712 [0.678, 0.755] | 0.712 |
| paper + entropy (450) | RF (paper config) | 0.762 ± 0.022 | 0.704 [0.670, 0.750] | 0.705 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.753 ± 0.009 | 0.694 [0.650, 0.728] | 0.693 |
| paper (300 grid H+S) (300) | Two-stage (GB) | 0.744 ± 0.007 | 0.683 [0.637, 0.721] | 0.683 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.732 ± 0.004 | 0.667 [0.622, 0.701] | 0.669 |
| paper row-summary (40) (40) | Gradient boosting | 0.730 ± 0.004 | 0.664 [0.615, 0.700] | 0.665 |
| paper row-summary (40) (40) | Two-stage (GB) | 0.728 ± 0.007 | 0.664 [0.632, 0.715] | 0.664 |
| paper row-summary (40) (40) | RF (paper config) | 0.722 ± 0.008 | 0.655 [0.601, 0.684] | 0.655 |
| paper + entropy (450) | Logistic regression | 0.718 ± 0.012 | 0.654 [0.594, 0.677] | 0.654 |
| paper (300 grid H+S) (300) | Logistic regression | 0.684 ± 0.011 | 0.613 [0.558, 0.638] | 0.613 |
| paper row-summary (40) (40) | Logistic regression | 0.649 ± 0.014 | 0.570 [0.535, 0.617] | 0.570 |

### disease-only (CSR/DR/MH)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Logistic regression | 0.832 ± 0.011 | 0.832 [0.785, 0.870] | 0.832 |
| v2 band-summary + disease (111) | Gradient boosting | 0.829 ± 0.007 | 0.828 [0.791, 0.874] | 0.827 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.805 ± 0.004 | 0.804 [0.755, 0.840] | 0.804 |
| v2 band-summary + disease (111) | RF (paper config) | 0.803 ± 0.015 | 0.802 [0.752, 0.841] | 0.801 |
| v2 texture, no entropy (200) | Gradient boosting | 0.797 ± 0.005 | 0.797 [0.750, 0.835] | 0.796 |
| v2 all (243) | Gradient boosting | 0.797 ± 0.009 | 0.797 [0.759, 0.846] | 0.797 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.783 ± 0.003 | 0.784 [0.743, 0.831] | 0.783 |
| v2 texture, no entropy (200) | RF (paper config) | 0.783 ± 0.003 | 0.783 [0.741, 0.832] | 0.783 |
| v2 all (243) | Logistic regression | 0.780 ± 0.004 | 0.780 [0.727, 0.818] | 0.780 |
| v2 all (243) | RF (paper config) | 0.778 ± 0.003 | 0.778 [0.730, 0.821] | 0.777 |
| v2 texture (band x sector) (220) | Logistic regression | 0.760 ± 0.011 | 0.760 [0.694, 0.795] | 0.760 |
| v2 texture, no entropy (200) | Logistic regression | 0.756 ± 0.008 | 0.756 [0.695, 0.794] | 0.755 |
| v2 disease-specific (23) | Gradient boosting | 0.701 ± 0.005 | 0.702 [0.654, 0.751] | 0.700 |
| v2 disease-specific (23) | Logistic regression | 0.695 ± 0.007 | 0.696 [0.639, 0.743] | 0.694 |
| v2 disease-specific (23) | RF (paper config) | 0.689 ± 0.004 | 0.689 [0.634, 0.735] | 0.687 |
| paper + entropy (450) | Gradient boosting | 0.656 ± 0.023 | 0.657 [0.571, 0.680] | 0.658 |
| paper + entropy (450) | RF (paper config) | 0.591 ± 0.013 | 0.592 [0.535, 0.643] | 0.593 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.586 ± 0.011 | 0.587 [0.518, 0.626] | 0.588 |
| paper row-summary (40) (40) | Gradient boosting | 0.562 ± 0.008 | 0.563 [0.496, 0.606] | 0.562 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.553 ± 0.021 | 0.554 [0.518, 0.625] | 0.557 |
| paper + entropy (450) | Logistic regression | 0.548 ± 0.013 | 0.548 [0.495, 0.608] | 0.549 |
| paper row-summary (40) (40) | RF (paper config) | 0.542 ± 0.008 | 0.544 [0.494, 0.604] | 0.543 |
| paper (300 grid H+S) (300) | Logistic regression | 0.536 ± 0.017 | 0.536 [0.500, 0.617] | 0.536 |
| paper row-summary (40) (40) | Logistic regression | 0.458 ± 0.011 | 0.458 [0.391, 0.504] | 0.456 |

### disease-only, 500x750 scans only (no size confound)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Logistic regression | 0.830 ± 0.024 | 0.836 [0.763, 0.854] | 0.832 |
| v2 band-summary + disease (111) | Gradient boosting | 0.815 ± 0.007 | 0.819 [0.778, 0.868] | 0.816 |
| v2 all (243) | Gradient boosting | 0.801 ± 0.017 | 0.808 [0.766, 0.857] | 0.804 |
| v2 all (243) | Logistic regression | 0.800 ± 0.017 | 0.806 [0.761, 0.856] | 0.803 |
| v2 texture, no entropy (200) | Gradient boosting | 0.798 ± 0.024 | 0.801 [0.775, 0.865] | 0.798 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.798 ± 0.018 | 0.801 [0.770, 0.859] | 0.797 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.789 ± 0.009 | 0.796 [0.760, 0.857] | 0.789 |
| v2 all (243) | RF (paper config) | 0.786 ± 0.015 | 0.794 [0.762, 0.858] | 0.786 |
| v2 texture, no entropy (200) | RF (paper config) | 0.786 ± 0.014 | 0.794 [0.761, 0.854] | 0.787 |
| v2 band-summary + disease (111) | RF (paper config) | 0.786 ± 0.006 | 0.791 [0.746, 0.839] | 0.784 |
| v2 texture (band x sector) (220) | Logistic regression | 0.784 ± 0.012 | 0.787 [0.743, 0.839] | 0.784 |
| v2 texture, no entropy (200) | Logistic regression | 0.782 ± 0.006 | 0.785 [0.736, 0.835] | 0.781 |
| v2 disease-specific (23) | Logistic regression | 0.692 ± 0.012 | 0.705 [0.641, 0.747] | 0.699 |
| v2 disease-specific (23) | Gradient boosting | 0.685 ± 0.006 | 0.695 [0.631, 0.741] | 0.690 |
| v2 disease-specific (23) | RF (paper config) | 0.673 ± 0.003 | 0.685 [0.625, 0.733] | 0.678 |
| paper + entropy (450) | Gradient boosting | 0.569 ± 0.019 | 0.565 [0.493, 0.615] | 0.570 |
| paper + entropy (450) | RF (paper config) | 0.559 ± 0.017 | 0.561 [0.536, 0.649] | 0.559 |
| paper row-summary (40) (40) | Gradient boosting | 0.539 ± 0.015 | 0.542 [0.500, 0.620] | 0.546 |
| paper row-summary (40) (40) | RF (paper config) | 0.513 ± 0.005 | 0.526 [0.465, 0.586] | 0.520 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.531 ± 0.009 | 0.521 [0.470, 0.593] | 0.525 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.502 ± 0.026 | 0.504 [0.478, 0.592] | 0.503 |
| paper + entropy (450) | Logistic regression | 0.495 ± 0.018 | 0.490 [0.405, 0.525] | 0.489 |
| paper (300 grid H+S) (300) | Logistic regression | 0.477 ± 0.035 | 0.472 [0.374, 0.494] | 0.470 |
| paper row-summary (40) (40) | Logistic regression | 0.436 ± 0.020 | 0.443 [0.406, 0.523] | 0.436 |
