
### all classes (CSR/DR/MH/NORMAL)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.893 ± 0.003 | 0.870 [0.831, 0.898] | 0.871 |
| v2 band-summary + disease + NDT robust (157) | Two-stage (GB) | 0.887 ± 0.009 | 0.870 [0.826, 0.891] | 0.868 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.890 ± 0.000 | 0.867 [0.832, 0.897] | 0.867 |
| v2 band-summary + disease + NDT (165) | Two-stage (GB) | 0.884 ± 0.002 | 0.866 [0.829, 0.894] | 0.864 |
| v2 all + NDT (297) | Two-stage (GB) | 0.872 ± 0.014 | 0.855 [0.804, 0.872] | 0.853 |
| v2 all + NDT (297) | Gradient boosting | 0.881 ± 0.011 | 0.855 [0.806, 0.877] | 0.856 |
| v2 band-summary + disease (111) | Gradient boosting | 0.876 ± 0.015 | 0.852 [0.795, 0.864] | 0.851 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.869 ± 0.003 | 0.851 [0.813, 0.879] | 0.847 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.869 ± 0.015 | 0.848 [0.794, 0.866] | 0.845 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.863 ± 0.002 | 0.843 [0.811, 0.879] | 0.840 |
| v2 all + NDT (297) | Logistic regression | 0.865 ± 0.004 | 0.841 [0.801, 0.868] | 0.840 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.863 ± 0.006 | 0.840 [0.797, 0.866] | 0.837 |
| v2 band-summary + disease (111) | Logistic regression | 0.861 ± 0.008 | 0.839 [0.795, 0.862] | 0.837 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.860 ± 0.005 | 0.836 [0.807, 0.873] | 0.834 |
| v2 all (243) | Gradient boosting | 0.861 ± 0.010 | 0.834 [0.787, 0.856] | 0.834 |
| v2 all (243) | Two-stage (GB) | 0.853 ± 0.012 | 0.834 [0.785, 0.855] | 0.830 |
| v2 all + NDT (297) | RF (paper config) | 0.843 ± 0.010 | 0.820 [0.774, 0.842] | 0.817 |
| v2 band-summary + disease (111) | RF (paper config) | 0.845 ± 0.006 | 0.818 [0.790, 0.857] | 0.816 |
| v2 all (243) | RF (paper config) | 0.840 ± 0.003 | 0.818 [0.778, 0.846] | 0.814 |
| v2 all (243) | Logistic regression | 0.839 ± 0.010 | 0.812 [0.765, 0.837] | 0.811 |
| NDT only (54) | Gradient boosting | 0.831 ± 0.001 | 0.800 [0.763, 0.836] | 0.802 |
| NDT only (54) | Two-stage (GB) | 0.822 ± 0.002 | 0.795 [0.763, 0.835] | 0.794 |
| NDT only (54) | RF (paper config) | 0.817 ± 0.009 | 0.790 [0.743, 0.818] | 0.789 |
| NDT robust (46) | Gradient boosting | 0.806 ± 0.003 | 0.772 [0.726, 0.805] | 0.774 |
| NDT robust (46) | Two-stage (GB) | 0.792 ± 0.003 | 0.760 [0.713, 0.795] | 0.760 |
| NDT only (54) | Logistic regression | 0.777 ± 0.003 | 0.756 [0.721, 0.801] | 0.750 |
| NDT robust (46) | RF (paper config) | 0.762 ± 0.002 | 0.731 [0.690, 0.772] | 0.730 |
| NDT robust (46) | Logistic regression | 0.705 ± 0.001 | 0.681 [0.638, 0.724] | 0.676 |

### disease-only (CSR/DR/MH)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.855 ± 0.000 | 0.855 [0.811, 0.896] | 0.855 |
| v2 all + NDT (297) | Gradient boosting | 0.841 ± 0.002 | 0.840 [0.798, 0.877] | 0.841 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.834 ± 0.005 | 0.834 [0.795, 0.879] | 0.834 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.833 ± 0.006 | 0.833 [0.784, 0.868] | 0.833 |
| v2 band-summary + disease (111) | Gradient boosting | 0.833 ± 0.003 | 0.832 [0.791, 0.874] | 0.832 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.830 ± 0.006 | 0.829 [0.794, 0.875] | 0.830 |
| v2 band-summary + disease (111) | Logistic regression | 0.825 ± 0.005 | 0.825 [0.785, 0.870] | 0.825 |
| v2 all + NDT (297) | Logistic regression | 0.820 ± 0.010 | 0.820 [0.765, 0.852] | 0.820 |
| v2 all + NDT (297) | RF (paper config) | 0.818 ± 0.002 | 0.818 [0.775, 0.861] | 0.818 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.817 ± 0.003 | 0.816 [0.769, 0.857] | 0.816 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.817 ± 0.006 | 0.816 [0.765, 0.854] | 0.816 |
| v2 band-summary + disease (111) | RF (paper config) | 0.810 ± 0.013 | 0.809 [0.752, 0.841] | 0.809 |
| v2 all (243) | Gradient boosting | 0.804 ± 0.000 | 0.803 [0.759, 0.846] | 0.803 |
| v2 all (243) | Logistic regression | 0.780 ± 0.005 | 0.779 [0.727, 0.818] | 0.780 |
| v2 all (243) | RF (paper config) | 0.780 ± 0.002 | 0.779 [0.730, 0.821] | 0.779 |
| NDT only (54) | Gradient boosting | 0.762 ± 0.010 | 0.763 [0.726, 0.816] | 0.763 |
| NDT robust (46) | Gradient boosting | 0.746 ± 0.003 | 0.747 [0.697, 0.793] | 0.747 |
| NDT only (54) | Logistic regression | 0.746 ± 0.000 | 0.746 [0.699, 0.792] | 0.746 |
| NDT only (54) | RF (paper config) | 0.735 ± 0.005 | 0.735 [0.680, 0.779] | 0.736 |
| NDT robust (46) | RF (paper config) | 0.706 ± 0.008 | 0.707 [0.665, 0.764] | 0.707 |
| NDT robust (46) | Logistic regression | 0.693 ± 0.011 | 0.694 [0.655, 0.756] | 0.694 |

### disease-only, 500x750 scans only (no size confound)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease + NDT (165) | Gradient boosting | 0.853 ± 0.000 | 0.855 [0.815, 0.897] | 0.853 |
| v2 band-summary + disease + NDT robust (157) | Logistic regression | 0.844 ± 0.002 | 0.849 [0.808, 0.892] | 0.847 |
| v2 band-summary + disease + NDT (165) | Logistic regression | 0.842 ± 0.004 | 0.847 [0.802, 0.882] | 0.845 |
| v2 band-summary + disease + NDT robust (157) | Gradient boosting | 0.844 ± 0.006 | 0.846 [0.814, 0.893] | 0.844 |
| v2 all + NDT (297) | Logistic regression | 0.827 ± 0.015 | 0.831 [0.800, 0.883] | 0.829 |
| v2 all + NDT (297) | Gradient boosting | 0.825 ± 0.006 | 0.828 [0.779, 0.871] | 0.827 |
| v2 band-summary + disease + NDT robust (157) | RF (paper config) | 0.822 ± 0.009 | 0.826 [0.791, 0.881] | 0.821 |
| v2 band-summary + disease + NDT (165) | RF (paper config) | 0.820 ± 0.004 | 0.822 [0.773, 0.864] | 0.817 |
| v2 band-summary + disease (111) | Logistic regression | 0.814 ± 0.013 | 0.822 [0.763, 0.854] | 0.818 |
| v2 all + NDT (297) | RF (paper config) | 0.811 ± 0.017 | 0.815 [0.790, 0.879] | 0.810 |
| v2 band-summary + disease (111) | Gradient boosting | 0.812 ± 0.007 | 0.815 [0.778, 0.868] | 0.812 |
| v2 all (243) | Gradient boosting | 0.792 ± 0.013 | 0.798 [0.766, 0.857] | 0.796 |
| v2 all (243) | Logistic regression | 0.790 ± 0.011 | 0.797 [0.761, 0.856] | 0.794 |
| v2 band-summary + disease (111) | RF (paper config) | 0.789 ± 0.006 | 0.794 [0.746, 0.839] | 0.787 |
| v2 all (243) | RF (paper config) | 0.783 ± 0.018 | 0.790 [0.762, 0.858] | 0.783 |
| NDT only (54) | Gradient boosting | 0.770 ± 0.013 | 0.764 [0.698, 0.803] | 0.766 |
| NDT only (54) | RF (paper config) | 0.752 ± 0.009 | 0.754 [0.711, 0.814] | 0.752 |
| NDT only (54) | Logistic regression | 0.748 ± 0.009 | 0.750 [0.686, 0.790] | 0.747 |
| NDT robust (46) | Gradient boosting | 0.752 ± 0.006 | 0.744 [0.686, 0.794] | 0.747 |
| NDT robust (46) | Logistic regression | 0.722 ± 0.013 | 0.723 [0.653, 0.763] | 0.718 |
| NDT robust (46) | RF (paper config) | 0.702 ± 0.004 | 0.707 [0.657, 0.765] | 0.701 |
