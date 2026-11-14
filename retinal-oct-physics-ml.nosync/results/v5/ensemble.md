| Task | Model | Balanced acc [95% CI] | Accuracy | Macro-F1 |
|---|---|---|---|---|
| OCTID CSR/DR/MH (500x750) | Logistic regression | 0.868 [0.828, 0.905] | 0.863 | 0.865 |
| OCTID CSR/DR/MH (500x750) | LightGBM | 0.838 [0.788, 0.876] | 0.835 | 0.835 |
| OCTID CSR/DR/MH (500x750) | Soft vote (LR+GBM+SVM) | 0.872 [0.833, 0.911] | 0.868 | 0.869 |
| OCTID 4-class (scanner-confounded) | Logistic regression | 0.843 [0.818, 0.886] | 0.864 | 0.840 |
| OCTID 4-class (scanner-confounded) | LightGBM | 0.864 [0.818, 0.887] | 0.890 | 0.866 |
| OCTID 4-class (scanner-confounded) | Soft vote (LR+GBM+SVM) | 0.870 [0.840, 0.902] | 0.890 | 0.869 |
| OCTDL 7-class | Logistic regression | 0.682 [0.639, 0.734] | 0.753 | 0.617 |
| OCTDL 7-class | LightGBM | 0.644 [0.588, 0.690] | 0.834 | 0.693 |
| OCTDL 7-class | Soft vote (LR+GBM+SVM) | 0.669 [0.605, 0.723] | 0.834 | 0.699 |
| OCTDL 6 diseases | Logistic regression | 0.674 [0.632, 0.737] | 0.764 | 0.601 |
| OCTDL 6 diseases | LightGBM | 0.615 [0.529, 0.657] | 0.845 | 0.675 |
| OCTDL 6 diseases | Soft vote (LR+GBM+SVM) | 0.653 [0.589, 0.722] | 0.852 | 0.688 |
| Kermany 4-class | Logistic regression | 0.677 [0.658, 0.702] | 0.677 | 0.676 |
| Kermany 4-class | LightGBM | 0.680 [0.659, 0.706] | 0.680 | 0.679 |
| Kermany 4-class | Soft vote (LR+GBM+SVM) | 0.695 [0.671, 0.718] | 0.696 | 0.694 |
| Kermany CNV/DME/DRUSEN | Logistic regression | 0.708 [0.682, 0.736] | 0.708 | 0.708 |
| Kermany CNV/DME/DRUSEN | LightGBM | 0.726 [0.700, 0.758] | 0.726 | 0.726 |
| Kermany CNV/DME/DRUSEN | Soft vote (LR+GBM+SVM) | 0.731 [0.706, 0.765] | 0.731 | 0.731 |
