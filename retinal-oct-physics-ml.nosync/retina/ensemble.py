"""v5 model zoo: the repo's LR and LightGBM plus an RBF-SVM, combined by soft voting.

Soft voting averages the class probabilities of M different models:

    p̂(k | x) = (1/M) Σ_m p̂_m(k | x),      ŷ = argmax_k p̂(k | x)

The three members make different kinds of mistakes (linear boundary, axis-aligned tree splits,
smooth kernel boundary), so averaging cancels part of each one's variance. No inner CV is
needed (unlike stacking), so patient grouping cannot leak.
"""
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def lr(seed=0, C=0.1):
    """Same as retina.evaluation.logreg (C=0.1 was confirmed near-optimal by a 0.01-1.0 sweep)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def gbm(seed=0, n_jobs=1):
    """Same hyper-parameters as retina.evaluation.hgb; n_jobs=1 avoids thread over-subscription
    on 2-4 core laptops (5x faster there, identical predictions)."""
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                          class_weight="balanced", random_state=seed, n_jobs=n_jobs, verbose=-1)


def svm(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed))


def soft_vote(seed=0):
    return VotingClassifier([("lr", lr(seed)), ("gbm", gbm(seed)), ("svm", svm(seed))], voting="soft")


V5_MODELS = {"Logistic regression": lr, "LightGBM": gbm, "Soft vote (LR+GBM+SVM)": soft_vote}
