import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "classifier_results.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(FEATURES_PATH)
print(f"Loaded {len(df)} songs")

# --- Period labels ---
PERIOD_MAP = {
    1970: "early", 1975: "early", 1978: "early", 1979: "early", 1984: "early",
    1988: "middle", 1992: "middle", 1996: "middle", 2002: "middle",
    2007: "late", 2012: "late", 2016: "late", 2025: "late",
}
df["period"] = df["year"].map(PERIOD_MAP)

FEATURE_COLS = [
    "ttr", "mtld", "avg_word_length", "avg_line_length", "vocab_size",
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
    "pos_ADP", "pos_CCONJ", "pos_SCONJ",
    "verb_noun_ratio", "function_word_ratio", "first_person_ratio",
    "num_lines", "num_stanzas", "repetition_rate",
]
LEXICAL_COLS = ["ttr", "mtld", "avg_word_length", "avg_line_length", "vocab_size"]
SYNTACTIC_COLS = [
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
    "pos_ADP", "pos_CCONJ", "pos_SCONJ",
    "verb_noun_ratio", "function_word_ratio", "first_person_ratio",
]

X_all = df[FEATURE_COLS].values
X_lex = df[LEXICAL_COLS].values
X_syn = df[SYNTACTIC_COLS].values
y = df["period"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Cross-validation ---
def run_cv(X, model, cv, label):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  {label}: {scores.mean():.3f} ± {scores.std():.3f}  (folds: {np.round(scores, 3)})")
    return {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}

print("\n=== Cross-validation (5-fold, stratified) ===")
cv_scores = {
    "lr_all":       run_cv(X_all, LogisticRegression(max_iter=1000, random_state=42), cv, "LR (all features)"),
    "lr_lexical":   run_cv(X_lex, LogisticRegression(max_iter=1000, random_state=42), cv, "LR (lexical only)"),
    "lr_syntactic": run_cv(X_syn, LogisticRegression(max_iter=1000, random_state=42), cv, "LR (syntactic only)"),
    "svm_all":      run_cv(X_all, SVC(kernel="rbf", random_state=42), cv, "SVM-RBF (all features)"),
    "svm_lexical":   run_cv(X_lex, SVC(kernel="rbf", random_state=42), cv, "SVM-RBF (lexical only)"),
    "svm_syntactic": run_cv(X_syn, SVC(kernel="rbf", random_state=42), cv, "SVM-RBF (syntactic only)"),
    # k-NN: direct supervised test of Hay et al. (2020) "Representation Learning of
    # Writing Style" — period is predicted from proximity in the same stylometric
    # feature space used elsewhere in Track A (not a separate embedding).
    "knn_all":       run_cv(X_all, KNeighborsClassifier(n_neighbors=5), cv, "KNN (all features)"),
    "knn_lexical":   run_cv(X_lex, KNeighborsClassifier(n_neighbors=5), cv, "KNN (lexical only)"),
    "knn_syntactic": run_cv(X_syn, KNeighborsClassifier(n_neighbors=5), cv, "KNN (syntactic only)"),
    "rf_all":        run_cv(X_all, RandomForestClassifier(n_estimators=200, random_state=42), cv, "Random Forest (all features)"),
}

# --- Feature importance from LR trained on full data ---
print("\n=== Feature importance (LR on full data) ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
lr_full = LogisticRegression(max_iter=1000, random_state=42)
lr_full.fit(X_scaled, y)

mean_abs_coef = np.mean(np.abs(lr_full.coef_), axis=0)
feature_importance = sorted(
    zip(FEATURE_COLS, mean_abs_coef.tolist()),
    key=lambda x: -x[1]
)
top_features = [{"feature": f, "coefficient": round(c, 4)} for f, c in feature_importance]
print("Top 10 features:")
for item in top_features[:10]:
    print(f"  {item['feature']:25s}  coef={item['coefficient']:.4f}")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(9, 5))
top10 = top_features[:10]
names = [t["feature"] for t in top10]
coefs = [t["coefficient"] for t in top10]
bars = ax.barh(names[::-1], coefs[::-1], color="steelblue")
ax.set_xlabel("Mean |Coefficient| (across periods)")
ax.set_title("Top 10 Features — Logistic Regression")
ax.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "A4_feature_importance.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved {fig_path}")

# --- Feature importance from Random Forest (independent, nonlinear cross-check) ---
print("\n=== Feature importance (Random Forest on full data) ===")
rf_full = RandomForestClassifier(n_estimators=200, random_state=42)
rf_full.fit(X_scaled, y)

feature_importance_rf = sorted(
    zip(FEATURE_COLS, rf_full.feature_importances_.tolist()),
    key=lambda x: -x[1]
)
top_features_rf = [{"feature": f, "importance": round(c, 4)} for f, c in feature_importance_rf]
print("Top 10 features (Random Forest):")
for item in top_features_rf[:10]:
    print(f"  {item['feature']:25s}  importance={item['importance']:.4f}")

# --- Per-period descriptive statistics (classifier-independent evidence of style change) ---
print("\n=== Feature means by period ===")
period_means_df = df.groupby("period")[FEATURE_COLS].mean().round(4)
period_means_df = period_means_df.reindex(["early", "middle", "late"])
print(period_means_df)
period_means = {
    feature: {period: float(period_means_df.loc[period, feature]) for period in ["early", "middle", "late"]}
    for feature in FEATURE_COLS
}

# --- Automatic periodization: sliding-window cosine similarity ---
print("\n=== Sliding-window periodization ===")

album_vectors = (
    df.sort_values("year")
    .groupby("year")[FEATURE_COLS]
    .mean()
    .reset_index()
)
years = album_vectors["year"].values
X_albums = StandardScaler().fit_transform(album_vectors[FEATURE_COLS].values)

window = 2
similarities = []
for i in range(window, len(years) - window + 1):
    v_before = X_albums[max(0, i - window):i].mean(axis=0).reshape(1, -1)
    v_after  = X_albums[i:min(len(years), i + window)].mean(axis=0).reshape(1, -1)
    sim = float(cosine_similarity(v_before, v_after)[0][0])
    sim = max(0.0, min(1.0, sim))  # treat anticorrelation as zero similarity; also guards float precision
    similarities.append({"year": int(years[i]), "similarity": round(sim, 4)})
    print(f"  {years[i]}: similarity to previous window = {sim:.3f}")

sim_values = [s["similarity"] for s in similarities]
threshold = np.mean(sim_values) - np.std(sim_values)
turning_points = [s for s in similarities if s["similarity"] < threshold]
print(f"\nThreshold: {threshold:.3f}")
print(f"Turning points detected: {[t['year'] for t in turning_points]}")

# Similarity line chart
fig, ax = plt.subplots(figsize=(10, 4))
tp_years = {t["year"] for t in turning_points}
year_vals = [s["year"] for s in similarities]
sim_vals = [s["similarity"] for s in similarities]
ax.plot(year_vals, sim_vals, marker="o", color="steelblue", linewidth=2)
for s in similarities:
    if s["year"] in tp_years:
        ax.axvline(s["year"], color="tomato", linestyle="--", alpha=0.7)
ax.axhline(threshold, color="gray", linestyle=":", label=f"Threshold ({threshold:.3f})")
# Annotate turning points
for s in similarities:
    if s["year"] in tp_years:
        ax.annotate(str(s["year"]), (s["year"], s["similarity"]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    color="tomato", fontsize=9)
ax.set_xlabel("Year")
ax.set_ylabel("Cosine similarity to previous window")
ax.set_title("Sliding-Window Style Similarity — Artzi Corpus")
ax.set_xticks(year_vals)
ax.set_xticklabels(year_vals, rotation=45, ha="right")
ax.legend(fontsize=9)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "A4_period_similarity.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved {fig_path}")

# --- Model comparison chart: accuracy by model x feature subset ---
print("\n=== Model comparison chart ===")
subsets = ["lexical", "syntactic", "all"]
models = ["lr", "svm", "knn"]
model_labels = {"lr": "Logistic Regression", "svm": "SVM-RBF", "knn": "KNN"}
colors = {"lr": "steelblue", "svm": "darkorange", "knn": "seagreen"}

fig, ax = plt.subplots(figsize=(9, 5))
bar_width = 0.25
x = np.arange(len(subsets))
for i, model in enumerate(models):
    means = [cv_scores[f"{model}_{subset}"]["mean"] for subset in subsets]
    stds = [cv_scores[f"{model}_{subset}"]["std"] for subset in subsets]
    ax.bar(x + (i - 1) * bar_width, means, bar_width, yerr=stds, capsize=3,
           label=model_labels[model], color=colors[model])
# Random Forest was only run on the "all" subset - plot it as a single reference bar
rf_x = x[subsets.index("all")] + 2 * bar_width
ax.bar(rf_x, cv_scores["rf_all"]["mean"], bar_width, yerr=cv_scores["rf_all"]["std"],
       capsize=3, label="Random Forest (all)", color="mediumpurple")
ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1, label="Chance baseline (1/3)")
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in subsets])
ax.set_ylabel("5-fold CV accuracy")
ax.set_title("Model Comparison — Accuracy by Feature Subset")
ax.legend(fontsize=8)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "A4_model_comparison.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved {fig_path}")

# --- Save results ---
results = {
    "cv_scores": cv_scores,
    "top_features": top_features,
    "top_features_rf": top_features_rf,
    "period_means": period_means,
    "turning_points": turning_points,
}
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nSaved results to data/classifier_results.json")
