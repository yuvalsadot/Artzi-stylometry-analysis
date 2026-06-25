import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_PATH = "../data/corpus.csv"
OUT_DATA = "../data/stylometric_space"
OUT_FIGURES = "../figures"

os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)


HEBREW_WORD_RE = re.compile(r"[א-ת]+")

FIRST_PERSON = {
    "אני", "אנחנו", "שלי", "שלנו", "אותי", "אותנו", "לי", "לנו", "בי", "בנו"
}

SECOND_PERSON = {
    "את", "אתה", "אתם", "אתן", "שלך", "שלכם", "שלכן",
    "אותך", "אתכם", "אתכן", "לך", "לכם", "לכן", "בך", "בכם", "בכן"
}


def get_period(year):
    if year <= 1984:
        return "Early"
    elif year <= 2002:
        return "Middle"
    else:
        return "Late"


def tokenize(text):
    return HEBREW_WORD_RE.findall(str(text))


def extract_features(row):
    text = str(row["text_clean"])
    words = tokenize(text)

    word_count = len(words)
    unique_words = len(set(words))
    ttr = unique_words / word_count if word_count > 0 else 0

    avg_word_length = (
        np.mean([len(w) for w in words]) if word_count > 0 else 0
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    line_count = len(lines)
    line_lengths = [len(tokenize(line)) for line in lines]
    avg_line_length = (
        np.mean(line_lengths) if line_lengths else 0
    )

    repeated_words = word_count - unique_words
    repetition_ratio = repeated_words / word_count if word_count > 0 else 0

    first_person_count = sum(1 for w in words if w in FIRST_PERSON)
    second_person_count = sum(1 for w in words if w in SECOND_PERSON)

    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "ttr": ttr,
        "avg_word_length": avg_word_length,
        "line_count": line_count,
        "avg_line_length": avg_line_length,
        "repetition_ratio": repetition_ratio,
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
        "comma_count": text.count(","),
        "quote_count": text.count('"') + text.count("״") + text.count("'"),
        "first_person_count": first_person_count,
        "second_person_count": second_person_count,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    feature_rows = df.apply(extract_features, axis=1)
    features_df = pd.DataFrame(list(feature_rows))

    meta_cols = ["song_id", "title", "year", "album"]
    result_df = pd.concat([df[meta_cols].reset_index(drop=True), features_df], axis=1)
    result_df["period"] = result_df["year"].apply(get_period)

    feature_cols = list(features_df.columns)

    X = result_df[feature_cols].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, clusters)

    result_df["style_cluster"] = clusters
    result_df["pca1"] = X_pca[:, 0]
    result_df["pca2"] = X_pca[:, 1]

    out_csv = os.path.join(OUT_DATA, "song_stylometric_clusters.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    period_table = pd.crosstab(
        result_df["style_cluster"],
        result_df["period"]
    )

    period_table_path = os.path.join(
        OUT_DATA,
        "stylometric_cluster_period_table.csv"
    )
    period_table.to_csv(period_table_path, encoding="utf-8-sig")

    means_by_period = result_df.groupby("period")[feature_cols].mean()
    means_path = os.path.join(
        OUT_DATA,
        "stylometric_features_by_period.csv"
    )
    means_by_period.to_csv(means_path, encoding="utf-8-sig")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        result_df["pca1"],
        result_df["pca2"],
        c=result_df["style_cluster"]
    )
    plt.title("Stylometric Space - KMeans Clusters")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGURES, "B5_stylometric_clusters.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        result_df["pca1"],
        result_df["pca2"],
        c=result_df["year"]
    )
    plt.title("Stylometric Space - Colored by Year")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Year")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGURES, "B5_stylometric_years.png"), dpi=300)
    plt.close()

    print("Silhouette score:", sil)
    print("Saved:")
    print(out_csv)
    print(period_table_path)
    print(means_path)
    print("../figures/B5_stylometric_clusters.png")
    print("../figures/B5_stylometric_years.png")
    print()
    print("Cluster counts:")
    print(result_df["style_cluster"].value_counts().sort_index())
    print()
    print("Period table:")
    print(period_table)
    print()
    print("Average features by period:")
    print(means_by_period)


if __name__ == "__main__":
    main()