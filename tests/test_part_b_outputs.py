import os
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))


def project_path(*parts):
    return os.path.join(ROOT, *parts)


def test_corpus_exists_and_has_required_columns():
    corpus_path = project_path("data", "corpus.csv")

    assert os.path.exists(corpus_path), "data/corpus.csv is missing"

    df = pd.read_csv(corpus_path)

    assert len(df) > 0, "corpus.csv is empty"

    required_columns = {"song_id", "title", "year", "album"}
    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, f"corpus.csv is missing columns: {missing_columns}"


def test_embedding_clustering_output_exists():
    clusters_path = project_path("data", "clustering", "song_clusters.csv")

    assert os.path.exists(clusters_path), (
        "data/clustering/song_clusters.csv is missing. "
        "Run notebooks/B3_clustering.py first."
    )

    df = pd.read_csv(clusters_path)

    required_columns = {"song_id", "title", "year", "album", "cluster"}
    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, f"song_clusters.csv is missing columns: {missing_columns}"

    assert len(df) > 0, "song_clusters.csv is empty"
    assert df["cluster"].nunique() == 3, "Expected exactly 3 clusters"
    assert df["cluster"].notna().all(), "cluster contains missing values"


def test_embedding_clustering_matches_corpus_size():
    corpus_df = pd.read_csv(project_path("data", "corpus.csv"))
    clusters_df = pd.read_csv(project_path("data", "clustering", "song_clusters.csv"))

    assert len(clusters_df) == len(corpus_df), (
        "song_clusters.csv should contain one row per song in corpus.csv"
    )


def test_embedding_clustering_figures_exist():
    expected_figures = [
        project_path("figures", "B3_pca_clusters.png"),
        project_path("figures", "B3_pca_years.png"),
    ]

    for fig_path in expected_figures:
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
        assert os.path.getsize(fig_path) > 0, f"Figure is empty: {fig_path}"


def test_stylometric_space_output_exists():
    style_path = project_path(
        "data",
        "stylometric_space",
        "song_stylometric_clusters.csv"
    )

    assert os.path.exists(style_path), (
        "data/stylometric_space/song_stylometric_clusters.csv is missing. "
        "Run notebooks/B5_stylometric_space.py first."
    )

    df = pd.read_csv(style_path)

    required_columns = {
        "song_id",
        "title",
        "year",
        "album",
        "period",
        "style_cluster",
        "pca1",
        "pca2",
        "word_count",
        "unique_words",
        "ttr",
        "avg_word_length",
        "line_count",
        "avg_line_length",
        "repetition_ratio",
        "first_person_count",
        "second_person_count",
    }

    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, (
        f"song_stylometric_clusters.csv is missing columns: {missing_columns}"
    )

    assert len(df) > 0, "song_stylometric_clusters.csv is empty"
    assert df["style_cluster"].nunique() == 3, "Expected exactly 3 style clusters"
    assert set(df["period"]).issubset({"Early", "Middle", "Late"}), (
        "Unexpected period values"
    )
    assert df["pca1"].notna().all(), "pca1 contains missing values"
    assert df["pca2"].notna().all(), "pca2 contains missing values"


def test_stylometric_space_matches_corpus_size():
    corpus_df = pd.read_csv(project_path("data", "corpus.csv"))
    style_df = pd.read_csv(
        project_path("data", "stylometric_space", "song_stylometric_clusters.csv")
    )

    assert len(style_df) == len(corpus_df), (
        "song_stylometric_clusters.csv should contain one row per song in corpus.csv"
    )


def test_stylometric_summary_tables_exist():
    expected_files = [
        project_path("data", "stylometric_space", "stylometric_cluster_period_table.csv"),
        project_path("data", "stylometric_space", "stylometric_features_by_period.csv"),
    ]

    for file_path in expected_files:
        assert os.path.exists(file_path), f"Missing file: {file_path}"

        df = pd.read_csv(file_path)
        assert len(df) > 0, f"File is empty: {file_path}"


def test_stylometric_figures_exist():
    expected_figures = [
        project_path("figures", "B5_stylometric_clusters.png"),
        project_path("figures", "B5_stylometric_years.png"),
    ]

    for fig_path in expected_figures:
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"
        assert os.path.getsize(fig_path) > 0, f"Figure is empty: {fig_path}"