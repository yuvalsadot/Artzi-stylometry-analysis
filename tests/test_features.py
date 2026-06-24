import os
import pandas as pd
import pytest

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")

EXPECTED_COLUMNS = [
    "song_id", "title", "year", "album",
    # lexical
    "ttr", "mtld", "avg_word_length", "avg_line_length", "vocab_size",
    # morphosyntactic
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
    "pos_ADP", "pos_CCONJ", "pos_SCONJ",
    "verb_noun_ratio", "function_word_ratio", "first_person_ratio",
    # structural
    "num_lines", "num_stanzas", "repetition_rate",
]


def load_features():
    return pd.read_csv(FEATURES_PATH)


def test_features_file_exists():
    assert os.path.exists(FEATURES_PATH), "data/features.csv not found"


def test_features_has_correct_row_count():
    df = load_features()
    assert len(df) == 167, f"Expected 167 rows, got {len(df)}"


def test_features_has_all_columns():
    df = load_features()
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_song_ids_are_unique():
    df = load_features()
    dupes = df[df.duplicated(subset="song_id")]["song_id"].tolist()
    assert not dupes, f"Duplicate song_ids: {dupes}"


def test_no_null_values():
    df = load_features()
    nulls = df[EXPECTED_COLUMNS].isnull().sum()
    cols_with_nulls = nulls[nulls > 0].index.tolist()
    assert not cols_with_nulls, f"Null values in columns: {cols_with_nulls}"


def test_ttr_in_valid_range():
    df = load_features()
    invalid = df[(df["ttr"] < 0) | (df["ttr"] > 1)]
    assert invalid.empty, f"TTR out of range for songs: {invalid['song_id'].tolist()}"


def test_mtld_is_positive():
    df = load_features()
    invalid = df[df["mtld"] <= 0]
    assert invalid.empty, f"MTLD not positive for songs: {invalid['song_id'].tolist()}"


def test_pos_ratios_sum_plausibly():
    df = load_features()
    pos_cols = ["pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV",
                "pos_PRON", "pos_ADP", "pos_CCONJ", "pos_SCONJ"]
    for col in pos_cols:
        invalid = df[(df[col] < 0) | (df[col] > 1)]
        assert invalid.empty, f"{col} out of [0, 1] for songs: {invalid['song_id'].tolist()}"


def test_repetition_rate_in_valid_range():
    df = load_features()
    invalid = df[(df["repetition_rate"] < 0) | (df["repetition_rate"] > 1)]
    assert invalid.empty, f"repetition_rate out of range for songs: {invalid['song_id'].tolist()}"


def test_num_lines_is_positive():
    df = load_features()
    invalid = df[df["num_lines"] <= 0]
    assert invalid.empty, f"num_lines not positive for songs: {invalid['song_id'].tolist()}"


def test_num_stanzas_is_positive():
    df = load_features()
    invalid = df[df["num_stanzas"] <= 0]
    assert invalid.empty, f"num_stanzas not positive for songs: {invalid['song_id'].tolist()}"


def test_function_word_ratio_in_valid_range():
    df = load_features()
    invalid = df[(df["function_word_ratio"] < 0) | (df["function_word_ratio"] > 1)]
    assert invalid.empty, f"function_word_ratio out of range for songs: {invalid['song_id'].tolist()}"


def test_first_person_ratio_in_valid_range():
    df = load_features()
    invalid = df[(df["first_person_ratio"] < 0) | (df["first_person_ratio"] > 1)]
    assert invalid.empty, f"first_person_ratio out of range for songs: {invalid['song_id'].tolist()}"
