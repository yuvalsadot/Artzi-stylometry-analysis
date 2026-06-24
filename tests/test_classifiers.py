import json
import os
import pytest

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "classifier_results.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_results_file_exists():
    assert os.path.exists(RESULTS_PATH), "data/classifier_results.json not found"


def test_results_has_required_keys():
    r = load_results()
    required = {"cv_scores", "top_features", "turning_points"}
    missing = required - set(r.keys())
    assert not missing, f"Missing keys: {missing}"


def test_cv_scores_structure():
    r = load_results()
    cv = r["cv_scores"]
    required_models = {"lr_all", "lr_lexical", "lr_syntactic", "svm_all"}
    missing = required_models - set(cv.keys())
    assert not missing, f"Missing model keys in cv_scores: {missing}"
    for model_key, scores in cv.items():
        assert "mean" in scores, f"{model_key} missing 'mean'"
        assert "std" in scores, f"{model_key} missing 'std'"
        assert 0.0 <= scores["mean"] <= 1.0, f"{model_key} mean accuracy out of [0,1]: {scores['mean']}"
        assert 0.0 <= scores["std"] <= 1.0, f"{model_key} std out of [0,1]: {scores['std']}"


def test_top_features_structure():
    r = load_results()
    top = r["top_features"]
    assert isinstance(top, list), "top_features must be a list"
    assert len(top) >= 5, f"Expected at least 5 top features, got {len(top)}"
    for item in top:
        assert "feature" in item, f"top_features item missing 'feature': {item}"
        assert "coefficient" in item, f"top_features item missing 'coefficient': {item}"


def test_turning_points_structure():
    r = load_results()
    tp = r["turning_points"]
    assert isinstance(tp, list), "turning_points must be a list"
    for item in tp:
        assert "year" in item, f"turning_points item missing 'year': {item}"
        assert "similarity" in item, f"turning_points item missing 'similarity': {item}"
        assert 1970 <= item["year"] <= 2025, f"turning point year out of range: {item['year']}"
        assert 0.0 <= item["similarity"] <= 1.0, f"similarity out of [0,1]: {item['similarity']}"


def test_feature_importance_figure_exists():
    path = os.path.join(FIGURES_DIR, "A4_feature_importance.png")
    assert os.path.exists(path), "figures/A4_feature_importance.png not found"


def test_period_similarity_figure_exists():
    path = os.path.join(FIGURES_DIR, "A4_period_similarity.png")
    assert os.path.exists(path), "figures/A4_period_similarity.png not found"
