import json
import os
import pytest

PARSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "parsed")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.json")


def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_parsed(song_id: int):
    path = os.path.join(PARSED_DIR, f"{song_id:03d}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_parsed_dir_exists():
    assert os.path.isdir(PARSED_DIR), "data/parsed/ directory not found"


def test_all_songs_have_parsed_file():
    corpus = load_corpus()
    missing = []
    for record in corpus:
        path = os.path.join(PARSED_DIR, f"{record['song_id']:03d}.json")
        if not os.path.exists(path):
            missing.append(record["song_id"])
    assert not missing, f"Missing parsed files for songs: {missing}"


def test_parsed_file_has_required_fields():
    corpus = load_corpus()
    for record in corpus:
        parsed = load_parsed(record["song_id"])
        assert "song_id" in parsed, f"Song {record['song_id']} missing song_id"
        assert "morphology" in parsed, f"Song {record['song_id']} missing morphology"
        assert "ner" in parsed, f"Song {record['song_id']} missing ner"


def test_morphology_tokens_have_required_fields():
    corpus = load_corpus()
    required = {"word", "lemma", "pos", "dep"}
    for record in corpus:
        parsed = load_parsed(record["song_id"])
        for i, token in enumerate(parsed["morphology"]):
            missing = required - set(token.keys())
            assert not missing, \
                f"Song {record['song_id']} token {i} missing fields: {missing}"


def test_morphology_is_nonempty():
    corpus = load_corpus()
    for record in corpus:
        parsed = load_parsed(record["song_id"])
        assert len(parsed["morphology"]) > 0, \
            f"Song {record['song_id']} has empty morphology list"


def test_ner_entities_have_required_fields():
    corpus = load_corpus()
    for record in corpus:
        parsed = load_parsed(record["song_id"])
        for entity in parsed["ner"]:
            assert "word" in entity, f"Song {record['song_id']} NER entity missing 'word'"
            assert "label" in entity, f"Song {record['song_id']} NER entity missing 'label'"
