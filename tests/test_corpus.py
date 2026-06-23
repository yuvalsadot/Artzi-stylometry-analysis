import json
import os
import re

CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "corpus.json"
)

def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_corpus_file_exists():
    assert os.path.exists(CORPUS_PATH), "data/corpus.json not found"


def test_corpus_has_correct_count():
    corpus = load_corpus()
    assert len(corpus) == 167, f"Expected 167 songs, got {len(corpus)}"


def test_every_record_has_required_fields():
    corpus = load_corpus()
    required = {"song_id", "title", "year", "album", "text"}
    for record in corpus:
        missing = required - set(record.keys())
        assert not missing, f"Record {record.get('song_id')} missing fields: {missing}"


def test_song_ids_are_unique():
    corpus = load_corpus()
    ids = [r["song_id"] for r in corpus]
    from collections import Counter
    dupes = [id for id, count in Counter(ids).items() if count > 1]
    assert not dupes, f"Duplicate song_ids found: {dupes}"


def test_no_empty_text():
    corpus = load_corpus()
    for record in corpus:
        assert record["text"].strip(), \
            f"Song {record['song_id']} ({record['title']}) has empty text"


def test_text_is_primarily_hebrew():
    corpus = load_corpus()
    hebrew_range = re.compile(r'[א-ת]')
    for record in corpus:
        text = record["text"]
        hebrew_chars = len(hebrew_range.findall(text))
        total_alpha = len(re.findall(r'[a-zA-Zא-ת]', text))
        if total_alpha > 0:
            ratio = hebrew_chars / total_alpha
            assert ratio >= 0.85, \
                f"Song {record['song_id']} ({record['title']}) text is less than 85% Hebrew: {ratio:.2f}"


def test_years_are_valid():
    corpus = load_corpus()
    valid_years = {1970, 1975, 1978, 1979, 1988, 1992, 1996, 2002, 2007, 2012, 2016, 2025}
    for record in corpus:
        assert record["year"] in valid_years, \
            f"Song {record['song_id']} has unexpected year {record['year']}"


def test_all_albums_present():
    corpus = load_corpus()
    expected_albums = {
        "שלמה ארצי", "את ואני", "משחקי 26", "גבר הולך לאיבוד", "דרכים",
        "חום יולי אוגוסט", "ירח", "שניים", "צימאון",
        "שפויים", "אושר אקספרס", "קצפת", "אותיות נחמה"
    }
    found_albums = {r["album"] for r in corpus}
    missing = expected_albums - found_albums
    assert not missing, f"Albums missing from corpus: {missing}"


def test_no_leading_trailing_whitespace_in_text():
    corpus = load_corpus()
    for record in corpus:
        text = record["text"]
        assert text == text.strip(), \
            f"Song {record['song_id']} has leading/trailing whitespace in text"


def test_no_consecutive_blank_lines():
    corpus = load_corpus()
    for record in corpus:
        text = record["text"]
        assert "\n\n\n" not in text, \
            f"Song {record['song_id']} has 3+ consecutive newlines in text"
