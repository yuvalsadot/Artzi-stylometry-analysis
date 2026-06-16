# A1 — Corpus Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single clean, unified corpus file (`data/corpus.json`) from the 167 raw song lyrics files and the Excel metadata, with all text normalized and all known data issues documented and handled.

**Architecture:** A single Jupyter notebook (`notebooks/A1_corpus_cleaning.ipynb`) loads raw lyrics + Excel metadata, normalizes text, flags and handles known issues (9 files with non-Hebrew characters, 1 empty file), and writes `data/corpus.json`. A companion test script (`tests/test_corpus.py`) validates the output file's integrity.

**Tech Stack:** Python 3, pandas, openpyxl, re (stdlib), json (stdlib), pytest, jupyter

---

## File Structure

| File | Purpose |
|------|---------|
| `notebooks/A1_corpus_cleaning.ipynb` | Main cleaning notebook — loads raw data, normalizes, outputs corpus |
| `data/corpus.json` | Output: clean unified corpus (one record per song) |
| `data/issues_report.json` | Output: log of every data issue found and how it was handled |
| `tests/test_corpus.py` | Validates corpus.json structure, completeness, and text quality |

---

## Task 1: Install dependencies and scaffold directories

**Files:**
- Create: `notebooks/A1_corpus_cleaning.ipynb`
- Create: `tests/test_corpus.py`

- [ ] **Step 1: Install required packages**

Run in terminal:
```bash
pip install pandas openpyxl pytest jupyter
```
Expected output: packages install without errors.

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p /Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI/data
mkdir -p /Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI/notebooks
mkdir -p /Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI/tests
```

- [ ] **Step 3: Create an empty notebook**

Create `notebooks/A1_corpus_cleaning.ipynb` with a single markdown cell:
```
# A1 — Corpus Cleaning
```

- [ ] **Step 4: Create empty test file**

Create `tests/test_corpus.py` with just an import:
```python
import json, os
```

---

## Task 2: Write the failing tests first

**Files:**
- Modify: `tests/test_corpus.py`

Before writing any cleaning code, write tests that define exactly what the clean corpus must look like. Run them — they should all fail because `data/corpus.json` doesn't exist yet.

- [ ] **Step 1: Write all corpus validation tests**

Replace the contents of `tests/test_corpus.py` with:

```python
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
    # 167 songs total; song 120 is empty and excluded → 166 valid entries
    assert len(corpus) == 166, f"Expected 166 songs, got {len(corpus)}"


def test_every_record_has_required_fields():
    corpus = load_corpus()
    required = {"song_id", "title", "year", "album", "text_clean"}
    for record in corpus:
        missing = required - set(record.keys())
        assert not missing, f"Record {record.get('song_id')} missing fields: {missing}"


def test_song_ids_are_unique():
    corpus = load_corpus()
    ids = [r["song_id"] for r in corpus]
    assert len(ids) == len(set(ids)), "Duplicate song_ids found"


def test_no_empty_text():
    corpus = load_corpus()
    for record in corpus:
        assert record["text_clean"].strip(), \
            f"Song {record['song_id']} ({record['title']}) has empty text_clean"


def test_text_is_primarily_hebrew():
    corpus = load_corpus()
    hebrew_range = re.compile(r'[א-ת]')
    for record in corpus:
        text = record["text_clean"]
        hebrew_chars = len(hebrew_range.findall(text))
        total_alpha = len(re.findall(r'[a-zA-Zא-ת]', text))
        if total_alpha > 0:
            ratio = hebrew_chars / total_alpha
            assert ratio >= 0.85, \
                f"Song {record['song_id']} ({record['title']}) is less than 85% Hebrew: {ratio:.2f}"


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
        text = record["text_clean"]
        assert text == text.strip(), \
            f"Song {record['song_id']} has leading/trailing whitespace"


def test_no_consecutive_blank_lines():
    corpus = load_corpus()
    for record in corpus:
        text = record["text_clean"]
        assert "\n\n\n" not in text, \
            f"Song {record['song_id']} has 3+ consecutive newlines"
```

- [ ] **Step 2: Run tests — confirm they all fail**

```bash
cd /Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI
python -m pytest tests/test_corpus.py -v
```

Expected: all tests FAIL with `FileNotFoundError` or `AssertionError`. This is correct — we haven't built the corpus yet.

---

## Task 3: Load metadata from Excel

**Files:**
- Modify: `notebooks/A1_corpus_cleaning.ipynb`

Add a new code cell to the notebook for each step below.

- [ ] **Step 1: Add cell — imports**

```python
import os
import re
import json
import pandas as pd

BASE_DIR = "/Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI"
LYRICS_DIR = os.path.join(BASE_DIR, "songs", "songs_lyrics")
EXCEL_PATH = os.path.join(BASE_DIR, "songs", "songs_summary.xlsx")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
```

- [ ] **Step 2: Add cell — load Excel metadata**

```python
df_meta = pd.read_excel(EXCEL_PATH, sheet_name="גיליון1")
df_meta.columns = ["song_id", "title", "year", "album"]
print(f"Loaded {len(df_meta)} songs from Excel")
print(df_meta.head())
print(df_meta["year"].value_counts().sort_index())
```

Expected output: 167 rows, columns song_id/title/year/album, year counts matching known distribution.

- [ ] **Step 3: Run the cell and verify output looks correct**

Should print 167 rows and year distribution: 1970:12, 1975:22, 1978:8, 1979:8, 1988:15, 1992:12, 1996:19, 2002:15, 2007:15, 2012:12, 2016:14, 2025:15.

---

## Task 4: Load raw lyrics and identify all issues

**Files:**
- Modify: `notebooks/A1_corpus_cleaning.ipynb`

- [ ] **Step 1: Add cell — load all raw lyrics**

```python
def load_raw_lyrics(song_id: int) -> str:
    filename = f"{song_id:03d}.txt"
    path = os.path.join(LYRICS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()

# Load all lyrics into the dataframe
df_meta["text_raw"] = df_meta["song_id"].apply(load_raw_lyrics)
print(f"Loaded lyrics for {len(df_meta)} songs")
```

- [ ] **Step 2: Add cell — detect and report all data issues**

```python
HEBREW_RE = re.compile(r'[א-ת]')
NON_HEBREW_ALPHA_RE = re.compile(r'[a-zA-Z]')

issues = []

for _, row in df_meta.iterrows():
    song_id = row["song_id"]
    title = row["title"]
    text = row["text_raw"]

    # Issue: empty file
    if not text.strip():
        issues.append({
            "song_id": song_id,
            "title": title,
            "issue": "empty_file",
            "detail": "File contains no text",
            "action": "exclude_from_corpus"
        })
        continue

    # Issue: non-Hebrew alphabetic characters
    non_heb = NON_HEBREW_ALPHA_RE.findall(text)
    if non_heb:
        sample = "".join(sorted(set(non_heb)))
        issues.append({
            "song_id": song_id,
            "title": title,
            "issue": "non_hebrew_chars",
            "detail": f"Contains Latin characters: {sample}",
            "action": "keep_strip_latin"
        })

    # Issue: special punctuation (em-dash, backtick-like chars)
    special = re.findall(r'[–´~]', text)
    if special:
        issues.append({
            "song_id": song_id,
            "title": title,
            "issue": "special_punctuation",
            "detail": f"Contains: {''.join(set(special))}",
            "action": "normalize_punctuation"
        })

print(f"Found {len(issues)} issues across {len(set(i['song_id'] for i in issues))} songs:")
for issue in issues:
    print(f"  Song {issue['song_id']:03d} ({issue['title']}): {issue['issue']} — {issue['action']}")
```

Expected output: issues for songs 23, 27, 34, 61, 69, 71, 75, 91, 120, 138 as identified during planning.

---

## Task 5: Write the text normalization function

**Files:**
- Modify: `notebooks/A1_corpus_cleaning.ipynb`

- [ ] **Step 1: Add cell — normalization function**

```python
def normalize_text(text: str) -> str:
    """
    Clean a raw Hebrew lyrics string.
    - Strip Latin characters (foreign words/phrases that appear in a few songs)
    - Normalize special punctuation to standard equivalents
    - Collapse excess whitespace and blank lines
    - Strip leading/trailing whitespace
    """
    # Remove Latin alphabetic characters (a-z, A-Z)
    # These appear in ~9 songs as isolated foreign phrases (e.g. "je t'aime", "Love")
    # We remove them rather than transliterate since Hebrew NLP tools ignore them anyway
    text = re.sub(r'[a-zA-Z]+', '', text)

    # Normalize special punctuation
    text = text.replace('–', '-')   # em-dash → hyphen
    text = text.replace('´', "'")   # backtick-like → apostrophe
    text = text.replace('~', '')    # tilde → remove

    # Remove Unicode directional marks (U+200E, U+200F) sometimes embedded in Hebrew text
    text = re.sub(r'[‎‏]', '', text)

    # Collapse multiple spaces on a single line to one space
    text = re.sub(r'[ \t]+', ' ', text)

    # Strip trailing space from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # Collapse 3+ consecutive newlines to exactly 2 (stanza separator)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that became empty after stripping Latin chars
    # (e.g. a line that was just "Love" becomes "")
    lines = text.split('\n')
    lines = [l for l in lines if l.strip()]
    text = '\n'.join(lines)

    return text.strip()
```

- [ ] **Step 2: Add cell — quick manual spot-check**

```python
# Spot-check song 91 (Love לאהבה) — had "Love" in it
raw_91 = df_meta.loc[df_meta["song_id"] == 91, "text_raw"].values[0]
clean_91 = normalize_text(raw_91)
print("RAW:")
print(raw_91[:300])
print("\nCLEAN:")
print(clean_91[:300])
```

Verify visually: the word "Love" should be gone, Hebrew text intact, no extra blank lines.

- [ ] **Step 3: Add cell — spot-check song 23 (je t'aime)**

```python
raw_23 = df_meta.loc[df_meta["song_id"] == 23, "text_raw"].values[0]
clean_23 = normalize_text(raw_23)
print("CLEAN song 23:")
print(clean_23[:400])
```

Verify: "je t'aime" removed, rest of song intact.

---

## Task 6: Build and save the final corpus

**Files:**
- Modify: `notebooks/A1_corpus_cleaning.ipynb`
- Create: `data/corpus.json`
- Create: `data/issues_report.json`

- [ ] **Step 1: Add cell — apply normalization and build corpus records**

```python
corpus = []

for _, row in df_meta.iterrows():
    song_id = int(row["song_id"])
    text_raw = row["text_raw"]

    # Exclude the empty file (song 120)
    if not text_raw.strip():
        print(f"Excluding song {song_id:03d} ({row['title']}) — empty file")
        continue

    text_clean = normalize_text(text_raw)

    corpus.append({
        "song_id": song_id,
        "title": row["title"],
        "year": int(row["year"]),
        "album": row["album"],
        "text_raw": text_raw,
        "text_clean": text_clean,
        "word_count": len(text_clean.split()),
        "line_count": len([l for l in text_clean.split('\n') if l.strip()]),
    })

print(f"Corpus built: {len(corpus)} songs")
```

Expected: "Corpus built: 166 songs" (167 minus the 1 empty file).

- [ ] **Step 2: Add cell — save corpus.json**

```python
corpus_path = os.path.join(DATA_DIR, "corpus.json")
with open(corpus_path, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"Saved corpus to {corpus_path}")
print(f"File size: {os.path.getsize(corpus_path) / 1024:.1f} KB")
```

- [ ] **Step 3: Add cell — save issues_report.json**

```python
issues_path = os.path.join(DATA_DIR, "issues_report.json")
with open(issues_path, "w", encoding="utf-8") as f:
    json.dump(issues, f, ensure_ascii=False, indent=2)

print(f"Saved issues report to {issues_path}")
```

- [ ] **Step 4: Add cell — print summary statistics**

```python
import collections

df_corpus = pd.DataFrame(corpus)
print("=== Corpus Summary ===")
print(f"Total songs:        {len(df_corpus)}")
print(f"Albums:             {df_corpus['album'].nunique()}")
print(f"Year range:         {df_corpus['year'].min()} – {df_corpus['year'].max()}")
print(f"Avg words/song:     {df_corpus['word_count'].mean():.0f}")
print(f"Min words/song:     {df_corpus['word_count'].min()} (song {df_corpus.loc[df_corpus['word_count'].idxmin(), 'song_id']})")
print(f"Max words/song:     {df_corpus['word_count'].max()} (song {df_corpus.loc[df_corpus['word_count'].idxmax(), 'song_id']})")
print()
print("Songs per album:")
print(df_corpus.groupby(["year","album"])["song_id"].count().to_string())
```

---

## Task 7: Run all tests — make them pass

**Files:**
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI
python -m pytest tests/test_corpus.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 2: If any test fails, diagnose and fix**

Common failure scenarios:
- `test_corpus_has_correct_count` fails with 167 instead of 166 → song 120 wasn't excluded. Check that `if not text_raw.strip(): continue` runs before appending.
- `test_text_is_primarily_hebrew` fails → normalization missed some Latin chars. Check the `re.sub(r'[a-zA-Z]+', '', text)` line in `normalize_text`.
- `test_no_consecutive_blank_lines` fails → the `re.sub(r'\n{3,}', '\n\n', text)` didn't run. Verify it's in `normalize_text`.

Re-run the full notebook top-to-bottom after any fix, then re-run tests.

- [ ] **Step 3: Confirm all 10 tests pass before continuing**

```
tests/test_corpus.py::test_corpus_file_exists PASSED
tests/test_corpus.py::test_corpus_has_correct_count PASSED
tests/test_corpus.py::test_every_record_has_required_fields PASSED
tests/test_corpus.py::test_song_ids_are_unique PASSED
tests/test_corpus.py::test_no_empty_text PASSED
tests/test_corpus.py::test_text_is_primarily_hebrew PASSED
tests/test_corpus.py::test_years_are_valid PASSED
tests/test_corpus.py::test_all_albums_present PASSED
tests/test_corpus.py::test_no_leading_trailing_whitespace_in_text PASSED
tests/test_corpus.py::test_no_consecutive_blank_lines PASSED
10 passed
```

---

## Self-Review

**Spec coverage check:**
- ✅ Load Excel metadata → Task 3
- ✅ Load all 167 raw lyrics → Task 4
- ✅ Flag/handle 9 files with non-Hebrew chars → Task 4 (detection) + Task 5 (normalization)
- ✅ Handle 1 empty file (120.txt) → excluded in Task 6, documented in issues_report.json
- ✅ Normalize text → Task 5
- ✅ Output `data/corpus.json` with required fields → Task 6
- ✅ Document issues → `data/issues_report.json` in Task 6
- ✅ Tests validate the output → Task 2 (written first, TDD) + Task 7 (make pass)

**Placeholder scan:** None found.

**Type consistency:** `song_id` is consistently `int` throughout (cast explicitly in Task 6 Step 1). `year` is `int`. `text_clean` is `str`. All test assertions reference these same field names.
