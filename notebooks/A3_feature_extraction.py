import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from lexicalrichness import LexicalRichness

BASE_DIR = "/Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI"
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.json")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Loaded {len(corpus)} songs")

TRACKED_POS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "CCONJ", "SCONJ"]
FUNCTION_POS = {"ADP", "CCONJ", "SCONJ", "PRON"}
FIRST_PERSON_SURFACE = {"לי", "אותי"}
EXCLUDE_POS = {"PUNCT", "SYM"}


def compute_ttr(words):
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_mtld(words):
    if len(words) < 10:
        return 0.0
    try:
        lex = LexicalRichness(" ".join(words))
        return lex.mtld(threshold=0.72)
    except Exception:
        return 0.0


def compute_avg_word_length(words):
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_avg_line_length(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(len(l.split()) for l in lines) / len(lines)


def compute_vocab_size(lemmas):
    return len(set(lemmas))


def compute_pos_features(morphology):
    content = [t for t in morphology if t["pos"] not in EXCLUDE_POS and t["pos"] != ""]
    n = len(content)
    all_pos = [t["pos"] for t in content]
    if n == 0:
        result = {f"pos_{p}": 0.0 for p in TRACKED_POS}
        result.update({"verb_noun_ratio": 0.0, "function_word_ratio": 0.0, "first_person_ratio": 0.0})
        return result

    counts = Counter(all_pos)
    result = {f"pos_{p}": counts.get(p, 0) / n for p in TRACKED_POS}

    noun_count = counts.get("NOUN", 0) + counts.get("PROPN", 0)
    verb_count = counts.get("VERB", 0) + counts.get("AUX", 0)
    result["verb_noun_ratio"] = verb_count / noun_count if noun_count > 0 else 0.0

    function_count = sum(1 for t in content if t["pos"] in FUNCTION_POS)
    result["function_word_ratio"] = function_count / n

    fp_count = sum(
        1 for t in content
        if (t.get("lemma") == "אני" and t.get("pos") == "PRON")
        or t.get("word") in FIRST_PERSON_SURFACE
    )
    result["first_person_ratio"] = fp_count / n

    return result


def compute_structural_features(text):
    lines = text.splitlines()
    non_empty = [l.strip() for l in lines if l.strip()]
    num_lines = len(non_empty)

    num_stanzas = 0
    in_stanza = False
    for line in lines:
        if line.strip():
            if not in_stanza:
                num_stanzas += 1
                in_stanza = True
        else:
            in_stanza = False
    num_stanzas = max(num_stanzas, 1)

    line_counts = Counter(non_empty)
    repeated = sum(1 for l in non_empty if line_counts[l] > 1)
    repetition_rate = repeated / num_lines if num_lines > 0 else 0.0

    return {"num_lines": num_lines, "num_stanzas": num_stanzas, "repetition_rate": repetition_rate}


# Spot-check on first song
sample = corpus[0]
parsed_path = os.path.join(PARSED_DIR, f"{sample['song_id']:03d}.json")
with open(parsed_path, encoding="utf-8") as f:
    parsed = json.load(f)

content_tokens = [t for t in parsed["morphology"] if t["pos"] not in EXCLUDE_POS and t["pos"] != ""]
words = [t["word"] for t in content_tokens]
lemmas = [t["lemma"] for t in content_tokens]
print(f"\nSpot-check: {sample['title']} ({sample['year']})")
print(f"  tokens={len(words)}, unique lemmas={len(set(lemmas))}")
print(f"  TTR={compute_ttr(words):.3f}, MTLD={compute_mtld(words):.1f}")
print(f"  avg_word_len={compute_avg_word_length(words):.2f}, avg_line_len={compute_avg_line_length(sample['text']):.2f}")
pos_f = compute_pos_features(parsed["morphology"])
print(f"  pos_NOUN={pos_f['pos_NOUN']:.3f}, pos_VERB={pos_f['pos_VERB']:.3f}")
print(f"  function_word_ratio={pos_f['function_word_ratio']:.3f}, first_person_ratio={pos_f['first_person_ratio']:.3f}")
struct = compute_structural_features(sample["text"])
print(f"  lines={struct['num_lines']}, stanzas={struct['num_stanzas']}, repetition={struct['repetition_rate']:.3f}")

# Process all songs
print("\nProcessing all 167 songs...")
rows = []
for record in corpus:
    song_id = record["song_id"]
    parsed_path = os.path.join(PARSED_DIR, f"{song_id:03d}.json")
    with open(parsed_path, encoding="utf-8") as f:
        parsed = json.load(f)

    morphology = parsed["morphology"]
    text = record["text"]

    content_tokens = [t for t in morphology if t["pos"] not in EXCLUDE_POS and t["pos"] != ""]
    words = [t["word"] for t in content_tokens]
    lemmas = [t["lemma"] for t in content_tokens]

    lexical = {
        "ttr": compute_ttr(words),
        "mtld": compute_mtld(words),
        "avg_word_length": compute_avg_word_length(words),
        "avg_line_length": compute_avg_line_length(text),
        "vocab_size": compute_vocab_size(lemmas),
    }
    pos_feats = compute_pos_features(morphology)
    struct = compute_structural_features(text)

    rows.append({
        "song_id": song_id,
        "title": record["title"],
        "year": record["year"],
        "album": record["album"],
        **lexical,
        **pos_feats,
        **struct,
    })

col_order = [
    "song_id", "title", "year", "album",
    "ttr", "mtld", "avg_word_length", "avg_line_length", "vocab_size",
    "pos_NOUN", "pos_VERB", "pos_ADJ", "pos_ADV", "pos_PRON",
    "pos_ADP", "pos_CCONJ", "pos_SCONJ",
    "verb_noun_ratio", "function_word_ratio", "first_person_ratio",
    "num_lines", "num_stanzas", "repetition_rate",
]
df = pd.DataFrame(rows)[col_order]
df.to_csv(FEATURES_PATH, index=False, encoding="utf-8")
print(f"\nSaved {len(df)} rows to data/features.csv")
print(df[["ttr", "mtld", "pos_NOUN", "pos_VERB", "first_person_ratio", "repetition_rate"]].describe().round(3))
