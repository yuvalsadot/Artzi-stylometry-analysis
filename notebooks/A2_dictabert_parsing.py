import os
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE_DIR = "/Users/I771657/Library/CloudStorage/OneDrive-SAPSE/Documents/School/MINI"
CORPUS_PATH = os.path.join(BASE_DIR, "data", "corpus.json")
PARSED_DIR = os.path.join(BASE_DIR, "data", "parsed")
Path(PARSED_DIR).mkdir(parents=True, exist_ok=True)

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)
print(f"Loaded {len(corpus)} songs")

# Load models
print("Loading morphology model (dicta-il/dictabert-tiny-joint)...")
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)
morph_model = AutoModel.from_pretrained('dicta-il/dictabert-tiny-joint', trust_remote_code=True)
morph_model.eval()
print("Morphology model loaded.")


def extract_morphology(result_dict: dict) -> list:
    """Extract token-level morphological data from dictabert-tiny-joint output.
    result_dict is result[0] — a dict with keys: text, tokens, root_idx, ner_entities
    """
    tokens = []
    for token_data in result_dict.get("tokens", []):
        morph = token_data.get("morph", {})
        syntax = token_data.get("syntax", {})
        tokens.append({
            "word": token_data.get("token", ""),
            "lemma": token_data.get("lex", token_data.get("token", "")),
            "pos": morph.get("pos", ""),
            "dep": syntax.get("dep_func", ""),
            "dep_head": syntax.get("dep_head", ""),
            "feats": morph.get("feats", {}),
        })
    return tokens


def extract_ner(result_dict: dict) -> list:
    """Extract NER entities from dictabert-tiny-joint output (ner_entities field)."""
    entities = []
    for entity in result_dict.get("ner_entities", []):
        entities.append({
            "word": entity.get("phrase", ""),
            "label": entity.get("label", ""),
            "start": entity.get("start", 0),
            "end": entity.get("end", 0),
        })
    return entities


# Spot-check on first song before running all
sample = corpus[0]
print(f"\nSpot-check: song {sample['song_id']} — {sample['title']}")
raw = morph_model.predict([sample["text"]], tokenizer, output_style='json')
morph = extract_morphology(raw[0])
ner = extract_ner(raw[0])
print(f"  Tokens: {len(morph)}")
print(f"  First 5 tokens: {morph[:5]}")
print(f"  NER entities: {len(ner)} — {ner[:3]}")

# Run on all songs
print("\nParsing all songs...")
start = time.time()

for i, record in enumerate(corpus):
    song_id = record["song_id"]
    out_path = os.path.join(PARSED_DIR, f"{song_id:03d}.json")

    if os.path.exists(out_path):
        continue  # resume if interrupted

    text = record["text"]
    raw = morph_model.predict([text], tokenizer, output_style='json')

    parsed = {
        "song_id": song_id,
        "title": record["title"],
        "year": record["year"],
        "album": record["album"],
        "morphology": extract_morphology(raw[0]),
        "ner": extract_ner(raw[0]),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        print(f"  {i + 1}/167 songs ({elapsed:.0f}s elapsed)")

elapsed = time.time() - start
n_files = len(list(Path(PARSED_DIR).glob("*.json")))
print(f"\nDone. {n_files} files in data/parsed/ ({elapsed:.0f}s total)")
