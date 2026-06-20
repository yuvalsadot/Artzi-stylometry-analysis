import pandas as pd
from pathlib import Path

metadata = pd.read_excel("../songs/songs_summary.xlsx")
lyrics_dir = Path("../songs/songs_lyrics")

rows = []

for _, row in metadata.iterrows():
    song_id = int(row["song_id"])
    file_path = lyrics_dir / f"{song_id:03d}.txt"

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        continue

    text_raw = file_path.read_text(encoding="utf-8", errors="ignore")

    text_clean = " ".join(text_raw.split())

    rows.append({
        "song_id": song_id,
        "title": row["title"],
        "year": row["year"],
        "album": row["album"],
        "text_raw": text_raw,
        "text_clean": text_clean,
        "word_count": len(text_clean.split())
    })

df = pd.DataFrame(rows)

Path("../data").mkdir(exist_ok=True)
df.to_csv("../data/corpus.csv", index=False, encoding="utf-8-sig")

print(df.head())
print(f"\nSaved {len(df)} songs to ../data/corpus.csv")
print(df[["song_id", "title", "year", "album", "word_count"]].head())