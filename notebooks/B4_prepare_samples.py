import pandas as pd
from pathlib import Path

df = pd.read_csv("../data/corpus.csv")

# ===== Define periods =====

early = df[df["year"] <= 1979]
middle = df[(df["year"] >= 1988) & (df["year"] <= 2002)]
late = df[df["year"] >= 2007]

# ===== Sample songs =====

early_sample = early.sample(5, random_state=42)
middle_sample = middle.sample(5, random_state=42)
late_sample = late.sample(5, random_state=42)

Path("../data/B4_samples").mkdir(parents=True, exist_ok=True)


def save_period(sample, filename):

    with open(f"../data/B4_samples/{filename}", "w", encoding="utf-8") as f:

        for _, row in sample.iterrows():

            f.write("=" * 70 + "\n")
            f.write(f"{row['title']} ({row['year']})\n")
            f.write("=" * 70 + "\n\n")

            f.write(row["text_raw"])
            f.write("\n\n\n")


save_period(early_sample, "early.txt")
save_period(middle_sample, "middle.txt")
save_period(late_sample, "late.txt")

print("Saved!")

print("\nEARLY")
print(early_sample["title"].tolist())

print("\nMIDDLE")
print(middle_sample["title"].tolist())

print("\nLATE")
print(late_sample["title"].tolist())