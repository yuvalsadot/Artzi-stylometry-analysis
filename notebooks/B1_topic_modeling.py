import re
import pandas as pd
from pathlib import Path
from gensim.corpora import Dictionary
from gensim.models import LdaModel

df = pd.read_csv("../data/corpus.csv")

hebrew_stopwords = {
    "של", "את", "על", "עם", "זה", "זו", "הוא", "היא", "הם", "הן",
    "אני", "אתה", "אתם", "אתן", "אנחנו", "לי", "לך", "לו", "לה",
    "מה", "מי", "כי", "אם", "לא", "כן", "גם", "כל", "או", "אז",
    "אבל", "רק", "עוד", "כמו", "יש", "אין", "היה", "היו", "להיות",
    "בין", "אל", "מן", "כבר", "אחד", "אחת","שוב", "עכשיו", "אותי", "אותך", "שלי", "איך", "שם", "פה", "כאן",
    "ולא", "שלא", "שאני", "ואני", "איתך", "איתי", "לנו", "לכם",
    "האם", "הנה", "כך", "עד", "בתוך", "ליד", "לפעמים", "כמעט",
    "פעם", "הכל", "הייתי", "יום", "בא", "באה"
}

def tokenize_hebrew(text):
    text = str(text)
    words = re.findall(r"[א-ת]+", text)
    words = [w for w in words if len(w) > 1]
    words = [w for w in words if w not in hebrew_stopwords]
    return words

texts = df["text_clean"].apply(tokenize_hebrew).tolist()

dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.6)

corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    passes=20,
    random_state=42
)

Path("../data/topic_model").mkdir(parents=True, exist_ok=True)

topics_rows = []

print("\n=== TOPICS ===")
for topic_id, words in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
    top_words = [word for word, weight in words]
    print(f"Topic {topic_id}: {', '.join(top_words)}")

    topics_rows.append({
        "topic_id": topic_id,
        "top_words": ", ".join(top_words),
        "label": ""
    })

topics_df = pd.DataFrame(topics_rows)
topics_df.to_csv("../data/topic_model/topics.csv", index=False, encoding="utf-8-sig")

song_topic_rows = []

for i, bow in enumerate(corpus):
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)

    row = {
        "song_id": df.loc[i, "song_id"],
        "title": df.loc[i, "title"],
        "year": df.loc[i, "year"],
        "album": df.loc[i, "album"],
    }

    for topic_id, prob in topic_distribution:
        row[f"topic_{topic_id}"] = prob

    song_topic_rows.append(row)

song_topics_df = pd.DataFrame(song_topic_rows)
song_topics_df.to_csv("../data/topic_model/song_topics.csv", index=False, encoding="utf-8-sig")

lda_model.save("../data/topic_model/lda_model.gensim")

print("\nSaved:")
print("../data/topic_model/topics.csv")
print("../data/topic_model/song_topics.csv")
print("../data/topic_model/lda_model.gensim")