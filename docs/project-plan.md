# Project Plan: Style Analysis of Shlomo Artzi

## Overview

The goal of this project is to detect turning points in the writing style of Shlomo Artzi across his career, using computational text analysis. The corpus consists of 167 songs from 13 albums spanning 1970–2025. The project requires both classical stylometry methods (based on published papers) and AI-based approaches, followed by a comparison and a written research report.

---

## What Has Already Been Done

- **Corpus collected**: 167 song lyrics saved as individual text files (`songs/songs_lyrics/001.txt` … `167.txt`)
- **Metadata**: Excel file (`songs/songs_summary.xlsx`) with song ID, title, year, and album for all 167 songs
- **Biography**: `songs/shlomo_artzi_bio.txt` with detailed background on the artist
- **Report skeleton**: `final_project.docx` already contains:
  - Author description and career overview
  - Research question and hypothesis
  - Literature review (Gómez-Adorno et al. 2018; Alsudais & Tchalian)

### Known data issues to address
- 9 files contain non-Hebrew characters (French phrase, encoding artifacts): `023.txt`, `027.txt`, `034.txt`, `061.txt`, `069.txt`, `071.txt`, `075.txt`, `091.txt`, `138.txt`
- 1 file is empty: `120.txt` — needs to be filled or removed from analysis
- Average song length: ~152 words (min 0, max 368)

---

## Work Tracks

The project splits into three parallel tracks after corpus cleaning. Track C (report) can begin immediately in parallel with everything else.

```
[Corpus Cleaning] ──┬──> [Track A: Classical NLP & Classifiers]  ──┐
                    ├──> [Track B: AI & Embeddings]                 ├──> [Merge & Report]
                    └──> [Track C: Report writing] (starts now)  ───┘
```

---

## Track A — Classical NLP & Stylometric Classifiers

**Who**: 1–2 people comfortable with Python and scikit-learn.

### A1. Corpus Cleaning & Normalization *(shared prerequisite — do first together)*

Before any analysis, the raw text must be cleaned.

**What to do:**
- Load each lyrics file and its metadata from the Excel
- Remove or flag the 9 files with non-Hebrew noise (decide: strip non-Hebrew chars, or exclude song from analysis)
- Handle the 1 empty file (120.txt) — find its lyrics online and fill it, or exclude it
- Normalize text: strip extra whitespace, normalize Hebrew punctuation (geresh, maqaf), remove line numbers or stage directions if any
- Build a single unified data structure: a list of dicts, each with `{song_id, title, year, album, text_raw, text_clean}`
- Save as `data/corpus.json` (or a pandas DataFrame saved as `data/corpus.csv`)

**Output**: `data/corpus.json` — clean, indexed corpus ready for all downstream steps.

---

### A2. Morphological & Syntactic Analysis (DictaBERT)

Hebrew is a morphologically rich language — you cannot just split on spaces and count words. You need a morphological analyzer to get lemmas (base forms), POS tags, and syntactic structure.

**Recommended tool**: [DictaBERT tiny joint model](https://huggingface.co/dicta-il/dictabert-tiny-joint) — runs morphological + syntactic analysis together.

**What to do:**
- Install: `pip install transformers torch`
- Load the model and run it on each song's clean text
- For each token, extract:
  - **Lemma** (base form of the word)
  - **POS tag** (noun, verb, adjective, preposition, function word, etc.)
  - **Dependency relation** (subject, object, modifier, etc.)
- Save the parsed output per song: `data/parsed/001.json`, etc.
- Also run **Named Entity Recognition (NER)** — DictaBERT has a NER variant; use it to identify person names, places, organizations in songs (useful for thematic analysis)

**Practical tip**: The tiny model is fast enough to run on CPU. Running 167 songs should take 5–15 minutes. If it's too slow, batch the inputs.

**Output**: `data/parsed/` — one JSON per song with token-level morphological and syntactic annotations.

---

### A3. Stylometric Feature Extraction

Once you have POS tags and lemmas, you can compute numerical features for each song that capture *how* it is written, not just *what* it says.

**What to do:**

For each song, compute the following features and save them to a feature matrix (`data/features.csv`, one row per song):

**Lexical features:**
- **Type-Token Ratio (TTR)**: unique words / total words — measures vocabulary diversity. High TTR = more varied vocabulary.
- **MTLD** (Measure of Textual Lexical Diversity): a more robust version of TTR that handles text length variation. Use the `lexicalrichness` Python package.
- **Average word length** (in characters)
- **Average sentence/line length** (in words)
- **Vocabulary size** (number of unique lemmas)

**Syntactic / morphological features:**
- **POS tag distribution**: proportion of nouns, verbs, adjectives, adverbs, prepositions, conjunctions, pronouns in the song
- **Verb-to-noun ratio**
- **Proportion of function words** (prepositions + conjunctions + pronouns) — often the most reliable stylometric signal
- **Proportion of first-person pronouns** (אני, אנחנו, לי, אותי) — measures personal/confessional tone

**Structural features:**
- **Number of lines**
- **Number of stanzas** (blank-line-separated blocks)
- **Repetition rate** (proportion of lines that repeat verbatim — chorus detection)

**Output**: `data/features.csv` — feature matrix with one row per song, columns = features, plus metadata columns (year, album).

---

### A4. Classifiers — Period Detection (Gómez-Adorno method)

This implements the approach from the first paper: train a classifier to distinguish between career periods, then measure how well the features separate them.

**What to do:**

1. **Define periods**: Split the 13 albums into three chronological groups:
   - Early (1970–1979): שלמה ארצי, את ואני, משחקי 26, גבר הולך לאיבוד, דרכים
   - Middle (1988–2002): חום יולי אוגוסט, ירח, שניים, צימאון
   - Late (2007–2025): שפויים, אושר אקספרס, קצפת, אותיות נחמה

   Note: there is a gap in the corpus (1980–1987) — this is expected and should be mentioned in the report.

2. **Train classifiers** using scikit-learn:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.model_selection import cross_val_score
   ```
   - Use the feature matrix from A3 as input, period labels as target
   - Run 5-fold cross-validation (important: the corpus is small, so don't do a simple train/test split)
   - Try both Logistic Regression and SVM (RBF kernel)
   - Try different feature subsets: lexical only, syntactic only, all features — compare accuracy

3. **Interpret the model**: Look at which features have the highest coefficients in the Logistic Regression — these are the features that changed most between periods. Report these in the paper.

4. **Automatic periodization (Alsudais method)**: Instead of using preset period labels, detect breakpoints from the data itself:
   - Sort songs by year
   - Use a sliding window to compute similarity between consecutive time windows (cosine similarity of feature vectors)
   - Find the years where similarity drops sharply — these are the turning points
   - Compare detected turning points to known career events (late 1970s transition, etc.)

**Output**: Classifier accuracy table, feature importance plot, turning point chart — all feed into the report.

---

## Track B — AI & Embeddings

**Who**: 1–2 people comfortable with HuggingFace and Python.

### B1. Topic Modeling (LDA)

LDA (Latent Dirichlet Allocation) discovers hidden thematic topics in the corpus without supervision. Each song gets a distribution over topics (e.g., 30% love, 50% nostalgia, 20% war), and you can track how topic proportions shift over time.

**What to do:**
- Use the lemmatized text from A2 (or run a simpler tokenizer if A2 isn't ready yet — split on spaces and strip vowel marks)
- Follow the [Hebrew LDA guide](https://www.cs.bgu.ac.il/~elhadad/nlpproj/LDAforHebrew.html) linked in the project spec
- Use `gensim` library:
  ```python
  from gensim.models import LdaModel
  from gensim.corpora import Dictionary
  ```
- Train with k=8–12 topics (experiment with different k values)
- For each topic, identify the top 10 words and give it a human-readable label (love, war, nature, family, etc.)
- For each song, record its topic distribution vector
- Aggregate topic distributions by album/year and visualize as a heatmap over time

**Output**: `data/topic_model/` — trained LDA model, per-song topic distributions, topic labels. Topic heatmap figure for the report.

---

### B2. Sentence Embeddings (DictaLM / DictaBERT)

Instead of hand-crafted features, use a pre-trained Hebrew language model to produce a dense vector representation of each song. These embeddings capture semantic meaning and style simultaneously.

**What to do:**
- Use [DictaBERT](https://huggingface.co/dicta-il/dictabert) to generate embeddings:
  ```python
  from transformers import AutoTokenizer, AutoModel
  import torch

  model_name = "dicta-il/dictabert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModel.from_pretrained(model_name)

  # For each song, encode text and take mean of last hidden states as embedding
  ```
- Since songs may exceed the model's token limit (512 tokens), split long songs into chunks and average the chunk embeddings
- Save all song embeddings as `data/embeddings.npy` (numpy array, shape 167 × 768)

**Output**: `data/embeddings.npy` — one 768-dimensional vector per song.

---

### B3. Clustering & Visualization

Use the embeddings to find natural groupings in the data without using year labels.

**What to do:**
1. **Dimensionality reduction for visualization**:
   - Apply PCA then t-SNE (or UMAP) to reduce 768 dimensions to 2D
   - Plot songs as points, colored by album or by decade
   - Visual clusters that correspond to career periods = strong evidence of style change

2. **K-means clustering**:
   - Try k=3 (matching the three manual periods)
   - Measure cluster purity: how well do the discovered clusters align with the manual period labels?
   - Try different k values and use the elbow method to find the natural number of clusters

3. **Hierarchical clustering**:
   - Build a dendrogram of all songs by embedding similarity
   - The dendrogram structure often reveals natural groupings and transitions more clearly than k-means

**Output**: t-SNE scatter plot, dendrogram, cluster-to-period alignment table — all for the report.

---

### B4. LLM Style Description

Use a large language model to characterize the style of each period qualitatively, complementing the quantitative analysis.

**What to do:**
- For each of the three periods, take a sample of 5–10 songs
- Send the lyrics to [DictaLM 2.0](https://huggingface.co/spaces/dicta-il/dictalm2.0-instruct-demo) (Hebrew LLM) or GPT-4 with a Hebrew prompt asking:
  > "קרא את מילות השירים הבאים של שלמה ארצי מתקופת [X]. תאר את סגנון הכתיבה: אוצר המילים, הנושאים, הטון, הצורה הפואטית, ואיך הסגנון הזה שונה מהשירים המוקדמים/המאוחרים יותר שלו."
- Collect the LLM's descriptions and quote/summarize them in the report
- Compare what the LLM identifies qualitatively to what the classifiers identify quantitatively — do they agree?

**Output**: Qualitative style descriptions per period, comparison table for the report.

---

## Track C — Report Writing

**Who**: 1 person, can start immediately and update sections as results arrive from Tracks A and B.

The report skeleton already exists in `final_project.docx`. The following sections still need to be written:

### C1. Corpus Statistics Section *(can write now)*

Describe the data:
- Total songs, albums, years covered
- Distribution of songs per album (bar chart)
- Average song length per album (bar chart)
- Note the gap in the corpus (1980–1987) and explain why (those albums aren't included)
- Mention the 9 files with non-Hebrew characters and how they were handled

### C2. Expand Literature Review *(can write now)*

The two required papers are already reviewed. Add 2–3 more recent papers:
- Search Google Scholar for: "stylometry Hebrew", "authorship attribution Hebrew NLP", "computational stylistics singer-songwriter", "lyric analysis over time"
- Summarize each paper in 1–2 paragraphs following the same format as the existing reviews

### C3. Methods Section *(write as Track A/B proceed)*

Describe each method used, in enough detail that someone could reproduce it:
- Feature extraction: what features, why these features
- Classifier setup: which models, cross-validation strategy, evaluation metric
- Periodization: how breakpoints were detected
- Embedding approach: which model, how embeddings were generated
- Clustering: which algorithm, how k was chosen

### C4. Results Section *(write after Track A/B finish)*

- Classifier accuracy table (feature sets × model types)
- Most discriminative features (with interpretation)
- Detected turning points (years) vs. known career events
- Topic heatmap over time
- t-SNE / PCA visualization of songs
- Comparison: where do classical and AI methods agree/disagree?

### C5. Conclusions *(write last)*

- Were turning points detected? Do they match the hypothesis (late 1970s transition)?
- Which method worked best?
- Limitations: small corpus, uneven album sizes, the 1980–1987 gap
- Suggestions for future work

---

## Suggested Division by Group Size

### 2 people
| Person | Tracks |
|--------|--------|
| Person 1 | A (all of classical NLP + classifiers) + corpus cleaning |
| Person 2 | B (all of AI/embeddings) + C (report writing) |

### 3 people
| Person | Tracks |
|--------|--------|
| Person 1 | A1 (corpus cleaning, shared) + A2–A3 (NLP + features) |
| Person 2 | A1 (corpus cleaning, shared) + A4 (classifiers + periodization) + B1 (LDA) |
| Person 3 | B2–B4 (embeddings, clustering, LLM) + C (report writing) |

### 4 people
| Person | Tracks |
|--------|--------|
| Person 1 | A1 (corpus cleaning) + A2–A3 (NLP + features) |
| Person 2 | A4 (classifiers + periodization) |
| Person 3 | B1–B4 (all AI track) |
| Person 4 | C (report writing + all visualizations) |

---

## Suggested Tools & Libraries

```
pip install transformers torch           # DictaBERT
pip install gensim                        # LDA topic modeling
pip install scikit-learn                  # classifiers
pip install pandas numpy matplotlib seaborn  # data + viz
pip install openpyxl                      # read Excel metadata
pip install lexicalrichness               # MTLD and vocab richness
pip install umap-learn                    # UMAP dimensionality reduction (optional)
```

---

## File Structure (suggested)

```
MINI/
├── songs/
│   ├── songs_lyrics/          # raw lyrics (001.txt … 167.txt)
│   ├── songs_summary.xlsx     # metadata
│   └── shlomo_artzi_bio.txt
├── data/
│   ├── corpus.json            # cleaned, unified corpus (output of A1)
│   ├── parsed/                # DictaBERT output per song (output of A2)
│   ├── features.csv           # stylometric feature matrix (output of A3)
│   ├── topic_model/           # LDA model + topic distributions (output of B1)
│   └── embeddings.npy         # DictaBERT embeddings (output of B2)
├── notebooks/
│   ├── A1_corpus_cleaning.ipynb
│   ├── A2_dictabert_parsing.ipynb
│   ├── A3_feature_extraction.ipynb
│   ├── A4_classifiers.ipynb
│   ├── B1_topic_modeling.ipynb
│   ├── B2_embeddings.ipynb
│   └── B3_clustering.ipynb
├── figures/                   # all output charts/plots
├── docs/
│   └── project-plan.md        # this file
├── final_project.docx         # research report
└── project.docx.md            # original assignment spec
```

---

## Dependency Order (what to do first)

1. **Everyone**: Do A1 (corpus cleaning) together — 1–2 hours, unblocks everything
2. **In parallel**:
   - Track A person starts A2 (DictaBERT parsing) — takes a few hours including model download
   - Track B person starts B1 (LDA topic modeling) — can start with raw text, doesn't need A2
   - Track C person starts C1 + C2 (corpus stats + literature review) — starts immediately
3. Once A2 is done: A3 (feature extraction) can start
4. Once A3 is done: A4 (classifiers) can start; B2 (embeddings) can also start independently of A3
5. Once A4 + B3 are done: results are available for C4 (results section)
6. Final week: everyone contributes to merging results, comparison, and conclusions
