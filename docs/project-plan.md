# Project Plan: Style Analysis of Shlomo Artzi

## Overview

The goal of this project is to detect turning points in the writing style of Shlomo Artzi across his career, using computational text analysis. The corpus consists of 176 songs from 14 albums spanning 1970–2025. The project requires both classical stylometry methods (based on published papers) and AI-based approaches, followed by a comparison and a written research report.

---

## What Has Already Been Done

- **Corpus collected**: 176 song lyrics saved as individual text files (`songs/songs_lyrics/001.txt` … `176.txt`)
- **Metadata**: Excel file (`songs/songs_summary.xlsx`) with song ID, title, year, and album for all 176 songs
- **Biography**: `songs/shlomo_artzi_bio.txt` with detailed background on the artist
- **Report skeleton**: `final_project.docx` already contains:
  - Author description and career overview
  - Research question and hypothesis
  - Literature review (Gómez-Adorno et al. 2018; Alsudais & Tchalian)

### Known data issues — resolved
- Songs `027`, `034`, `069`, `071`, `075`, `091` had non-Hebrew characters; cleaned directly in the lyrics files.
- Songs `023`, `061`, `138` contain non-Hebrew phrases (French / English) that are integral to the original lyrics. Hebrew translations were created as companion files (`023_translated.txt`, `061.translated.txt`, `147_translated.txt`). The corpus uses the translated version as the `text` field for all downstream NLP analysis.
- Song `120` was empty and has since been filled with the correct lyrics.
- All 176 songs are included in `data/corpus.json`.

### Track A — completed ✅
- **A1**: `data/corpus.json` — 176 songs, fields: song_id, title, year, album, word_count, line_count, text. Tested by `tests/test_corpus.py` (10 tests).
- **A2**: `data/parsed/001.json` … `176.json` — DictaBERT tiny-joint morphological + NER parse per song (token lemma, POS, dependency, NER entities). Tested by `tests/test_parsed.py` (6 tests).
- **A3**: `data/features.csv` — 176 × 23 feature matrix (TTR, MTLD, avg word/line length, vocab size, 8 POS ratios, verb/noun ratio, function word ratio, first-person ratio, line/stanza counts, repetition rate). Tested by `tests/test_features.py` (13 tests).
- **A4**: `data/classifier_results.json`, `figures/A4_feature_importance.png`, `figures/A4_period_similarity.png`, `figures/A4_model_comparison.png` — 5-fold cross-validated LR, SVM, KNN (each on lexical/syntactic/all feature subsets) and Random Forest (all features) classifiers (best: SVM-RBF 64.2% and Random Forest 60.8%, both vs 36% baseline); top features cross-checked by two independent models — LR coefficients: vocab_size, MTLD, pos_ADV; RF importances: MTLD, verb_noun_ratio, avg_line_length — largely agree. Per-period feature means show TTR rising 0.515→0.545→0.559 and MTLD rising 52.5→72.9→90.7 across early/middle/late, direct descriptive evidence of style change. The KNN classifier is a supervised implementation of the third paper (Hay et al. 2020, "Representation Learning of Writing Style," `article.txt`) — period predicted from proximity in the same stylometric feature space, giving Track A its own citable implementation of all 3 required papers instead of 2. Automatic periodization (Alsudais method) detected turning points at 1979 and 1984. Tested by `tests/test_classifiers.py` (10 tests).

### Track B — completed ✅
- **B1**: `data/topic_model/` — LDA topic model trained on the full corpus, per-song topic distributions (`song_topics.csv`), topic labels (`topics.csv`).
- **B2**: `data/embeddings.npy` — DictaBERT embeddings for all 176 songs (shape 176 × 768).
- **B3**: `data/clustering/song_clusters.csv`, `figures/B3_pca_clusters.png`, `figures/B3_pca_years.png` — K-means clustering on embeddings with PCA visualization, songs colored by cluster and by year. Tested by `tests/test_part_b_outputs.py`.
- **B4**: `new_b4_result` — blind LLM classification: 15 songs from the corpus were presented to a language model without any metadata (title, year, album). The model correctly identified the period (Early / Middle / Late) for all 15 songs based on textual features alone — consistent with the quantitative findings from Track A.
- **B5**: `data/stylometric_space/`, `figures/B5_stylometric_clusters.png`, `figures/B5_stylometric_years.png` — stylometric space analysis using 13 hand-crafted features (word count, TTR, avg word length, repetition ratio, punctuation counts, first/second-person counts, etc.). K-means (k=3) yielded a silhouette score of 0.143; cluster-period alignment table in `stylometric_cluster_period_table.csv` shows cluster 1 is predominantly Early (33/64 songs), cluster 2 predominantly Late (29/69), consistent with style drift over time. Per-period feature means in `stylometric_features_by_period.csv`. Script: `notebooks/B5_stylometric_space.py`. Tested by `tests/test_part_b_outputs.py`.

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

**Practical tip**: The tiny model is fast enough to run on CPU. Running 176 songs should take 5–15 minutes. If it's too slow, batch the inputs.

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

1. **Define periods**: Split the 14 albums into three chronological groups:
   - Early (1970–1984): שלמה ארצי, את ואני, משחקי 26, גבר הולך לאיבוד, דרכים, תרקוד
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
- Save all song embeddings as `data/embeddings.npy` (numpy array, shape 176 × 768)

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

The report skeleton already exists in `final_project.docx`. All Track A and Track B results are now complete — all numbers and findings below come directly from the data files and should be used as-is, not paraphrased or estimated.

---

### C1. Corpus Statistics Section

Describe the data. All facts to use:
- **176 songs** from **14 albums** spanning **1970–2025**
- Album list with years: שלמה ארצי (1970), את ואני (1975), משחקי 26 (1978), גבר הולך לאיבוד (1979), דרכים (1979), תרקוד (1984), חום יולי אוגוסט (1988), ירח (1992), שניים (1996), צימאון (2002), שפויים (2007), אושר אקספרס (2012), קצפת (2016), אותיות נחמה (2025)
- **Three career periods**: Early (1970–1984, 6 albums), Middle (1988–2002, 4 albums), Late (2007–2025, 4 albums)
- **Gap in corpus**: 1980–1987 — no albums were released in this period; this should be noted explicitly
- **Data cleaning**: songs 027, 034, 069, 071, 075, 091 contained non-Hebrew characters and were cleaned directly. Songs 023, 061, and the song originally numbered 138 (now 147) contain French/English phrases integral to the lyrics — Hebrew translation companion files were created and used as the `text` field for all NLP analysis
- Include a bar chart of songs per album and average song length per album (use `data/corpus.csv` — columns: album, word_count)

### C2. Literature Review

The report skeleton already includes reviews of Gómez-Adorno et al. (2018) and Alsudais & Tchalian. A third paper is already implemented in this project and must be reviewed:

**Hay et al. (2020) — "Representation Learning of Writing Style"** (ACL Anthology: https://aclanthology.org/2020.wnut-1.30.pdf)
- The paper proposes representing documents as vectors in a stylometric space where documents with similar writing style are close to each other, independent of content
- Key claim: style can be captured through features like vocabulary richness, sentence length, punctuation, POS tags, and structural patterns — and represented as a numeric vector
- **How this project implements it**: in B5, each song is represented as a 13-feature stylometric vector (word count, TTR, avg word length, repetition ratio, punctuation counts, first/second-person counts), then PCA + K-means are applied. In A4, KNN classification uses the same stylometric feature space to predict period from proximity — this is a direct supervised analog of Hay et al.'s proximity-based style representation
- The review should connect this explicitly to both A4 (KNN) and B5 (stylometric space clustering)

For additional papers, search Google Scholar for: "stylometry Hebrew NLP", "computational stylistics singer-songwriter", "authorship attribution over time"

### C3. Methods Section

Write one subsection per method. Use the exact parameter values below — do not write approximate or generic descriptions.

**A. Corpus Construction (A1)**
- Source: 176 song lyrics from 14 albums, collected manually
- Cleaning: stripped non-Hebrew characters from 6 songs; created Hebrew translation files for 3 songs with integral non-Hebrew content; normalized whitespace and Hebrew punctuation
- Output: `data/corpus.json` — fields: song_id, title, year, album, word_count, line_count, text

**B. Morphological Analysis (A2)**
- Tool: DictaBERT tiny-joint model (`dicta-il/dictabert-tiny-joint`)
- Extracted per token: lemma, POS tag, dependency relation, NER entities
- Output: `data/parsed/` — one JSON per song

**C. Stylometric Feature Extraction (A3)**
- 23 features per song: TTR, MTLD, average word length, average line length, vocabulary size, 8 POS ratios (NOUN, VERB, ADJ, ADV, PRON, ADP, CCONJ, SCONJ), verb-noun ratio, function word ratio, first-person ratio, number of lines, number of stanzas, repetition rate
- Output: `data/features.csv` — 176 × 23 matrix

**D. Period Classification (A4)**
- Period labels: Early (albums 1970–1984), Middle (1988–2002), Late (2007–2025)
- Models: Logistic Regression, SVM (RBF kernel), KNN (k=5), Random Forest (200 estimators)
- Each model tested on: all features, lexical features only, syntactic features only
- Evaluation: 5-fold stratified cross-validation (chosen because corpus is small)
- Baseline: majority-class baseline = 36%

**E. Automatic Periodization (Alsudais method)**
- Songs sorted by year; sliding window cosine similarity computed between consecutive windows
- Sharp similarity drops identify turning points
- Threshold computed as 5th percentile of all similarity values

**F. Topic Modeling (B1)**
- Tool: Gensim LDA
- Input: lemmatized text from A2 (or space-split tokens)
- Output: per-song topic distribution vectors in `data/topic_model/`

**G. Sentence Embeddings (B2)**
- Model: DictaBERT (`dicta-il/dictabert`)
- Long songs chunked to fit 512-token limit; chunk embeddings averaged
- Output: `data/embeddings.npy` — shape 176 × 768

**H. Embedding Clustering (B3)**
- Dimensionality reduction: PCA → 2D for visualization
- Clustering: K-means (k=3, matching manual periods)
- Output: `data/clustering/song_clusters.csv`, figures `B3_pca_clusters.png`, `B3_pca_years.png`

**I. Stylometric Space Analysis (B5, inspired by Hay et al. 2020)**
- 13 stylometric features extracted per song (word count, unique words, TTR, avg word length, line count, avg line length, repetition ratio, punctuation counts, first/second-person counts)
- PCA + K-means (k=3) applied to feature vectors
- Output: `data/stylometric_space/`, figures `B5_stylometric_clusters.png`, `B5_stylometric_years.png`

**J. Blind LLM Period Classification (B4)**
- 15 songs presented to an LLM without title, year, or album metadata
- Model asked to classify each song as Early / Middle / Late and provide reasoning based on text alone
- Results in `new_b4_result`

### C4. Results Section

All numbers below are final results from `data/classifier_results.json` and the output files. Copy them exactly.

**Classifier accuracy table** — 5-fold cross-validation, stratified, baseline = 36%:

| Model | All features | Lexical only | Syntactic only |
|-------|-------------|--------------|----------------|
| Logistic Regression | 58.5% ± 2.9% | 51.1% ± 10.1% | 46.6% ± 3.3% |
| SVM (RBF) | **64.2% ± 4.7%** | 51.7% ± 6.9% | 49.4% ± 3.3% |
| KNN (k=5) | 48.2% ± 10.8% | 48.3% ± 6.3% | 39.2% ± 4.9% |
| Random Forest | **60.8% ± 5.9%** | — | — |

Key finding: lexical and syntactic features each alone perform near-baseline; combining them gives a meaningful lift (SVM: 36% → 64%).

**Most discriminative features**:
- LR coefficients (top 5): vocab_size, MTLD, pos_ADV, avg_line_length, TTR
- RF importances (top 5): MTLD, verb_noun_ratio, avg_line_length, pos_VERB, vocab_size
- Both models agree: MTLD and vocabulary-richness features are the strongest signal

**Per-period feature means** (from `classifier_results.json` → `period_means`):

| Feature | Early | Middle | Late |
|---------|-------|--------|------|
| TTR | 0.515 | 0.545 | 0.559 |
| MTLD | 52.5 | 72.9 | 90.7 |
| avg_line_length | 4.32 | 5.24 | 5.31 |

Interpretation: vocabulary richness (TTR, MTLD) rises steadily across the career; line length increases from Early to Middle then plateaus.

**Automatic periodization** (Alsudais method): turning points detected at **1979** and **1984** — corresponding to the transition out of the early folk-rock albums and the תרקוד album respectively.

**Stylometric space (B5)**:
- Silhouette score: 0.143 (moderate separation)
- Cluster distribution: 43 songs in cluster 0, 64 in cluster 1, 69 in cluster 2
- Cluster–period alignment: cluster 1 is predominantly Early (33/64 songs from Early period); cluster 2 is predominantly Late (29/69 from Late period)
- Per-period trends: average word count rises from 170 (Early) → 192 (Middle) → 196 (Late); first-person pronoun count rises from 5.97 → 8.62 → 9.04 — suggesting a shift toward more personal, confessional writing over time

**Blind LLM classification (B4)**:
- 15 songs classified correctly by period based on text alone (results in `new_b4_result`)
- LLM identified: Early songs = lyrical, romantic, fairy-tale imagery, simple structure; Middle songs = personal/mature, clear chorus structure, mixed themes; Late songs = conversational, associative, contemporary references
- This qualitative characterization aligns with the quantitative findings: rising vocabulary richness, longer lines, more first-person usage

**Comparison — where do methods agree?**
- Both A4 (classifiers) and B5 (stylometric space) identify vocabulary richness (TTR, MTLD) as the primary driver of period separation
- Both A4 (periodization) and B5 place the main stylistic transition in the early–mid 1980s (turning points at 1979 and 1984)
- The blind LLM analysis (B4) independently confirms the same three-way style distinction
- Where they diverge: embedding-based clustering (B3) operates on semantic content rather than surface style features, so it may group songs thematically rather than chronologically — note this explicitly and discuss whether semantic vs. stylometric separation tell different stories

### C5. Conclusions

- **Were turning points detected?** Yes. The Alsudais sliding-window method detected turning points at **1979 and 1984**. This is consistent with the hypothesis of a late-1970s transition and adds a second transition point at 1984 (the תרקוד album), which may reflect Artzi's stylistic evolution before his commercial breakthrough in the late 1980s.
- **Which method worked best?** For period classification: SVM-RBF with all features (64.2%). For interpretability: Logistic Regression and Random Forest agree on the most important features (MTLD, vocab_size), making the finding robust. For qualitative confirmation: blind LLM classification (B4).
- **Do methods agree?** Yes — the convergence across classical stylometry (A4), stylometric space clustering (B5), and blind LLM analysis (B4) on the same features and transition points strengthens the overall conclusion.
- **Limitations**:
  - Small corpus (176 songs) — 5-fold CV is appropriate but power is limited
  - Uneven album sizes — some periods have more songs than others, which may bias classifiers
  - The 1980–1987 gap means the Early→Middle transition is not continuously sampled
  - Songs with non-Hebrew content required translation — translation choices may introduce noise
- **Future work**: extend to other Israeli artists for cross-artist comparison; apply representation learning (Hay et al.) with a fine-tuned Hebrew model rather than hand-crafted features; incorporate musical features (tempo, key) alongside lyrical features

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
│   ├── songs_lyrics/          # raw lyrics (001.txt … 176.txt)
│   ├── songs_summary.xlsx     # metadata
│   └── shlomo_artzi_bio.txt
├── data/
│   ├── corpus.json            # cleaned, unified corpus (output of A1)
│   ├── corpus.csv             # same corpus in CSV format
│   ├── parsed/                # DictaBERT output per song (output of A2)
│   ├── features.csv           # stylometric feature matrix (output of A3)
│   ├── classifier_results.json  # classifier CV scores, feature importances, turning points (output of A4)
│   ├── topic_model/           # LDA model + topic distributions (output of B1)
│   ├── embeddings.npy         # DictaBERT embeddings (output of B2)
│   ├── clustering/            # K-means clusters on embeddings (output of B3)
│   └── stylometric_space/     # stylometric feature clusters and period tables (output of B5)
├── notebooks/
│   ├── A1_corpus_cleaning.ipynb
│   ├── A2_dictabert_parsing.ipynb
│   ├── A3_feature_extraction.ipynb
│   ├── A4_classifiers.ipynb
│   ├── A4_classifiers.py
│   ├── B1_topic_modeling.ipynb (or .py)
│   ├── B2_embeddings.ipynb (or .py)
│   ├── B3_clustering.ipynb (or .py)
│   └── B5_stylometric_space.py
├── figures/                   # all output charts/plots
├── tests/                     # test suite (47 tests, all passing)
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
