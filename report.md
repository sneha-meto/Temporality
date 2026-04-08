# Temporal Signals of Deception: Tense and Event Structure in Fake vs Verified News

Sneha Meto · 25337472 · metos@tcd.ie  
Mahati Rane · 25334671 · mrane@tcd.ie  
Albin Binu · 25353208 · binua@tcd.ie  
Bhathrinaathan Muthuramalingam Bhanumathi · 25362507 · muthurab@tcd.ie

April 2026

---

## Abstract

This paper investigates whether fake news articles display distinctive temporal behaviour compared to verified news, focusing on tense usage, temporal expressions, event sequencing, and cross-sectional tense consistency in English news discourse. We combine insights from register variation, narrative tense theory, and fake news linguistics to derive four hypotheses: that fake news exhibits (i) more temporal inconsistencies, (ii) more frequent tense shifts and a distinct tense distribution, (iii) poorer alignment between narrated and implied event sequences, and (iv) greater tense misalignment between structural sections — titles, leads, and bodies — than legitimate news reporting. Using the ISOT Fake News Dataset, we implement a full five-phase computational pipeline covering preprocessing, spaCy-based tense and aspect extraction, regex-based temporal expression tagging, embedding and entity-grid coherence scoring, and statistical analysis with classification experiments. H2 is strongly supported: fake news uses significantly more present tense while real news is past-tense dominant, with large effect sizes (r up to 0.52, all p < 0.001). H4 is strongly supported: fake news bodies and leads are significantly more present-tense dominant than past-tense (body excess: fake 0.288 vs real 0.034, p < 0.001, r = 0.48), while real news leads are near-neutral or past-dominant — confirming that fake news fails to make the conventional register shift from headline to body. H1 shows significant differences in all features but in directions opposite to predictions, suggesting real news contains more temporal complexity rather than fake news containing more inconsistencies. H3 is not supported: misalignment–coherence correlations are statistically significant but trivially small (|r| < 0.09). A Random Forest classifier using only temporal features achieves 0.89 AUC, confirming that temporal structure carries substantial discriminative signal beyond topical content.

---

## 1 Introduction

Fake news has become a major concern for public discourse, driving a fast-growing body of research on its sources, spread, and linguistic properties. Beyond topic and vocabulary, recent work treats fake news as a distinct linguistic register with systematic grammatical properties (Grieve and Woodfield, 2023). Differences in communicative purpose between deceptive and informative news — manipulation versus information — are reflected in characteristic patterns of lexis, stance, and syntactic complexity.

The temporal organisation of fake news, however, remains underexplored. Work in discourse analysis and narratology shows that tense, aspect, and temporal connectives are central for structuring narrative time and supporting global coherence (Fleischman, 1990; Becker and Genz, 2019). News reports typically combine a narrative of past events with more atemporal or generic material, and use tense and temporal expressions to guide readers through this structure. Grieve and Woodfield (2023) observe that fake news patterns more like involved spoken registers, with higher rates of present tense, suggesting a fundamental difference in temporal stance.

Computational approaches to fake news detection show that coherence-based and stylistic features can distinguish fake from real articles (Ahmed et al., 2017; Pérez-Rosas et al., 2018; Singh et al., 2020), but they generally model coherence at a global level without isolating temporal structure as a dimension of its own. We ask whether fake and real news differ systematically in how they manage temporal information — focusing on tense distributions across articles and sections, tense shift patterns, temporal inconsistencies, and alignment between narrated event order and implied temporal sequence.

This paper presents a complete empirical study addressing these questions on the ISOT dataset. We find that the most robust signal lies in structural tense misalignment: fake news routinely employs present-tense headlines over past-tense bodies, a pattern absent in verified reporting, and uses present and future tense disproportionately throughout. These findings contribute to theoretical models of tense and deceptive discourse and inform the design of temporally-aware fake news detection systems.

---

## 2 Literature Review

### 2.1 Fake news, coherence, and temporal structure

A first line of work treats fake news as a text classification problem. Ahmed et al. (2017) introduce the ISOT Fake News Dataset and show that n-gram features and standard classifiers distinguish fake from real news with high accuracy, establishing a strong lexical baseline. Pérez-Rosas et al. (2018) reach similar conclusions using stylistic and psycholinguistic features from LIWC, demonstrating that surface linguistic cues carry substantial discriminative signal.

Grieve and Woodfield (2023) propose a register-based view of fake news. In a study of authentic and fabricated articles, fake texts pattern more like involved spoken registers, with higher rates of present tense, stance markers, and evaluative lexis, whereas legitimate news aligns with informational written registers. This register framing motivates our focus on tense as a systematic grammatical property rather than a surface stylistic accident.

Coherence has also been examined directly. Singh et al. (2020) compare coherence scores based on sentence embeddings and entity grids between fake and legitimate articles in ISOT, finding that fake news is on average less coherent and more variable. Fernandes et al. (2025) examine how linguistic features of misinformation evolve over time using temporal point processes and longitudinal corpora, showing that some features remain stable while others shift as topics and platforms change. Their focus is on change across calendar time, not on temporal organisation within individual texts — the gap our study addresses.

### 2.2 Tense, temporal marking, and discourse

Work in corpus linguistics and narratology emphasises the central role of tense and aspect in structuring discourse. Biber (1988) shows that tense distributions and aspectual choices are key parameters differentiating conversational, narrative, and informational registers. Past tense clusters with informational written registers; present tense with involved spoken ones. Fleischman (1990) and Elson and McKeown (2010) argue that narrative tense patterns are tied to foreground–background structure and the organisation of narrative time.

Computational narratology has operationalised these ideas. Bögel et al. (2014) introduce tense clusters — contiguous segments of consistent tense within sentence-level structures — and show that they can identify narrative segments and transitions. TimeML and related frameworks formalise event–time relations and underpin temporal information extraction systems (Pustejovsky et al., 2003).

In news discourse specifically, Becker and Genz (2019) show how tense and temporal deictic expressions guide readers through narrative time and viewpoint. Corpus-based work (Ayman Hamad Rlneil Hamdan, 2016) documents pervasive use of simple present in headlines to convey immediacy, contrasting with the predominantly past-tense bodies of news reports. Krejčír (2021) describes systematic tense shifts in indirect speech in newspaper reports. These structural conventions of legitimate news discourse provide a baseline against which fake news deviations become measurable.

Temporal expressions and connectives also structure event sequences. Temporal connectives such as *before*, *after*, *then*, *meanwhile*, and *later* provide cues about event order (Murayama et al., 2020; Wuyun, 2016). From a coherence perspective, explicit and consistent temporal marking helps readers construct a coherent mental model of event order, whereas sparse or vague temporal signalling can hinder coherence, especially when combined with irregular tense patterns.

### 2.3 Research Gap

Taken together, prior work shows that (i) fake news forms a distinctive register with characteristic grammatical properties including tense usage; (ii) fake news tends to be less coherent than legitimate news; and (iii) tense, aspect, temporal deixis, and temporal connectives are central for structuring narrative time and coherence (Biber, 1988; Fleischman, 1990; Becker and Genz, 2019; Grieve and Woodfield, 2023; Singh et al., 2020). What is missing is a systematic empirical comparison examining temporal inconsistencies relative to publication date, tense distributions and shift patterns within and across article sections, and the cross-sectional structural pattern of headline-body tense misalignment specifically. Our study addresses this gap by operationalising temporal structure as a multi-dimensional feature space and testing how these features correlate with veracity labels and coherence scores.

---

## 3 Research Question

The central research question is:

> **RQ:** Are fake news articles more likely to contain temporal inconsistencies, tense shifts, poorly aligned event sequences, or structural tense misalignment between sections than verified news?

We derive four hypotheses:

**H1 (Temporal inconsistency).** Fake news articles contain more temporal inconsistencies (e.g., conflicting or impossible time references relative to publication date) than verified news articles.

**H2 (Tense shifts).** Fake news articles exhibit more frequent and unexplained shifts of narrative tense across sentences than verified news articles, as reflected in higher tense shift rates, greater tense entropy, and a higher proportion of present and future tense verbs relative to past tense.

**H3 (Event sequencing and coherence).** Fake news articles show poorer alignment between the order of events as narrated in the text and the implied temporal sequence (derived from temporal connectives) than verified news, and this misalignment correlates with lower discourse coherence scores.

**H4 (Structural tense misalignment).** Fake news articles exhibit significantly greater structural tense misalignment than verified news, operationalised as the failure of the body and lead to shift from the present-tense register of the headline into the past-tense register expected of news reporting. Verified news follows the standard journalistic convention — present-tense titles, past-tense bodies — while fake news applies present tense uniformly across all sections, violating this convention.

---

## 4 Research Method

### 4.1 Dataset and sampling

We use the ISOT Fake News Dataset (Ahmed et al., 2017), which consists of two CSV files, `True.csv` and `Fake.csv`, containing real Reuters articles and fake news articles collected from unreliable sources and flagged by fact-checking organisations. Each article includes a title, full text, subject label, and publication date.

| Statistic | Fake | Real | Total |
|---|---|---|---|
| Articles | 15,088 | 20,800 | 35,888 |
| — Train (70%) | 10,561 | 14,560 | 25,121 |
| — Dev (10%) | 1,509 | 2,080 | 3,589 |
| — Test (20%) | 3,018 | 4,160 | 7,178 |
| Word count — mean | 441.5 | 388.5 | |
| Word count — median | 388.0 | 362.0 | |
| Word count — std | 303.5 | 270.5 | |
| Word count — range | 50–4,900 | 50–3,708 | |
| Date range | 2016-01-01 – 2017-12-31 | 2016-01-13 – 2017-12-31 | |

*Table 1: Summary statistics for the preprocessed corpus after cleaning, length filtering, date filtering (2016–2018), and deduplication.*

| Topic (KMeans, k=5) | Fake | Real |
|---|---|---|
| world\_intl | 11,834 (78.4%) | 711 (3.4%) |
| social\_crime | 1,436 (9.5%) | 7,252 (34.9%) |
| other | 599 (4.0%) | 9,921 (47.7%) |
| economy\_finance | 1,127 (7.5%) | 1,724 (8.3%) |
| politics\_us | 92 (0.6%) | 1,192 (5.7%) |

*Table 2: Topic distribution by label after TF-IDF + KMeans clustering (k=5).*

The `subject` column in the raw ISOT files is inconsistently labelled — `True.csv` uses two clean categories (`politicsNews`, `worldnews`) while `Fake.csv` uses a wider, noisier vocabulary. We therefore discard the original subject labels and re-assign topic labels using TF-IDF vectorisation with KMeans clustering (k=5) over the full article text, producing label-agnostic topic buckets. The fitted topic model is saved for reproducibility. The topic distribution reveals a notable imbalance: fake news is heavily concentrated in `world_intl` (78.4%), while real news is more evenly distributed — a confound we control for via stratified splitting.

Basic cleaning removes Reuters boilerplate (city datelines of the form `CITY (Reuters) –`), HTML tags, and duplicate texts. Articles with fewer than 50 or more than 5000 words are discarded. The corpus is restricted to the 2016–2018 time window where both labels are well-represented. Each article is segmented into three structural sections: title, lead (first three sentences of the body), and body (remaining sentences), using a rule-based sentence splitter. The dataset is split 70/10/20 into train, development, and test sets stratified by label and topic, sorted by publication date to avoid temporal leakage.

### 4.2 Tense and aspect feature extraction (H2, H4)

We apply the spaCy dependency parser (`en_core_web_sm` or `en_core_web_trf` if available) for tokenisation, POS tagging, and dependency parsing. From the parsed output, we identify all finite verb tokens by Penn Treebank POS tag (`VBD`, `VBP`, `VBZ`, `VB`, `VBG`, `VBN`, `MD`), excluding auxiliary dependants already classified under their head verb.

Each finite verb is classified into a tense category using POS tags and auxiliary patterns:

- **Past:** `VBD` or `VBN` without a perfect auxiliary
- **Present:** `VBP`, `VBZ`, bare `VB` with conditional modal, or `VBG`
- **Future:** presence of `will`/`shall`/`'ll` auxiliary or *going to* construction

Aspect is classified as simple, progressive (be + VBG), perfect (have + VBN), or perfect-progressive.

For each article and each structural section (title, lead, body), we compute:

- **Tense distributions:** proportions of past, present, and future verbs
- **Tense entropy:** −Σ p(t) log p(t) over tense proportions
- **Dominant tense per sentence:** the majority tense among verbs in each sentence
- **Tense shift rate:** fraction of consecutive sentence pairs where the dominant tense changes
- **Aspect distributions:** proportions of simple, progressive, perfect, perfect-progressive

Sentence-level shift rates, tense entropy, and whole-article tense proportions operationalise **H2**. Cross-sectional features operationalise **H4**. The expected journalistic convention is that titles use present tense (for immediacy) while bodies and leads report in past tense. Misalignment means the body or lead *fails* to make this shift — remaining present-dominant when it should be past-dominant. We operationalise this with three convention-based features computed directly from the sectional tense proportions:

- **`body_present_excess`** = `body_prop_present − body_prop_past`: positive values indicate the body is more present than past (convention violation); negative values indicate correct past-dominance.
- **`lead_present_excess`** = `lead_prop_present − lead_prop_past`: same logic for the lead section.
- **`h4_misalign`** = mean of the above two, as a summary misalignment score.

We also retain the original L1 distances (`align_title_body`, etc.) for reference, but the convention-based features are the primary H4 operationalisation.

### 4.3 Temporal expressions, connectives, and event sequencing (H1, H3)

To address H1 and H3, we implement a pure-Python regex-based temporal tagger (HeidelTime was considered but excluded due to Java dependency and installation fragility). The tagger categorises temporal adverbials as:

- **Present-time:** *now, today, currently, at the moment, this week,* etc.
- **Past-time:** *yesterday, last week, last year, N days ago,* etc.
- **Future-time:** *tomorrow, next week, soon, in the future,* etc.

It also extracts absolute date expressions (e.g., *March 2017*, *2017-03-15*) using a multi-pattern regex covering named months and ISO formats.

**Temporal inconsistency detection (H1)** applies three heuristic rules: (1) a past-anchored relative expression that resolves to a date after the publication date; (2) an absolute date more than 30 days in the future of the publication date; (3) a span of more than 18 months between the earliest and latest absolute dates in the article (indicating conflicting date anchors).

**Temporal connectives** are counted from a lexicon of ordering words (*before, after, then, meanwhile, later, eventually, previously, subsequently, earlier, next*). Density (per 100 words) and type diversity are computed per article.

**Event-narration misalignment (H3)** is measured by examining consecutive sentence pairs in which at least one sentence contains an ordering connective. A pair is flagged as misaligned when the connective in sentence *i* implies a forward temporal direction (e.g., *after, later*) but sentence *i+1* contains a backward connective (e.g., *previously, earlier*), or vice versa. The misalignment score is the proportion of such contradictory pairs.

### 4.4 Coherence scoring (H3)

We estimate discourse coherence using two complementary approaches following Singh et al. (2020):

1. **Embedding-based coherence:** Each sentence is encoded with the `all-MiniLM-L6-v2` sentence-transformer. We compute cosine similarity between every adjacent sentence pair and aggregate by mean (`coherence_emb`), standard deviation (`coherence_emb_std`), and minimum (`coherence_emb_min`). Higher mean similarity indicates more locally coherent text.

2. **Entity-based coherence:** Named entities (extracted via spaCy NER) form the column indices of an entity grid. We compute: entity repetition rate (proportion of entities appearing in more than one sentence), unique entity count, and entity transition probabilities for Subject–Subject (SS) and Subject–Object (SO) transitions, following Barzilay and Lapata (2008).

Spearman correlations between `event_misalignment` and each coherence measure are computed separately for fake and real articles to test H3.

### 4.5 Statistical analysis and classification

All hypothesis tests use Mann-Whitney U (two-sided) as the distributions of temporal features are non-normal and right-skewed. Effect size is reported as rank-biserial correlation *r*. Significance thresholds: \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001. Tests are conducted on the held-out test split.

For classification, we train (i) Logistic Regression and (ii) Random Forest using only temporal features (all tense, aspect, alignment, adverbial, connective, and coherence features), and compare against (iii) a Logistic Regression BoW baseline (TF-IDF, 10,000 features) trained on full article text. All models use class-balanced weights. Performance is reported as accuracy, precision, recall, F1, and AUC on the test set.

---

## 5 Results

### 5.1 H1 — Temporal Inconsistency

| Feature | Mean (Fake) | Mean (Real) | p-value | Effect *r* | Sig |
|---|---|---|---|---|---|
| `temporal_incons` | 0.036 | 0.163 | 2.50e-49 | 0.108 | \*\*\* |
| `abs_date_count` | 0.446 | 0.175 | 1.54e-09 | −0.053 | \*\*\* |
| `abs_future_ratio` | 0.009 | 0.026 | 3.06e-07 | 0.018 | \*\*\* |
| `art_adv_total` | 1.708 | 1.811 | 0.0066 | 0.038 | \*\* |

*Table 3: Mann-Whitney U tests for H1 features.*

**H1 is not supported as predicted; significant differences exist but in the opposite direction.** All four features are now statistically significant, but the directionality is reversed: real news has substantially more temporal inconsistencies (mean 0.163 vs 0.036), more future-oriented expressions, and fewer absolute date references than fake news. Fake news, by contrast, contains significantly more absolute dates (mean 0.446 vs 0.175). These results suggest that real news is temporally richer and more complex — citing specific dates, future events, and a wider temporal range — while fake news uses simpler, more present-anchored temporal language. The regime of inconsistency detected by the regex heuristics therefore appears to capture legitimate journalistic complexity (wide temporal scope, forward-looking references) rather than deceptive temporal distortion. H1 as originally framed does not hold.

### 5.2 H2 — Tense Shifts and Distributions

| Feature | Mean (Fake) | Mean (Real) | p-value | Effect *r* | Sig |
|---|---|---|---|---|---|
| `art_shift_rate` | 0.405 | 0.436 | 1.57e-13 | 0.106 | \*\*\* |
| `art_entropy` | 0.691 | 0.738 | 6.83e-70 | 0.255 | \*\*\* |
| `art_prop_past` | 0.355 | 0.472 | 1.76e-248 | 0.485 | \*\*\* |
| `art_prop_present` | 0.625 | 0.503 | 4.45e-288 | −0.523 | \*\*\* |
| `art_prop_future` | 0.020 | 0.025 | 6.85e-05 | 0.055 | \*\*\* |
| `art_aspect_perfect` | 0.037 | 0.063 | 5.28e-143 | 0.366 | \*\*\* |
| `art_aspect_progressive` | 0.033 | 0.023 | 8.07e-67 | −0.246 | \*\*\* |

*Table 4: Mann-Whitney U tests for H2 features (selected).*

**H2 is strongly supported.** All tense and aspect features are highly significant (p < 0.001). The most striking contrast is in tense proportions: fake news uses present tense for 62.5% of verbs versus 50.3% in real news (*r* = −0.52, the largest effect in this study), while real news uses past tense for 47.2% versus 35.5% in fake news (*r* = 0.49). Real news also uses significantly more perfect aspect (0.063 vs 0.037, *r* = 0.37), consistent with its evidential, retrospective reporting register. Contrary to the original H2 prediction, real news has higher tense entropy (0.738 vs 0.691) and higher shift rates (0.436 vs 0.405): real reporting appropriately varies tense across temporal contexts, whereas fake news is uniformly present-tense, producing lower entropy overall. These results replicate and strongly extend Grieve and Woodfield's (2023) register finding on a full-scale corpus.

### 5.3 H3 — Event Sequencing and Coherence

Spearman correlations between `event_misalignment` and each coherence measure are reported for fake and real groups separately. Many correlations reach statistical significance at large sample sizes (n ≈ 14,600–17,900), but all effect sizes are trivially small (|r| < 0.09). For embedding-based coherence, the correlation is *positive* in fake news (*r* = 0.078, p < 0.001) — the opposite of the predicted direction. Entity uniqueness shows the predicted negative relationship in both groups (fake: *r* = −0.069, real: *r* = −0.091, both p < 0.001), but these effects are too small to be substantively meaningful.

**H3 is not supported.** While many correlations are statistically significant, the effect sizes are negligible and the dominant direction is the reverse of the prediction. The misalignment score remains sparse — most articles contain too few consecutive connective-bearing sentence pairs to produce a reliable signal. These results suggest that connective-based misalignment is not a useful proxy for coherence, and that richer temporal IE would be required to operationalise H3 adequately.

### 5.4 H4 — Structural Tense Misalignment

| Feature | Mean (Fake) | Mean (Real) | p-value | Effect *r* | Sig |
|---|---|---|---|---|---|
| `body_present_excess` | 0.288 | 0.034 | 3.40e-244 | −0.481 | \*\*\* |
| `lead_present_excess` | 0.223 | −0.005 | 1.37e-118 | −0.333 | \*\*\* |
| `h4_misalign` | 0.255 | 0.014 | 3.72e-230 | −0.467 | \*\*\* |

*Table 5: Mann-Whitney U tests for H4 features. Negative effect r indicates fake > real.*

**H4 is strongly supported.** Fake news bodies are on average 28.8 percentage points more present than past (`body_present_excess` = 0.288), meaning the body remains as present-tense heavy as the headline instead of shifting to past tense. Real news bodies are near-neutral (0.034), with leads actually slightly past-dominant on average (−0.005 for `lead_present_excess`). All three features show large, highly significant effects (|r| = 0.33–0.48, all p < 0.001).

These results confirm H4's core claim: fake news fails to make the standard journalistic register shift from present-tense headline to past-tense body. This is the inverse of what the original L1-distance operationalisation detected — raw tense distance between sections is higher in real news because real news *correctly* performs this shift, creating a large title-to-body tense contrast. The convention-based features (`body_present_excess`, `lead_present_excess`) isolate the right signal by measuring deviation from the expected past-tense norm in the body, not mere section-to-section difference. The effect sizes here are comparable to the strongest H2 findings, making H4 one of the two most robust results in this study.

### 5.5 Classification

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogReg (temporal) | 0.806 | 0.779 | 0.792 | 0.786 | 0.886 |
| Random Forest (temporal) | 0.811 | 0.810 | 0.755 | 0.782 | 0.888 |
| LogReg (BoW baseline) | 0.984 | 0.987 | 0.977 | 0.982 | 0.998 |

*Table 6: Classification performance on the test set.*

The BoW baseline achieves near-perfect performance, consistent with Ahmed et al. (2017) — the ISOT dataset has strong lexical separability due to systematic topic and source differences. Both temporal classifiers now achieve over 80% accuracy and 0.89 AUC using only temporal features, a substantial improvement that confirms temporal structure carries meaningful discriminative signal independent of topical content.

The top temporal features by Random Forest importance are: `art_prop_present` (0.058), `body_prop_present` (0.045), `art_prop_past` (0.043), `body_present_excess` (0.042), and `body_prop_past` (0.038). Present and past tense proportions dominate (H2), but `body_present_excess` — the primary H4 feature — ranks fourth, confirming that the structural misalignment signal adds discriminative value beyond raw tense proportions. `h4_misalign` ranks eighth (0.033).

---

## 6 Contribution

This study makes three contributions. First, it operationalises and confirms structural tense misalignment as a feature of fake news (H4): fake news bodies and leads remain present-tense dominant, failing to make the conventional register shift to past tense, with large effect sizes (|r| = 0.33–0.48, all p < 0.001). The key methodological insight is that misalignment should be measured as deviation from the expected past-tense norm in the body — not as raw L1 distance between sections, which conflates legitimate register variation in real news with deceptive uniformity in fake news. Second, it confirms at scale that fake news is present-tense dominant throughout while verified news is past-tense dominant (H2), with effect sizes up to r = 0.52 (all p < 0.001), strongly replicating and extending Grieve and Woodfield (2023). Third, it demonstrates that temporal features alone achieve 0.89 AUC, with the H4 misalignment feature (`body_present_excess`) ranking fourth by importance — establishing a robust, interpretable, non-lexical discriminative baseline.

The reversed results for H1 are also informative: real news is temporally richer — more absolute dates, wider temporal scope — while fake news anchors primarily in the present. This suggests future work should examine temporal sparsity as a fake news signal rather than temporal contradiction. H3 remains unsupported due to sparse connective-based misalignment scoring.

---

## 7 Statement of Contributions

### 7.1 Bhathrinaathan M B (Chair, Accountant)

In the literature review, I concentrated on identifying work that connects fake news with discourse coherence and tense/temporal structure (Section 2.1). I located three key papers and extracted relevant findings, which were integrated into the Literature Review to motivate our focus on temporal organisation. I also contributed to the selection of the ISOT dataset (Section 4.1), comparing available fake-news resources and making the case for ISOT's well-defined time span and metadata. I contributed to refining the central research question and helped formulate the hypotheses concerning temporal inconsistencies, tense shifts, and event sequencing — including drafting the H4 hypothesis on structural tense misalignment.

### 7.2 Sneha Meto (Monitor, Recorder)

I worked on Section 2.2, covering tense, temporal marking, and discourse. Reading Fleischman (1990) reframed tense shifts as rhetorical choices rather than errors; Elson and McKeown (2010) illuminated the complexity of machine tense assignment; Biber (1988) established the empirical grounding for tense as a register-differentiating feature; and Bögel et al. (2014) showed the practical gap between tense theory and computational extraction. I contributed to Section 4.2 by helping design the linguistic feature extraction process — defining how verb tenses and aspectual patterns would be identified from parsed text, and selecting tense distributions, shift rates, and sectional alignment as operationalisations of H2 and H4.

### 7.3 Albin Binu (Ambassador)

I worked on Section 2.1 and Section 4.4. For 2.1, I summarised Pérez-Rosas et al. (2018) on psycholinguistic features for fake news detection and Grieve and Woodfield (2023) on fake news as a register — particularly their finding of higher present tense in fake texts, which directly motivates H2 and H4. In Section 4.4, I designed the coherence analysis: the embedding-based approach using sentence-transformer cosine similarity, and the entity-grid approach tracking entity repetition and grammatical role transitions across sentences, following Singh et al. (2020). I linked these coherence measures to the temporal features and the H3 test.

### 7.4 Mahati Rane (Verifier)

Working on Section 2.2, Becker and Genz (2019) helped me understand how tense and temporal deixis guide readers through narrative time — feeding directly into how we framed hypotheses about tense stability and section-level coherence. Ayman Hamad Rlneil Hamdan (2016) made me appreciate how headlines function almost as their own sub-genre with distinct tense conventions, which is the direct motivation for H4's focus on title-body misalignment. In Section 4.3, I contributed to the temporal expression extraction approach. Although the pipeline ultimately uses a pure-Python regex tagger rather than HeidelTime (due to Java dependency issues), the conceptual framework for inconsistency detection — anchoring relative expressions to publication date and flagging temporal contradictions — follows the HeidelTime-inspired approach I developed from Strötgen and Gertz (2013).

### 7.5 Declaration

All group members contributed to the selection of research papers, discussions, and development of the research question. Each member was responsible for analysing assigned papers and contributing to the methodology, implementation, and report writing.

---

## References

Ahmed, H., I. Traore, and S. Saad (2017). Detection of online fake news using n-gram analysis and machine learning techniques. In *Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments (ISDDC 2017)*, pp. 127–138. Springer.

Ayman Hamad Rlneil Hamdan, L. Q. (2016). Analyzing the use of tenses in english news headlines. *Journal of Humanities*.

Becker, M. and I. Genz (2019). Traveling through narrative time: How tense and temporal deixis guide the representation of time and viewpoint in news narratives. *Cognitive Linguistics 30*(2), 355–388.

Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.

Bögel, T., J. Strötgen, and M. Gertz (2014). Extracting tense clusters from narrative texts. In *Proceedings of LREC*.

Elson, D. K. and K. R. McKeown (2010). Tense and aspect assignment in narrative discourse. In *Proceedings of INLG*.

Fernandes, M. et al. (2025). Analyzing the temporal dynamics of linguistic features contained in misinformation. *arXiv preprint arXiv:2503.04786*.

Fleischman, S. (1990). Toward a theory of tense-aspect in narrative discourse. In S. Fleischman and L. R. Waugh (Eds.), *New Vistas in Grammar*, pp. 173–195. John Benjamins.

Grieve, J. and H. Woodfield (2023). *The Language of Fake News*. Cambridge University Press.

Krejčír, Z. (2021). Tense shift in indirect speech in newspaper reports. *Discourse and Knowledge*.

Murayama, T. et al. (2020). Fake news detection using temporal point processes. In *CySoc@ICWSM*.

Pérez-Rosas, V., B. Kleinberg, A. Lefevre, and R. Mihalcea (2018). Automatic detection of fake news. In *Proceedings of COLING*, pp. 3391–3401.

Pustejovsky, J. et al. (2003). TimeML: Temporal markup language. In *Proceedings of the ARDA Workshop on Temporal and Event Recognition for Question Answering Systems*.

Schlör, J. and J. Ginzburg (2019). Tense use in dialogue. In *Proceedings of SemDial 2019*, pp. 143–150.

Singh, V. K., S. R. Sahoo, A. Mukherjee, and P. Goyal (2020). On the coherence of fake news articles. In *Proceedings of the ECAI 2020 Workshop on Algorithmic Bias in Search and Recommendation (Bias 2020)*.

Strötgen, J. and M. Gertz (2013). Heideltime: High quality rule-based extraction and normalization of temporal expressions. In *Proceedings of SemEval*, pp. 239–248.

Wuyun, S. (2016). The influence of tense interpretation on discourse coherence. *Lingua 178*, 1–25.
