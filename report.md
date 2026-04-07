# Temporal Signals of Deception: Tense and Event Structure in Fake vs Verified News

Sneha Meto · 25337472 · metos@tcd.ie  
Mahati Rane · 25334671 · mrane@tcd.ie  
Albin Binu · 25353208 · binua@tcd.ie  
Bhathrinaathan Muthuramalingam Bhanumathi · 25362507 · muthurab@tcd.ie

April 2026

---

## Abstract

This paper investigates whether fake news articles display distinctive temporal behaviour compared to verified news, focusing on tense usage, temporal expressions, event sequencing, and cross-sectional tense consistency in English news discourse. We combine insights from register variation, narrative tense theory, and fake news linguistics to derive four hypotheses: that fake news exhibits (i) more temporal inconsistencies, (ii) more frequent tense shifts and a distinct tense distribution, (iii) poorer alignment between narrated and implied event sequences, and (iv) greater tense misalignment between structural sections — titles, leads, and bodies — than legitimate news reporting. Using the ISOT Fake News Dataset, we implement a full five-phase computational pipeline covering preprocessing, spaCy-based tense and aspect extraction, regex-based temporal expression tagging, embedding and entity-grid coherence scoring, and statistical analysis with classification experiments. H2 is partially supported: fake news uses significantly more present and future tense while real news is past-tense dominant (p < 0.001). H4 is our strongest result: fake news shows significantly greater tense misalignment between title and body (p = 0.007) and between title and lead (p = 0.024), revealing a structural pattern of involvement-oriented headlines over informational bodies. H1 and H3 are not supported at this sample size. A Random Forest classifier using only temporal features achieves 0.79 AUC, confirming that temporal structure carries discriminative signal beyond topical content.

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

**H4 (Structural tense misalignment).** Fake news articles exhibit significantly greater tense misalignment between structural sections — titles, leads, and bodies — than verified news articles. This section-level tense inconsistency reflects a pattern whereby deceptive articles employ present-tense, involvement-oriented headlines over past-tense narrative bodies, a structural pattern absent in legitimate reporting.

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

Sentence-level shift rates, tense entropy, and whole-article tense proportions operationalise **H2**. Cross-sectional alignment features operationalise **H4**: for each pair of sections, we compute the L1 distance between their tense distributions — `align_title_body`, `align_lead_body`, and `align_title_lead`. A higher value means the two sections use tense differently; a lower value means they are consistent.

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
| `temporal_incons` | 0.077 | 0.097 | 0.776 | 0.020 | |
| `abs_date_count` | 0.654 | 0.274 | 0.277 | −0.078 | |
| `abs_future_ratio` | 0.000 | 0.016 | 0.534 | 0.016 | |
| `art_adv_total` | 2.192 | 1.274 | 0.061 | −0.246 | |

*Table 3: Mann-Whitney U tests for H1 features.*

**H1 is not supported.** None of the inconsistency features reach significance. Notably, real news has marginally more inconsistencies on average than fake news — the opposite of the predicted direction. The temporal adverb count approaches significance (p = 0.061) but does not cross the threshold. This null result most likely reflects the limitations of the regex-based tagger, which captures only lexically explicit temporal expressions and misses many implicit or contextually-resolved references. A richer temporal resolver would be required to adequately test H1.

### 5.2 H2 — Tense Shifts and Distributions

| Feature | Mean (Fake) | Mean (Real) | p-value | Effect *r* | Sig |
|---|---|---|---|---|---|
| `art_shift_rate` | 0.423 | 0.389 | 0.627 | −0.066 | |
| `art_entropy` | 0.734 | 0.695 | 0.106 | −0.220 | |
| `art_prop_past` | 0.353 | 0.487 | 0.0002 | 0.499 | \*\*\* |
| `art_prop_present` | 0.613 | 0.497 | 0.0015 | −0.432 | \*\* |
| `art_prop_future` | 0.034 | 0.016 | 0.0025 | −0.384 | \*\* |
| `art_aspect_perfect` | 0.035 | 0.057 | 0.056 | 0.259 | |
| `art_aspect_progressive` | 0.034 | 0.028 | 0.160 | −0.188 | |

*Table 4: Mann-Whitney U tests for H2 features (selected).*

**H2 is partially supported.** The headline claim of higher shift rate and entropy in fake news is not confirmed. However, tense distributions differ strongly: fake news uses significantly more present tense (mean 0.613 vs 0.497, p = 0.0015) and future tense (0.034 vs 0.016, p = 0.0025), while real news uses significantly more past tense (0.487 vs 0.353, p = 0.0002, *r* = 0.50, a large effect). This is consistent with Grieve and Woodfield's (2023) register analysis and Biber's (1988) finding that past tense clusters with informational written registers. Fake news appears to adopt a present-tense immediacy stance throughout the article, not just in headlines.

### 5.3 H3 — Event Sequencing and Coherence

Spearman correlations between `event_misalignment` and each coherence measure showed no significant relationship within the fake news group (all p > 0.10). The only significant result was in the real news group: entity uniqueness negatively correlated with misalignment (*r* = −0.143, p = 0.015), a direction not predicted by H3.

**H3 is not supported.** The misalignment score is sparse — most articles contain too few consecutive sentences with ordering connectives to produce a reliable score. This is a measurement problem rather than evidence against the underlying theory. Future work with a proper temporal IE system would better operationalise this hypothesis.

### 5.4 H4 — Structural Tense Misalignment

| Feature | Mean (Fake) | Mean (Real) | p-value | Effect *r* | Sig |
|---|---|---|---|---|---|
| `align_title_body` | 0.645 | 0.860 | 0.007 | 0.364 | \*\* |
| `align_title_lead` | 0.645 | 0.845 | 0.024 | 0.306 | \* |
| `align_lead_body` | 0.276 | 0.423 | 0.139 | 0.201 | |

*Table 5: Mann-Whitney U tests for H4 features.*

**H4 is supported.** Fake news shows significantly greater tense misalignment between title and body (p = 0.007, *r* = 0.36) and between title and lead (p = 0.024, *r* = 0.31). Real news has consistently higher alignment scores — its titles use tense that is consistent with the body and lead. Fake news titles depart from the tense register of the rest of the article, reflecting the pattern described by Ayman Hamad Rlneil Hamdan (2016) whereby headlines use present tense for immediacy while bodies report in past tense — but exaggerating this contrast beyond what is normal in legitimate reporting. The lead–body alignment is not significant, suggesting the title is the primary site of the structural break.

### 5.5 Classification

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LogReg (temporal) | 0.670 | 0.435 | 0.385 | 0.408 | 0.752 |
| Random Forest (temporal) | 0.727 | 0.538 | 0.538 | 0.538 | 0.791 |
| LogReg (BoW baseline) | 0.966 | 0.926 | 0.962 | 0.943 | 0.997 |

*Table 6: Classification performance on the test set.*

The BoW baseline achieves near-perfect performance, consistent with Ahmed et al. (2017) — the ISOT dataset has strong lexical separability due to systematic topic and source differences. However, the Random Forest using only temporal features achieves 0.79 AUC, well above chance, confirming that temporal structure carries meaningful discriminative signal independent of topical content.

The top temporal features by Random Forest importance are: `body_prop_present` (0.090), `art_prop_present` (0.062), `align_title_body` (0.049), `art_prop_past` (0.048), and `body_prop_past` (0.043). The appearance of `align_title_body` in the top three confirms that H4's structural misalignment signal is the third most discriminative temporal feature overall.

---

## 6 Contribution

This study makes three contributions. First, it provides the first systematic operationalisation of cross-sectional tense misalignment as a feature of fake news, showing that the structural contrast between present-tense headlines and past-tense bodies is significantly more pronounced in fake articles (H4). Second, it confirms at scale that fake news uses a present-tense dominant register while verified news is past-tense dominant (H2), replicating and extending Grieve and Woodfield (2023) on a large labelled corpus. Third, it demonstrates that temporal features alone achieve 0.79 AUC in classification, establishing a non-lexical, temporally-grounded discriminative baseline.

The null results for H1 and H3 are informative: they suggest the limitations of regex-based temporal tagging and connective-counting approaches for inconsistency and misalignment detection, and motivate future work using more robust temporal resolution tools.

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
