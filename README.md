# Haystack & Elasticsearch Q&A Search

Scrapes Q&A data from the web and builds two search pipelines on top of it — one keyword-based (Elasticsearch) and one semantic (Haystack with embeddings).

Built as a university project in collaboration with [@voltmaister](https://github.com/voltmaister).

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Haystack](https://img.shields.io/badge/Haystack-FAQ_Pipeline-00A1E0)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-7.x-005571?logo=elasticsearch&logoColor=white)
[![CI](https://github.com/NickLitharis/haystack-elastic/actions/workflows/ci.yml/badge.svg)](https://github.com/NickLitharis/haystack-elastic/actions/workflows/ci.yml)

---

## What It Does

1. **Scrapes** a Q&A website and stores question/answer pairs as JSON
2. **Indexes** the data into Elasticsearch
3. **Searches** using two approaches side by side:

| Notebook | Approach | How It Works |
|---|---|---|
| `Elastic_Search_Index.ipynb` | Keyword search | Elasticsearch BM25 query against the `q&a` index |
| `Haystack_search.ipynb` | Semantic search | `FAQPipeline` + `EmbeddingRetriever` — finds answers by embedding similarity, not keyword overlap |

The semantic pipeline returns the top-5 most relevant answers for any user query using vector embeddings stored in `ElasticsearchDocumentStore`.

---

## Setup

### Prerequisites
- Elasticsearch 7.x running on `localhost:9200`
- Python 3.8+

### Install dependencies

```bash
uv sync
```

### Run

1. Open `Elastic_Search_Index.ipynb` — scrapes and indexes the Q&A data into Elasticsearch
2. Open `Haystack_search.ipynb` — runs the semantic FAQ pipeline against the indexed data

Both notebooks are designed to run on Google Colab.

---

## Pipeline Architecture

```
Web Scraping → JSON → Elasticsearch Index
                              ↓
              EmbeddingRetriever (semantic similarity)
                              ↓
                     FAQPipeline → Top-K Answers
```

---

## Evaluation

"Which pipeline is better?" needs a measurement rather than an impression. [`metrics.py`](metrics.py)
scores a retrieval run against a hand-labelled gold set:

| Metric | Question it answers |
| --- | --- |
| `recall@k` | Did the top *k* contain the answer at all? |
| `precision@k` | How much of the top *k* was relevant? |
| `MRR` | How high did the first correct answer rank? |

```python
from metrics import evaluate

# retrieved document ids, best first — one entry per query
run = {"what is X?": ["doc_12", "doc_3", "doc_7"]}

# the document ids that actually answer each query
labels = {"what is X?": {"doc_7"}}

print(evaluate(run, labels))
# {'mrr': 0.333…, 'recall@1': 0.0, 'recall@3': 1.0, 'precision@3': 0.333…, …}
```

The metrics are unit-tested under [`tests/`](tests/) and run on every push, so the scoring
itself doesn't have to be taken on faith.

### Results

| Pipeline | MRR | recall@1 | recall@3 | recall@5 | precision@5 |
| --- | --- | --- | --- | --- | --- |
| Elasticsearch (BM25 keyword) | — | — | — | — | — |
| Haystack (semantic embeddings) | — | — | — | — | — |

**Not yet measured.** The corpus is scraped at runtime and no gold set is committed, so
producing these numbers needs a local run:

```bash
uv sync
# 1. Elasticsearch 7.x on localhost:9200
# 2. run both notebooks to populate the index
# 3. label ~20 queries with their answering document ids
# 4. metrics.evaluate(run, labels) per pipeline, then fill the table above
```
