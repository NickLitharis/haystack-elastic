# Haystack & Elasticsearch Q&A Search

Scrapes Q&A data from the web and builds two search pipelines on top of it — one keyword-based (Elasticsearch) and one semantic (Haystack with embeddings).

Built as a university project in collaboration with [@voltmaister](https://github.com/voltmaister).

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Haystack](https://img.shields.io/badge/Haystack-FAQ_Pipeline-00A1E0)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-7.x-005571?logo=elasticsearch&logoColor=white)

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
pip install farm-haystack elasticsearch==7.*
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
