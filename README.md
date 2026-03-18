# RLA-RAG: Agent + RL + RAG (MVP)

This repository is a from-scratch starter for a master's-level final project:
- Agentic RAG pipeline
- RL controller (PPO) for dynamic action selection
- Retrieval, query rewriting, evidence summarization, and answer generation

## 1) What this MVP includes

- Toy multi-hop style dataset (`8` QA items + supporting docs)
- Hybrid retrieval:
  - Sparse retriever (word TF-IDF)
  - Dense-like retriever (char n-gram TF-IDF as a lightweight semantic proxy)
- Agent action space:
  - `retrieve_sparse`
  - `retrieve_dense`
  - `rewrite_query`
  - `summarize_evidence`
  - `answer_now`
- Gymnasium environment with dense reward shaping
- PPO training script and baseline evaluation script

## 2) Project structure

```text
.
├── requirements.txt
├── scripts
│   ├── train.py
│   ├── evaluate.py
│   └── play.py
└── src
    └── rla_rag
        ├── agent
        ├── data
        ├── env
        ├── eval
        ├── retrieval
        └── pipeline.py
```

## 3) Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=src
python scripts/train.py --timesteps 30000
python scripts/evaluate.py --model models/ppo_rla_rag.zip --episodes 300
python scripts/play.py --model models/ppo_rla_rag.zip
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python .\scripts\train.py --timesteps 30000
python .\scripts\evaluate.py --model .\models\ppo_rla_rag.zip --episodes 300
python .\scripts\play.py --model .\models\ppo_rla_rag.zip
```

## 4) Metrics

- Answer quality: `accuracy` (exact match on normalized text)
- Retrieval quality: `support_coverage` (fraction of gold support docs retrieved)
- Efficiency: `avg_steps`, `avg_token_cost`
- Stability: variance across random episodes

## 5) Next steps for final project

- Replace toy data with HotpotQA / 2WikiMultihopQA
- Replace dense proxy retriever with embedding model + FAISS
- Add reranker model and citation faithfulness metrics
- Add ablations and significance testing

