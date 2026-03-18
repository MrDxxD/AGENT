# RLA-RAG: Agent + RL + RAG (MVP)

This repository is a from-scratch starter for a master's-level final project:
- Agentic RAG pipeline
- RL controller (PPO) for dynamic action selection
- Retrieval, query rewriting, evidence summarization, and answer generation

## 1) What this project includes now

- Toy multi-hop dataset (`8` QA items + docs)
- HotpotQA json loader (`supporting_facts` + `context` parsing)
- Train/dev/test split to avoid train-eval contamination
- Hybrid retrieval:
  - Sparse retriever (word TF-IDF)
  - Dense-like retriever (char n-gram TF-IDF proxy)
- Agent action space:
  - `retrieve_sparse`
  - `retrieve_dense`
  - `rewrite_query`
  - `summarize_evidence`
  - `answer_now`
- Gymnasium environment with reward shaping
- PPO training script and baseline evaluation script

## 2) Project structure

```text
.
|- requirements.txt
|- scripts
|  |- train.py
|  |- evaluate.py
|  |- play.py
|  `- prepare_hotpot.py
|- tests
`- src
   `- rla_rag
```

## 3) Quickstart (toy)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python .\scripts\train.py --dataset toy --timesteps 30000 --model-out .\models\ppo_rla_rag.zip
python .\scripts\evaluate.py --dataset toy --model .\models\ppo_rla_rag.zip --episodes 300 --split test
python .\scripts\play.py --dataset toy --model .\models\ppo_rla_rag.zip --split test
pytest
```

## 4) HotpotQA usage

Expected file format: official HotpotQA json (list of samples, each with `_id`, `question`, `answer`, `supporting_facts`, `context`).

```powershell
$env:PYTHONPATH="src"
python .\scripts\prepare_hotpot.py --hotpot-path D:\data\hotpot_dev_distractor_v1.json --max-samples 2000
python .\scripts\train.py --dataset hotpot --hotpot-path D:\data\hotpot_dev_distractor_v1.json --max-samples 2000 --timesteps 50000 --model-out .\models\ppo_hotpot.zip
python .\scripts\evaluate.py --dataset hotpot --hotpot-path D:\data\hotpot_dev_distractor_v1.json --max-samples 2000 --model .\models\ppo_hotpot.zip --split test --episodes 300
```

## 5) Core metrics

- Answer quality: `accuracy` (normalized exact match)
- Retrieval quality: `avg_support_coverage`
- Efficiency: `avg_steps`, `avg_token_cost`
- Utility objective: `avg_reward`

## 6) Next steps for final project

- Replace rule-based reasoner with LLM-based answerer + faithfulness checks
- Add FAISS embedding retriever and cross-encoder reranker
- Add ablation runner and CSV result export
- Run final experiments on official train/dev splits with fixed seeds
