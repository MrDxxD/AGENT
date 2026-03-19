# RLA-RAG: Agent + RL + RAG (MVP)

This repository is a from-scratch starter for a master's-level final project:
- Agentic RAG pipeline
- RL controller (PPO) for dynamic action selection
- Retrieval, query rewriting, evidence summarization, and answer generation

## 1) What this project includes now

- Toy multi-hop dataset (`8` QA items + supporting docs)
- HotpotQA JSON loader (`supporting_facts` + `context` parsing)
- Two split modes:
  - `random_split`: single file + random train/dev/test split
  - `official_split`: explicit `--train-path` and `--dev-path`
- Hybrid retrieval:
  - Sparse retriever (word TF-IDF)
  - Dense-like retriever (char n-gram TF-IDF proxy)
- Agent action space:
  - `retrieve_sparse`
  - `retrieve_dense`
  - `rewrite_query`
  - `summarize_evidence`
  - `answer_now`
- PPO training, baseline evaluation, and playable rollout
- Ablation runner with multi-seed CSV export

## 2) Project structure

```text
.
|- requirements.txt
|- scripts
|  |- train.py
|  |- evaluate.py
|  |- play.py
|  |- prepare_hotpot.py
|  `- run_ablations.py
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

## 4) HotpotQA (official split)

Expected file format: official HotpotQA JSON (list of samples with `_id`, `question`, `answer`, `supporting_facts`, `context`).

```powershell
$env:PYTHONPATH="src"
python .\scripts\prepare_hotpot.py --hotpot-path D:\data\hotpot_train_v1.1.json --max-samples 1000
python .\scripts\prepare_hotpot.py --hotpot-path D:\data\hotpot_dev_distractor_v1.json --max-samples 500

python .\scripts\train.py --dataset hotpot --train-path D:\data\hotpot_train_v1.1.json --dev-path D:\data\hotpot_dev_distractor_v1.json --max-train-samples 5000 --max-dev-samples 1000 --timesteps 80000 --model-out .\models\ppo_hotpot.zip

python .\scripts\evaluate.py --dataset hotpot --train-path D:\data\hotpot_train_v1.1.json --dev-path D:\data\hotpot_dev_distractor_v1.json --model .\models\ppo_hotpot.zip --split dev --episodes 300
```

## 5) Ablations and CSV export

Available ablation flags in `train/evaluate/play`:
- `--disable-dense`
- `--disable-rewrite`
- `--disable-summary`
- `--disable-coverage-reward`
- `--disable-token-penalty`

Run multi-seed ablations:

```powershell
$env:PYTHONPATH="src"
python .\scripts\run_ablations.py --dataset toy --timesteps 4000 --eval-episodes 120 --seeds 42,43 --out-csv .\results\ablations_runs.csv --out-summary-csv .\results\ablations_summary.csv
```

## 6) Core metrics

- Answer quality: `accuracy` (normalized exact match)
- Retrieval quality: `avg_support_coverage`
- Efficiency: `avg_steps`, `avg_token_cost`
- Utility objective: `avg_reward`

## 7) Next steps for final project

- Replace rule-based reasoner with LLM-based answerer + faithfulness checks
- Add FAISS embedding retriever and cross-encoder reranker
- Add final ablation sets and significance test report
- Run full experiments on official train/dev splits with fixed seeds
