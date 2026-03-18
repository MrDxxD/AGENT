import random
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from rla_rag.agent.query_rewriter import rewrite_query
from rla_rag.agent.reasoner import normalize_text
from rla_rag.data.dataset import Document, QASample
from rla_rag.pipeline import Pipeline


class Action(IntEnum):
    RETRIEVE_SPARSE = 0
    RETRIEVE_DENSE = 1
    REWRITE_QUERY = 2
    SUMMARIZE_EVIDENCE = 3
    ANSWER_NOW = 4


class RlaRagEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        docs: List[Document],
        qa_samples: List[QASample],
        pipeline: Pipeline,
        max_steps: int = 6,
        max_token_budget: int = 400,
        seed: int = 42,
    ):
        super().__init__()
        self.docs = docs
        self.qa_samples = qa_samples
        self.pipeline = pipeline
        self.max_steps = max_steps
        self.max_token_budget = max_token_budget
        self.rng = random.Random(seed)

        self.action_space = Discrete(len(Action))
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        self.current_sample: Optional[QASample] = None
        self.current_query = ""
        self.evidence_doc_ids: Set[str] = set()
        self.evidence_chunks: List[str] = []
        self.summary = ""
        self.step_count = 0
        self.token_cost = 0
        self.last_action = -1
        self.best_sparse_score = 0.0
        self.best_dense_score = 0.0
        self.last_confidence = 0.0
        self.support_coverage = 0.0

    def peek_random_sample(self) -> QASample:
        return self.rng.choice(self.qa_samples)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None, sample_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng.seed(seed)
        if sample_id is not None:
            matches = [q for q in self.qa_samples if q.qid == sample_id]
            self.current_sample = matches[0] if matches else self.rng.choice(self.qa_samples)
        else:
            self.current_sample = self.rng.choice(self.qa_samples)

        self.current_query = self.current_sample.question
        self.evidence_doc_ids = set()
        self.evidence_chunks = []
        self.summary = ""
        self.step_count = 0
        self.token_cost = 0
        self.last_action = -1
        self.best_sparse_score = 0.0
        self.best_dense_score = 0.0
        self.last_confidence = 0.0
        self.support_coverage = 0.0
        return self._obs(), {"qid": self.current_sample.qid}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((14,), dtype=np.float32)
        obs[0] = min(1.0, self.step_count / max(1, self.max_steps))
        obs[1] = min(1.0, len(self.evidence_doc_ids) / max(1, len(self.docs)))
        obs[2] = float(self.support_coverage)
        obs[3] = float(min(1.0, self.best_sparse_score))
        obs[4] = float(min(1.0, self.best_dense_score))
        obs[5] = min(1.0, len(self.current_query.split()) / 30.0)
        obs[6] = 1.0 if self.summary else 0.0
        obs[7] = float(self.last_confidence)
        obs[8] = min(1.0, self.token_cost / max(1, self.max_token_budget))
        if self.last_action >= 0:
            obs[9 + self.last_action] = 1.0
        return obs

    def _estimate_token_cost(self, text: str) -> int:
        return max(1, len(text.split()))

    def _update_coverage(self):
        if not self.current_sample:
            self.support_coverage = 0.0
            return
        support = self.current_sample.support_doc_ids
        self.support_coverage = len(self.evidence_doc_ids & support) / max(1, len(support))

    def _add_docs(self, doc_ids: List[str]) -> int:
        before = len(self.evidence_doc_ids)
        for doc_id in doc_ids:
            if doc_id not in self.evidence_doc_ids:
                self.evidence_doc_ids.add(doc_id)
                self.evidence_chunks.append(self.pipeline.get_doc_text(doc_id))
        return len(self.evidence_doc_ids) - before

    def _answer(self):
        answer, confidence = self.pipeline.reasoner.answer(
            self.current_sample.question,
            self.evidence_chunks,
            self.summary,
        )
        self.last_confidence = confidence
        return answer, confidence

    def _finalize_info(
        self,
        predicted: str,
        reward: float,
        terminated_by_answer: bool,
        terminated: bool,
        truncated: bool,
    ) -> Dict:
        gold = self.current_sample.answer
        is_correct = normalize_text(predicted) == normalize_text(gold)
        return {
            "predicted_answer": predicted,
            "gold_answer": gold,
            "is_correct": is_correct,
            "support_coverage": self.support_coverage,
            "steps": self.step_count,
            "token_cost": self.token_cost,
            "done_by_answer": terminated_by_answer,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "qid": self.current_sample.qid,
        }

    def step(self, action: int):
        assert self.current_sample is not None, "Environment not reset."

        action_id = int(action)
        if action_id < 0 or action_id >= len(Action):
            self.step_count += 1
            reward = -1.0
            info = self._finalize_info(
                predicted="INVALID_ACTION",
                reward=reward,
                terminated_by_answer=False,
                terminated=False,
                truncated=True,
            )
            return self._obs(), float(reward), False, True, info

        self.step_count += 1
        self.last_action = action_id
        reward = -0.01
        terminated = False
        truncated = False
        terminated_by_answer = False
        predicted_answer = "UNKNOWN"
        previous_coverage = self.support_coverage

        act = Action(action_id)
        if act == Action.RETRIEVE_SPARSE:
            results = self.pipeline.search_sparse(self.current_sample.question, self.current_query, k=3)
            new_docs = self._add_docs([r.doc_id for r in results])
            if results:
                self.best_sparse_score = max(self.best_sparse_score, results[0].score)
            reward += 0.08 * new_docs if new_docs > 0 else -0.03
            self.token_cost += self._estimate_token_cost(self.current_query) + 20

        elif act == Action.RETRIEVE_DENSE:
            results = self.pipeline.search_dense(self.current_sample.question, self.current_query, k=3)
            new_docs = self._add_docs([r.doc_id for r in results])
            if results:
                self.best_dense_score = max(self.best_dense_score, results[0].score)
            reward += 0.08 * new_docs if new_docs > 0 else -0.03
            self.token_cost += self._estimate_token_cost(self.current_query) + 20

        elif act == Action.REWRITE_QUERY:
            old_query = self.current_query
            self.current_query = rewrite_query(
                self.current_sample.question,
                self.current_query,
                self.evidence_chunks,
            )
            changed = self.current_query != old_query
            if not self.evidence_doc_ids:
                reward -= 0.08
            else:
                reward += 0.01 if changed else -0.02
            self.token_cost += self._estimate_token_cost(self.current_query)

        elif act == Action.SUMMARIZE_EVIDENCE:
            old_summary = self.summary
            self.summary = self.pipeline.summarize(self.evidence_chunks, max_sentences=2)
            changed = self.summary != old_summary and bool(self.summary)
            if not self.evidence_doc_ids:
                reward -= 0.08
            else:
                reward += 0.01 if changed else -0.02
            self.token_cost += 10

        elif act == Action.ANSWER_NOW:
            predicted_answer, confidence = self._answer()
            is_correct = normalize_text(predicted_answer) == normalize_text(self.current_sample.answer)
            if not self.evidence_doc_ids:
                reward -= 0.60
            reward += 1.20 if is_correct else -0.90
            reward += 0.05 * confidence
            terminated = True
            terminated_by_answer = True

        self._update_coverage()
        coverage_gain = self.support_coverage - previous_coverage
        reward += 0.70 * coverage_gain
        reward -= 0.002 * max(0, self.token_cost - self.max_token_budget)

        if not terminated and self.step_count >= self.max_steps:
            predicted_answer, confidence = self._answer()
            is_correct = normalize_text(predicted_answer) == normalize_text(self.current_sample.answer)
            reward += 1.0 if is_correct else -0.80
            reward += 0.05 * confidence
            truncated = True

        info = self._finalize_info(
            predicted=predicted_answer,
            reward=reward,
            terminated_by_answer=terminated_by_answer,
            terminated=terminated,
            truncated=truncated,
        )
        return self._obs(), float(reward), terminated, truncated, info
