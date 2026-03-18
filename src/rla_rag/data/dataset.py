import random
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QASample:
    qid: str
    question: str
    answer: str
    support_doc_ids: Set[str]


def load_toy_dataset() -> Tuple[List[Document], List[QASample]]:
    docs = [
        Document(
            doc_id="d1",
            title="Paris and France",
            text=(
                "Paris is the capital city of France. "
                "France is in Western Europe."
            ),
        ),
        Document(
            doc_id="d2",
            title="Eiffel Tower",
            text=(
                "The Eiffel Tower is a landmark in Paris. "
                "It was designed by Gustave Eiffel."
            ),
        ),
        Document(
            doc_id="d3",
            title="Albert Einstein",
            text=(
                "Albert Einstein developed the theory of relativity. "
                "Einstein was born in Ulm, Germany."
            ),
        ),
        Document(
            doc_id="d4",
            title="Germany",
            text=(
                "Berlin is the capital of Germany. "
                "Germany is a country in Europe."
            ),
        ),
        Document(
            doc_id="d5",
            title="Python Language",
            text=(
                "Python is a programming language created by Guido van Rossum. "
                "It emphasizes readability."
            ),
        ),
        Document(
            doc_id="d6",
            title="Netherlands",
            text=(
                "Amsterdam is the capital of the Netherlands. "
                "Guido van Rossum is Dutch."
            ),
        ),
        Document(
            doc_id="d7",
            title="Relativity and Light",
            text=(
                "The theory of relativity changed modern physics. "
                "It was introduced by Albert Einstein."
            ),
        ),
        Document(
            doc_id="d8",
            title="European Capitals",
            text=(
                "Paris is in France, Berlin is in Germany, "
                "and Amsterdam is in the Netherlands."
            ),
        ),
    ]

    qa_samples = [
        QASample(
            qid="q1",
            question="What is the capital of the country where the Eiffel Tower is located?",
            answer="Paris",
            support_doc_ids={"d2", "d1"},
        ),
        QASample(
            qid="q2",
            question="Which city is the capital of Einstein's birth country?",
            answer="Berlin",
            support_doc_ids={"d3", "d4"},
        ),
        QASample(
            qid="q3",
            question="Who created the Python programming language?",
            answer="Guido van Rossum",
            support_doc_ids={"d5"},
        ),
        QASample(
            qid="q4",
            question="What is the capital of the country of Guido van Rossum?",
            answer="Amsterdam",
            support_doc_ids={"d6"},
        ),
        QASample(
            qid="q5",
            question="Who introduced the theory that changed modern physics?",
            answer="Albert Einstein",
            support_doc_ids={"d7"},
        ),
        QASample(
            qid="q6",
            question="In which country is Berlin located?",
            answer="Germany",
            support_doc_ids={"d4"},
        ),
        QASample(
            qid="q7",
            question="What country has Paris as its capital?",
            answer="France",
            support_doc_ids={"d1"},
        ),
        QASample(
            qid="q8",
            question="Which city is the capital of the Netherlands?",
            answer="Amsterdam",
            support_doc_ids={"d6"},
        ),
    ]
    return docs, qa_samples


def split_qa_samples(
    qa_samples: Sequence[QASample],
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[QASample], List[QASample], List[QASample]]:
    samples = list(qa_samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    train_end = max(1, int(n * train_ratio))
    dev_end = min(n, train_end + max(1, int(n * dev_ratio)))

    train = samples[:train_end]
    dev = samples[train_end:dev_end]
    test = samples[dev_end:]

    if not test:
        test = dev[-1:] if dev else train[-1:]
        if dev:
            dev = dev[:-1]
        elif train:
            train = train[:-1]

    return train, dev, test

