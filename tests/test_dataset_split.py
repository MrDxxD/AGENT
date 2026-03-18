from rla_rag.data.dataset import load_toy_dataset, split_qa_samples


def test_split_non_overlap_and_non_empty_test():
    _, qa = load_toy_dataset()
    train, dev, test = split_qa_samples(qa, seed=42)

    train_ids = {x.qid for x in train}
    dev_ids = {x.qid for x in dev}
    test_ids = {x.qid for x in test}

    assert train_ids
    assert test_ids
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)
    assert len(train_ids | dev_ids | test_ids) == len(qa)
