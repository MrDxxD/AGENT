import argparse

from rla_rag.data.loaders import load_project_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    docs, qa_samples = load_project_dataset(
        dataset="hotpot",
        hotpot_path=args.hotpot_path,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )

    non_empty_support = sum(1 for x in qa_samples if x.support_doc_ids)
    print(f"docs={len(docs)}")
    print(f"qa_samples={len(qa_samples)}")
    print(f"samples_with_support={non_empty_support}")
    if qa_samples:
        first = qa_samples[0]
        print(f"sample_qid={first.qid}")
        print(f"sample_question={first.question}")
        print(f"sample_answer={first.answer}")
        print(f"sample_support_docs={len(first.support_doc_ids)}")


if __name__ == "__main__":
    main()
