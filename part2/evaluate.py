from argparse import ArgumentParser
import json
from pathlib import Path

from utils import compute_metrics, load_queries_and_records


def summarize_errors(questions, gold_queries, gold_records, predictions, records, errors):
    """Group examples by the observed execution outcome."""
    categories = {
        name: {"count": 0, "example": None}
        for name in ("execution_error", "missing_records", "extra_records", "different_records")
    }
    for index, (gold, predicted, error) in enumerate(zip(gold_records, records, errors)):
        gold_set, predicted_set = set(gold), set(predicted)
        if error:
            category = "execution_error"
        elif predicted_set == gold_set:
            continue
        elif predicted_set < gold_set:
            category = "missing_records"
        elif predicted_set > gold_set:
            category = "extra_records"
        else:
            category = "different_records"

        summary = categories[category]
        summary["count"] += 1
        if summary["example"] is None:
            summary["example"] = {
                "row": index + 1,
                "question": questions[index],
                "gold_sql": gold_queries[index],
                "predicted_sql": predictions[index],
                "execution_error": error or None,
                "missing_record_count": len(gold_set - predicted_set) if not error else None,
                "extra_record_count": len(predicted_set - gold_set) if not error else None,
            }
    return categories


def main():
    parser = ArgumentParser()
    parser.add_argument("-ps", "--predicted_sql", required=True)
    parser.add_argument("-pr", "--predicted_records", required=True)
    parser.add_argument("-ds", "--development_sql", required=True)
    parser.add_argument("-dr", "--development_records", required=True)
    parser.add_argument("--development_questions", type=Path,
                        help="Include error counts and examples as JSON using the paired .nl file")
    args = parser.parse_args()

    sql_em, record_em, record_f1, errors = compute_metrics(
        args.development_sql, args.predicted_sql, args.development_records, args.predicted_records
    )
    if args.development_questions is None:
        print("Record F1: ", record_f1)
        return

    questions = args.development_questions.read_text(encoding="utf-8").splitlines()
    gold_queries, gold_records, _ = load_queries_and_records(args.development_sql, args.development_records)
    predictions, records, _ = load_queries_and_records(args.predicted_sql, args.predicted_records)
    if len(questions) != len(gold_queries):
        parser.error("Expected one question per development SQL query")

    report = {
        "total_queries": len(questions),
        "sql_exact_match": sql_em,
        "record_exact_match": record_em,
        "record_f1": float(record_f1),
        "execution_error_rate": sum(map(bool, errors)) / len(questions),
        "errors": summarize_errors(questions, gold_queries, gold_records, predictions, records, errors),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
