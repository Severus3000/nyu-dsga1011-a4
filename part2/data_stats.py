"""Q4 token statistics for the paired train and development data."""

from pathlib import Path
from statistics import mean

from transformers import T5TokenizerFast


DATA_DIR = Path(__file__).resolve().parent / "data"


def token_statistics(tokenizer, lines):
    token_ids = tokenizer(lines, add_special_tokens=False)["input_ids"]
    vocabulary = {token_id for sequence in token_ids for token_id in sequence}
    return mean(len(sequence) for sequence in token_ids), len(vocabulary)


def main():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small", local_files_only=True)

    for split in ("train", "dev"):
        natural_language = (DATA_DIR / f"{split}.nl").read_text(encoding="utf-8").splitlines()
        sql_queries = (DATA_DIR / f"{split}.sql").read_text(encoding="utf-8").splitlines()
        if len(natural_language) != len(sql_queries):
            raise ValueError(f"{split} contains unmatched natural language and SQL rows")
        if not natural_language or any(not line.strip() for line in natural_language + sql_queries):
            raise ValueError(f"{split} contains missing examples")

        variants = (
            ("raw", natural_language, sql_queries),
            ("stripped", [line.strip() for line in natural_language], [line.strip() for line in sql_queries]),
        )
        for stage, questions, queries in variants:
            nl_mean, nl_vocab = token_statistics(tokenizer, questions)
            sql_mean, sql_vocab = token_statistics(tokenizer, queries)
            print(f"{split} {stage}: examples={len(questions)}")
            print(f"  NL: mean length={nl_mean:.4f}, vocabulary={nl_vocab}")
            print(f"  SQL: mean length={sql_mean:.4f}, vocabulary={sql_vocab}")


if __name__ == "__main__":
    main()
