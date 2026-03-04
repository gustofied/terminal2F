"""Quick data quality checks on the email-to-cc-bcc parquet."""

import json
import re
from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).parent / "v1.parquet"


def check(path: Path = PARQUET):
    df = pd.read_parquet(path)
    n = len(df)
    print(f"rows: {n}\n")

    # --- Q2 duplicates Q1 ---
    verbatim = (df["question_1"] == df["question_2"]).sum()
    near = sum(1 for _, r in df.iterrows() if r["question_1"][:100] == r["question_2"][:100])
    print(f"Q2 == Q1 verbatim:       {verbatim} ({verbatim/n*100:.1f}%)")
    print(f"Q2 first 100 chars == Q1: {near} ({near/n*100:.1f}%)\n")

    # --- BCC people mentioned in email text ---
    bcc_mentioned, bcc_total = 0, 0
    non_bcc_mentioned, non_bcc_total = 0, 0
    bcc_addressed = 0

    for _, row in df.iterrows():
        a1 = json.loads(row["answer_1"])
        bcc_set = set(a1.get("bcc", []))
        all_emails = set(a1.get("to", [])) | set(a1.get("cc", [])) | bcc_set

        for line in row["email_list"].split("\n"):
            for email in all_emails:
                if email not in line:
                    continue
                name = line.split("<")[0].strip().lstrip("- ").strip()
                if not name or len(name) <= 3:
                    continue
                first = name.split()[0]
                mentioned = name in row["question_1"]

                if email in bcc_set:
                    bcc_total += 1
                    if mentioned:
                        bcc_mentioned += 1
                    # check direct addressing
                    patterns = [
                        rf"(?:Hi|Hello|Hey|Dear|Thanks|Thank you),?\s+{re.escape(first)}",
                        rf"{re.escape(first)},?\s+(?:can you|could you|please|I need|would you)",
                        rf"(?:cc|loop|looping|copying|include|including)\s+{re.escape(first)}",
                    ]
                    if any(re.search(p, row["question_1"], re.IGNORECASE) for p in patterns):
                        bcc_addressed += 1
                else:
                    non_bcc_total += 1
                    if mentioned:
                        non_bcc_mentioned += 1

    print(f"BCC people full name in Q1:     {bcc_mentioned}/{bcc_total} ({bcc_mentioned/bcc_total*100:.1f}%)")
    print(f"non-BCC people full name in Q1: {non_bcc_mentioned}/{non_bcc_total} ({non_bcc_mentioned/non_bcc_total*100:.1f}%)")
    print(f"BCC people directly addressed:  {bcc_addressed} ({bcc_addressed/n*100:.1f}% of rows)\n")

    # --- field overlap ---
    to_bcc, cc_bcc = 0, 0
    for _, row in df.iterrows():
        for i in range(1, 4):
            a = json.loads(row[f"answer_{i}"])
            if set(a.get("to", [])) & set(a.get("bcc", [])):
                to_bcc += 1
                break
        for i in range(1, 4):
            a = json.loads(row[f"answer_{i}"])
            if set(a.get("cc", [])) & set(a.get("bcc", [])):
                cc_bcc += 1
                break

    print(f"To/BCC overlap (any turn): {to_bcc}")
    print(f"CC/BCC overlap (any turn): {cc_bcc}")


if __name__ == "__main__":
    check()
