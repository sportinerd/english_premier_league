#!/usr/bin/env python3
"""
inspect_json.py  –  hard-wired to the file you supplied
Run:
    python inspect_json.py
"""

import json
from pathlib import Path
from collections import OrderedDict

# ---------------------------------------------------------
# 1. Hard-coded path – change only if the file moves
# ---------------------------------------------------------
JSON_PATH = Path(r"F:\Arifin bhai\english_premier_league\all_league_data.json")


# ---------------------------------------------------------
# 2. Helper
# ---------------------------------------------------------
def count_fields_and_subentries(obj):
    total_fields = 0
    total_subentries = 0

    def _walk(o):
        nonlocal total_fields, total_subentries
        if isinstance(o, dict):
            total_subentries += 1
            total_fields += len(o)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for item in o:
                _walk(item)

    _walk(obj)
    return total_fields, total_subentries


# ---------------------------------------------------------
# 3. Main logic
# ---------------------------------------------------------
def main():
    if not JSON_PATH.exists():
        exit(f"File not found: {JSON_PATH}")
    if JSON_PATH.stat().st_size == 0:
        exit("JSON file is empty (0 bytes).")

    try:
        with JSON_PATH.open(encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as e:
        exit(f"Invalid JSON: {e}")

    # Top-level entries
    if isinstance(data, list):
        top_entries = data
    elif isinstance(data, dict):
        top_entries = list(data.values())
    else:
        exit("Root element must be a JSON array or object.")

    # Build textual report
    report_lines = [
        f"File: {JSON_PATH.name}",
        f"Total top-level entries: {len(top_entries)}",
        "-" * 50
    ]

    for idx, entry in enumerate(top_entries, 1):
        fields, subs = count_fields_and_subentries(entry)
        report_lines.append(
            f"Entry #{idx:<4} | fields+subfields: {fields:<6} | sub-entries: {subs}"
        )

    report_path = JSON_PATH.with_name("structure_report.txt")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Structure report → {report_path}")

    # Save first three entries
    first_three = top_entries[:3]
    first_path = JSON_PATH.with_name("first_three_entries.json")
    with first_path.open("w", encoding="utf-8") as f:
        json.dump(first_three, f, indent=2, ensure_ascii=False)
    print(f"First three entries → {first_path}")


if __name__ == "__main__":
    main()