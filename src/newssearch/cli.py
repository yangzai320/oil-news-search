"""Build an index from headline CSV/TSV files.

newssearch build --out index data/*.csv
"""

import argparse
import csv
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from .index import build_index, save_index

TIME_COLUMNS = ("news_time", "published_utc", "timestamp")
TITLE_COLUMNS = ("title", "title_clean")


def _parse_ts(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value.isdigit() and len(value) == 14:  # GDELT: YYYYMMDDHHMMSS
        return int(datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp())
    try:
        parsed = datetime.fromisoformat(value.replace(" UTC", "+00:00").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def read_records(paths: list[Path]) -> Iterator[tuple[int, str]]:
    csv.field_size_limit(sys.maxsize)
    for path in paths:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            time_col = next((c for c in TIME_COLUMNS if c in reader.fieldnames), None)
            title_col = next((c for c in TITLE_COLUMNS if c in reader.fieldnames), None)
            if time_col is None or title_col is None:
                raise SystemExit(f"{path}: need one of {TIME_COLUMNS} and one of {TITLE_COLUMNS}")
            for row in reader:
                ts = _parse_ts(row[time_col] or "")
                title = row[title_col] or ""
                if ts is not None and len(title.strip()) >= 10:
                    yield ts, title


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="newssearch")
    sub = parser.add_subparsers(required=True)
    build = sub.add_parser("build", help="build an index directory from headline files")
    build.add_argument("files", nargs="+", type=Path)
    build.add_argument("--out", default="index", type=Path)
    args = parser.parse_args(argv)

    index = build_index(read_records(args.files), source=", ".join(p.name for p in args.files[:3]))
    save_index(index, args.out)
    m = index.meta
    print(
        f"{m['records_seen']:,} records -> {m['num_docs']:,} unique headlines, "
        f"{m['vocab_size']:,} terms, {m['num_postings']:,} postings in {m['build_seconds']}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
