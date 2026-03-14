from pathlib import Path

import pandas as pd


ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_text_table(path, expected_columns=None, encodings=ENCODING_CANDIDATES):
    """
    Read a small text table with light format detection.

    The THINGS fMRI metadata ships as `.tsv` files that are often comma-delimited,
    and some Windows copies may be encoded as cp1252/latin-1 instead of utf-8.
    """

    path = Path(path)
    last_error = None

    for encoding in encodings:
        try:
            sample = path.read_text(encoding=encoding, errors="strict")[:4096]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        delimiter = _infer_delimiter(sample)
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            on_bad_lines="skip",
            engine="python",
        )
        if expected_columns and not set(expected_columns).intersection(df.columns):
            alternate_delimiter = "\t" if delimiter == "," else ","
            alternate_df = pd.read_csv(
                path,
                sep=alternate_delimiter,
                encoding=encoding,
                on_bad_lines="skip",
                engine="python",
            )
            if set(expected_columns).intersection(alternate_df.columns):
                return alternate_df
        return df

    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "utf-8"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 1),
        f"Unable to decode {path} with supported encodings {encodings}",
    )


def _infer_delimiter(sample):
    first_line = sample.splitlines()[0] if sample else ""
    return "\t" if first_line.count("\t") > first_line.count(",") else ","
