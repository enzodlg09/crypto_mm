from __future__ import annotations
import glob
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from crypto_mm.data.sinks import Schema, RotatingCsvSink, RotatingParquetSink, daily_glob


def _fake_now_sequence(seq: List[float]):
    it = iter(seq)
    last = [None]

    def _now():
        try:
            last[0] = next(it)
            return last[0]
        except StopIteration:
            assert last[0] is not None
            return last[0]
    return _now


def test_csv_rotation_and_dedup(tmp_path: Path):
    schema = Schema(columns=["ts", "a", "b"], dtypes={"ts": "string", "a": "float64", "b": "Int64"})

    day1 = 1_700_000_000.0  # some UTC epoch
    day2 = day1 + 86400.0 + 1.0
    sink = RotatingCsvSink(tmp_path, "quotes", schema, fsync=True, now_fn=_fake_now_sequence([day1, day1, day1, day2, day2]))

    # ecrit 3 lignes (dont 1 doublon ts) sur day1
    sink.write_rows([
        {"ts": "2025-08-21T10:00:00Z", "a": 1.0, "b": 1},
        {"ts": "2025-08-21T10:00:01Z", "a": 2.0, "b": 2},
        {"ts": "2025-08-21T10:00:01Z", "a": 999.0, "b": 999},  # doublon -> ignoré
    ])
    # day2
    sink.write_rows([
        {"ts": "2025-08-22T10:00:00Z", "a": 3.0, "b": 3},
    ])

    # day1: concat parts
    g1 = daily_glob(tmp_path, "quotes", day="2023-11-14")  # not used
    # utilise la date dérivée de day1
    d1 = pd.concat(pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(tmp_path / "202*_*quotes_*.csv"))))
    assert list(d1.columns) == schema.columns
    # 2 lignes (doublon filtré)
    assert len(d1[d1["ts"].str.startswith("2025-08-21")]) == 2

    # day2
    d2 = pd.concat(pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(tmp_path / "202*_*quotes_*.csv"))))
    assert len(d2[d2["ts"].str.startswith("2025-08-22")]) == 1


def test_parquet_parts_are_readable(tmp_path: Path):
    schema = Schema(columns=["ts", "x", "y"], dtypes={"ts": "string", "x": "float64", "y": "int64"})
    sink = RotatingParquetSink(tmp_path, "pnl", schema, now_fn=_fake_now_sequence([1_700_000_000.0]))

    p = sink.write_rows([
        {"ts": "2025-08-21T00:00:00Z", "x": 1.23, "y": 5},
        {"ts": "2025-08-21T00:00:01Z", "x": 2.34, "y": 6},
    ])
    assert p.exists()

    files = glob.glob(str(tmp_path / "202*_*pnl_*.parquet"))
    assert len(files) == 1

    df = pd.read_parquet(files[0])
    assert list(df.columns) == ["ts", "x", "y"]
    assert len(df) == 2


def test_no_tmp_leftovers(tmp_path: Path):
    schema = Schema(columns=["ts", "a"])
    sink = RotatingCsvSink(tmp_path, "trades", schema)

    sink.write_rows([{"ts": "t1", "a": 1}])

    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
