from __future__ import annotations
import csv
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # Parquet/typed CSV dépendront de pandas si dispo


# ----------------------------- helpers ------------------------------------------

def _utc_date_str(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _fsync_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _atomic_write_bytes(tmp_path: Path, final_path: Path, data: bytes) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(data)
        _fsync_file(f)
    # Remplace atomiquement
    os.replace(tmp_path, final_path)


# ----------------------------- Schema -------------------------------------------

@dataclass(frozen=True)
class Schema:
    """
    Schéma stable (ordre des colonnes + types "logiques").
    Pour CSV: on garantit l'ordre; pour Parquet: on passe des dtypes pandas.
    """
    columns: List[str]
    dtypes: Dict[str, str] = field(default_factory=dict)  # pandas dtype strings ("float64", "Int64", "string")

    def normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Filtre + ordonne les colonnes; remplace None par '' pour CSV."""
        out: Dict[str, Any] = {}
        for c in self.columns:
            v = row.get(c, None)
            out[c] = "" if v is None else v
        return out


# --------------------------- Base rotator ---------------------------------------

@dataclass
class DailyRotator:
    base_dir: Path
    stream: str
    now_fn: Callable[[], float] = field(default_factory=lambda: datetime.now(tz=timezone.utc).timestamp)

    current_day: str = field(init=False)
    seq: int = field(default=0, init=False)
    seen_ts: Dict[str, set] = field(default_factory=dict, init=False)  # day -> {str ts}

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_day = _utc_date_str(self.now_fn())

    def _roll_if_needed(self) -> None:
        d = _utc_date_str(self.now_fn())
        if d != self.current_day:
            self.current_day = d
            self.seq = 0
        if d not in self.seen_ts:
            self.seen_ts[d] = set()

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _mark_ts(self, ts: str) -> bool:
        """Retourne True si ts est nouveau (accepté), False si déjà vu (doublon)."""
        self._roll_if_needed()
        bucket = self.seen_ts[self.current_day]
        if ts in bucket:
            return False
        bucket.add(ts)
        return True


# --------------------------- CSV sink (part files) ------------------------------

class RotatingCsvSink:
    """
    Écrit en **fichiers part journaliers**:
      YYYY-MM-DD_<stream>_<seq>.csv
    Chaque écriture est atomique (temp -> rename). Pas de lignes corrompues en cas de crash.
    Anti-doublon par `ts` (si présent dans le row).

    Lecture: pandas.read_csv(glob("YYYY-MM-DD_<stream>_*.csv")) puis concat.
    """
    def __init__(
        self,
        base_dir: str | Path,
        stream: str,
        schema: Schema,
        *,
        fsync: bool = True,
        now_fn: Optional[Callable[[], float]] = None,
        include_header_each_part: bool = True,
    ) -> None:
        self.rot = DailyRotator(Path(base_dir), stream, now_fn or (lambda: datetime.now(tz=timezone.utc).timestamp()))
        self.schema = schema
        self.fsync = fsync
        self.include_header_each_part = include_header_each_part

        # On persiste un fichier JSON de schéma (utile pour consumers)
        meta = {"columns": self.schema.columns, "dtypes": self.schema.dtypes}
        (self.rot.base_dir / f"{stream}.schema.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def write_rows(self, rows: Iterable[Dict[str, Any]]) -> Path:
        """
        Écrit un batch dans un nouveau fichier-part atomique. Retourne le chemin final.
        Filtre les doublons de ts (si colonne 'ts' présente).
        """
        self.rot._roll_if_needed()
        day = self.rot.current_day
        seq = self.rot._next_seq()
        final = self.rot.base_dir / f"{day}_{self.rot.stream}_{seq:05d}.csv"
        tmp = final.with_suffix(final.suffix + f".{uuid.uuid4().hex}.tmp")

        # Prépare lignes CSV en mémoire (petits batches typiques)
        lines: List[str] = []
        w = csv.DictWriter  # juste pour connaître l'ordre
        if self.include_header_each_part:
            lines.append(",".join(self.schema.columns) + "\n")

        for r in rows:
            if "ts" in r:
                ts = str(r["ts"])
                if not self.rot._mark_ts(ts):
                    continue  # skip duplicate ts
            nr = self.schema.normalize_row(r)
            # sérialisation simple CSV; quotes si besoin
            out = []
            for c in self.schema.columns:
                v = nr[c]
                s = str(v)
                if any(ch in s for ch in [",", '"', "\n"]):
                    s = '"' + s.replace('"', '""') + '"'
                out.append(s)
            lines.append(",".join(out) + "\n")

        data = "".join(lines).encode("utf-8")
        _atomic_write_bytes(tmp, final, data)

        # fsync du dossier pour solidifier le rename (optionnel)
        if self.fsync:
            try:
                dfd = os.open(str(self.rot.base_dir), os.O_RDONLY)
                os.fsync(dfd)
                os.close(dfd)
            except Exception:
                pass

        return final


# --------------------------- Parquet sink (part files) --------------------------

class RotatingParquetSink:
    """
    Écrit des **parts journaliers Parquet**: YYYY-MM-DD_<stream>_<seq>.parquet
    Écriture atomique (temp -> rename). Dtypes pandas respectés si fournis.

    Lecture: pandas.read_parquet(glob("YYYY-MM-DD_<stream>_*.parquet")) puis concat.
    """
    def __init__(
        self,
        base_dir: str | Path,
        stream: str,
        schema: Schema,
        *,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        if pd is None:
            raise RuntimeError("pandas requis pour Parquet")
        self.rot = DailyRotator(Path(base_dir), stream, now_fn or (lambda: datetime.now(tz=timezone.utc).timestamp()))
        self.schema = schema
        meta = {"columns": self.schema.columns, "dtypes": self.schema.dtypes}
        (self.rot.base_dir / f"{stream}.schema.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def write_rows(self, rows: Iterable[Dict[str, Any]]) -> Path:
        self.rot._roll_if_needed()
        day = self.rot.current_day
        seq = self.rot._next_seq()
        final = self.rot.base_dir / f"{day}_{self.rot.stream}_{seq:05d}.parquet"
        tmp = final.with_suffix(final.suffix + f".{uuid.uuid4().hex}.tmp")

        # anti-doublon ts si présent
        materialized: List[Dict[str, Any]] = []
        for r in rows:
            if "ts" in r:
                ts = str(r["ts"])
                if not self.rot._mark_ts(ts):
                    continue
            materialized.append(self.schema.normalize_row(r))
        if not materialized:
            # rien à écrire: retourner un path "final" inexistant pour signaler no-op
            return final

        df = pd.DataFrame(materialized, columns=self.schema.columns)
        # Applique dtypes pandas s'ils sont fournis
        for col, dt in (self.schema.dtypes or {}).items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dt)
                except Exception:
                    pass

        # écriture temp -> rename (parquet gère les headers/types)
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)
        # fsync du dossier
        try:
            dfd = os.open(str(self.rot.base_dir), os.O_RDONLY)
            os.fsync(dfd)
            os.close(dfd)
        except Exception:
            pass

        return final


# ------------------------------ Facade ------------------------------------------

def daily_glob(base_dir: str | Path, stream: str, day: Optional[str] = None, suffix: str = "csv") -> str:
    day = day or _utc_date_str()
    return str(Path(base_dir) / f"{day}_{stream}_*.{suffix}")
