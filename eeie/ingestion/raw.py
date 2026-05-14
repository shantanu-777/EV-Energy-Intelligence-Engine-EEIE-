"""Raw dataset location and verification helpers.

Datasets are downloaded manually by the project owner and extracted under
``<settings.data_dir>/raw/<slug>/``. This module is the single source of
truth for resolving those paths and confirming the on-disk layout matches
the registry. Phase 2/3 adapters call :func:`find_csv` to obtain a
``Path`` they can pass to ``pandas.read_csv``.

Nothing here downloads, mutates, or writes files: it is read-only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from eeie.config.settings import get_settings
from eeie.ingestion.datasets import DatasetSpec, get_dataset, list_datasets

__all__ = [
    "RawDatasetStatus",
    "find_csv",
    "raw_dir",
    "raw_root",
    "verify",
    "verify_all",
]


def raw_root() -> Path:
    """Return the top-level raw-data directory ``<data_dir>/raw/``."""
    return get_settings().data_dir / "raw"


def raw_dir(slug: str) -> Path:
    """Return ``<data_dir>/raw/<slug>/`` (creation is not enforced)."""
    return raw_root() / slug


@dataclass(frozen=True)
class RawDatasetStatus:
    """On-disk status of one registered dataset."""

    spec: DatasetSpec
    directory: Path
    directory_exists: bool
    found_files: tuple[str, ...]
    missing_expected: tuple[str, ...]
    primary_csv_path: Path | None
    missing_sample_columns: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True iff dir exists, expected files are present, and headers match."""
        return (
            self.directory_exists
            and not self.missing_expected
            and self.primary_csv_path is not None
            and not self.missing_sample_columns
        )

    def to_lines(self) -> list[str]:
        """Human-readable multi-line status block used by the CLI."""
        marker = "OK" if self.ok else "MISSING"
        return [
            f"[{marker}] {self.spec.slug}  ({self.spec.title})",
            f"  dir             : {self.directory}",
            f"  dir exists      : {self.directory_exists}",
            f"  found files     : {', '.join(self.found_files) or '(none)'}",
            f"  missing files   : {', '.join(self.missing_expected) or '(none)'}",
            f"  primary csv     : {self.primary_csv_path or '(not found)'}",
            f"  missing columns : "
            f"{', '.join(self.missing_sample_columns) or '(none)'}",
            f"  target tables   : "
            f"{', '.join(self.spec.target_canonical) or '(unmapped)'}",
        ]


def _read_header(path: Path) -> tuple[str, ...]:
    """Return the column names of the first CSV header row."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return ()
    return tuple(col.strip() for col in header)


def verify(slug: str) -> RawDatasetStatus:
    """Verify the on-disk layout for one dataset. Never raises on missing data."""
    spec = get_dataset(slug)
    directory = raw_dir(slug)
    directory_exists = directory.is_dir()

    found_files: tuple[str, ...] = ()
    if directory_exists:
        found_files = tuple(
            sorted(p.name for p in directory.iterdir() if p.is_file())
        )

    missing_expected = tuple(
        name for name in spec.expected_files if name not in found_files
    )

    primary_csv_path: Path | None = None
    if directory_exists:
        candidate = directory / spec.primary_csv
        if candidate.is_file():
            primary_csv_path = candidate

    missing_sample_columns: tuple[str, ...] = ()
    if primary_csv_path is not None and spec.sample_columns:
        header_set = set(_read_header(primary_csv_path))
        missing_sample_columns = tuple(
            col for col in spec.sample_columns if col not in header_set
        )

    return RawDatasetStatus(
        spec=spec,
        directory=directory,
        directory_exists=directory_exists,
        found_files=found_files,
        missing_expected=missing_expected,
        primary_csv_path=primary_csv_path,
        missing_sample_columns=missing_sample_columns,
    )


def verify_all() -> list[RawDatasetStatus]:
    """Verify every registered dataset in stable (sorted) slug order."""
    return [verify(spec.slug) for spec in list_datasets()]


def find_csv(slug: str, filename: str | None = None) -> Path:
    """Return a ``Path`` to a CSV inside ``raw_dir(slug)``.

    If ``filename`` is ``None`` the registered ``primary_csv`` is used.
    Raises ``FileNotFoundError`` if the file is missing -- callers get a
    clear, single failure mode instead of pandas's generic IO error.
    """
    spec = get_dataset(slug)
    target = filename or spec.primary_csv
    path = raw_dir(slug) / target
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw file not found: {path}. "
            f"Run `python -m eeie.ingestion.cli verify --slug {slug}` for details."
        )
    return path
