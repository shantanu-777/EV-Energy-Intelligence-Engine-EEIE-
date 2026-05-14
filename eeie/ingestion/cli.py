"""Ingestion CLI.

Phase 1 exposes only ``verify``, which reports whether each registered
dataset is on disk where the adapters (Phase 2/3) will expect it.

Invoke via either::

    python -m eeie.ingestion.cli verify --all
    eeie-ingest verify --slug charging_patterns
"""

from __future__ import annotations

import typer
from loguru import logger

from eeie.ingestion.raw import RawDatasetStatus, verify, verify_all

app = typer.Typer(add_completion=False, help="EEIE real-dataset ingestion utilities.")


@app.callback()
def _root() -> None:
    """Root callback so Typer keeps the subcommand layer even with one command."""


def _print_statuses(statuses: list[RawDatasetStatus]) -> int:
    """Print each status block; return the number of failures."""
    failures = 0
    for status in statuses:
        for line in status.to_lines():
            typer.echo(line)
        typer.echo("")
        if not status.ok:
            failures += 1
    return failures


@app.command("verify")
def cmd_verify(
    slug: str | None = typer.Option(
        None,
        "--slug",
        "-s",
        help="Verify only this dataset slug. Omit to verify all.",
    ),
    all_: bool = typer.Option(
        False,
        "--all",
        help="Verify every registered dataset (default if --slug is omitted).",
    ),
) -> None:
    """Check that raw datasets are on disk under data/raw/<slug>/."""
    statuses = verify_all() if (slug is None or all_) else [verify(slug)]
    failures = _print_statuses(statuses)
    total = len(statuses)
    typer.echo(f"Verified {total} dataset(s); {failures} need attention.")
    if failures:
        logger.warning("{} dataset(s) failed verification.", failures)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
