"""eeie.ingestion CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from eeie.ingestion.adapters import available_slugs, get_adapter, write_curated
from eeie.ingestion.raw import RawDatasetStatus, find_csv, verify, verify_all

app = typer.Typer(add_completion=False, help="EEIE real-dataset ingestion utilities.")


@app.callback()
def _root() -> None:
    pass


def _print_statuses(statuses: list[RawDatasetStatus]) -> int:
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


def _resolve_run_slugs(slug: str | None, all_: bool) -> list[str]:
    if slug is None or all_:
        return available_slugs()
    if slug not in available_slugs():
        raise typer.BadParameter(
            f"No adapter registered for {slug!r}. "
            f"Available: {', '.join(available_slugs()) or '(none)'}."
        )
    return [slug]


@app.command("run")
def cmd_run(
    slug: str | None = typer.Option(
        None,
        "--slug",
        "-s",
        help="Run only this dataset slug. Omit to run every adapter.",
    ),
    all_: bool = typer.Option(False, "--all", help="Run every registered adapter."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Base output directory; defaults to data/curated/.",
    ),
) -> None:
    """Run dataset adapter(s) and write curated parquet snapshots."""
    slugs = _resolve_run_slugs(slug, all_)
    failures = 0
    for s in slugs:
        try:
            raw_path = find_csv(s)
            frame = get_adapter(s)(raw_path)
            target = output / s if output is not None else None
            written = write_curated(frame, output_dir=target)
            typer.echo(
                f"[OK] {s}: {frame.summary()} -> {len(written)} parquet file(s)"
            )
        except Exception as exc:
            logger.exception("Adapter {} failed: {}", s, exc)
            typer.echo(f"[FAIL] {s}: {exc}")
            failures += 1
    typer.echo(f"Ran {len(slugs)} adapter(s); {failures} failed.")
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
