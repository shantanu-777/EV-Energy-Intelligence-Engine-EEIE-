"""eeie.ingestion CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from eeie.config import get_settings
from eeie.db import session_scope
from eeie.db.engine import get_engine
from eeie.db.hypertables import bootstrap_hypertables
from eeie.ingestion.adapters import available_slugs, get_adapter, write_curated
from eeie.ingestion.loader import load_curated_bundle
from eeie.ingestion.raw import RawDatasetStatus, find_csv, verify, verify_all
from eeie.ingestion.unifier import synthesize_training_grid, unify_pipeline

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


@app.command("load-db")
def cmd_load_db(
    slug: str | None = typer.Option(
        None,
        "--slug",
        "-s",
        help="Restrict to one slug (omit for every adapter whose CSV resolves).",
    ),
    all_: bool = typer.Option(False, "--all", help="Explicitly run every registered slug."),
    bootstrap_hypertables_flag: bool = typer.Option(
        True,
        "--bootstrap-hypertables/--no-bootstrap-hypertables",
        help="Bootstrap Timescale hypertables before insert (Postgres only).",
    ),
) -> None:
    """Unify curated adapters and bulk-load into Postgres for training."""
    settings = get_settings()
    settings.ensure_dirs()

    try:
        slugs = _resolve_run_slugs(slug, all_)
    except typer.BadParameter:
        raise

    frames = []
    for s in slugs:
        try:
            raw_path = find_csv(s)
            frames.append(get_adapter(s)(raw_path))
            typer.echo(f"[adapt] {s}: {frames[-1].summary()}")
        except Exception as exc:
            logger.exception("Skipping {} — {}", s, exc)
            typer.echo(f"[skip] {s}: {exc}")

    if not frames:
        typer.echo("No adapters produced frames; aborting.")
        raise typer.Exit(code=1)

    merged = unify_pipeline(frames)
    merged = synthesize_training_grid(merged, seed=settings.sim_seed)
    if merged.telemetry.empty:
        typer.echo(
            "Unified frame has no telemetry. Add battery_charging and/or charging_patterns "
            "(session-derived hourly rows)."
        )
        raise typer.Exit(code=1)

    engine = get_engine()
    if bootstrap_hypertables_flag:
        bootstrap_hypertables(engine)
    with session_scope() as session:
        counts = load_curated_bundle(session, merged)
    typer.echo(f"Loaded: {counts}")


if __name__ == "__main__":
    app()
