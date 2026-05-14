"""Real-world dataset registry.

Each dataset we ingest is described once here so adapters, tests, and the
``eeie-ingest`` CLI agree on slug, expected filenames, and target
canonical EEIE tables.

Downloads are manual per project policy: users place extracted files
under ``data/raw/<slug>/``. Phase 2 (and beyond) adapters read those
files and project them into the canonical schema defined in
``eeie.db.models`` / ``eeie.ingestion.schemas``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DatasetSpec(BaseModel):
    """Static description of a single raw dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slug: str = Field(..., description="Folder name under data/raw/.")
    title: str
    kaggle_url: str
    license_note: str
    primary_csv: str = Field(
        ..., description="Filename inside data/raw/<slug>/ that adapters will read."
    )
    expected_files: tuple[str, ...] = Field(
        default=(),
        description="All filenames the verifier should expect inside the slug folder.",
    )
    sample_columns: tuple[str, ...] = Field(
        default=(),
        description=(
            "Subset of columns the primary CSV must contain. Used by the verifier "
            "as a smoke check that the right archive was unzipped into the right slug."
        ),
    )
    target_canonical: tuple[str, ...] = Field(
        default=(),
        description=(
            "Names of canonical EEIE tables this dataset will be mapped into during "
            "Phase 2/3 adapter implementation."
        ),
    )
    description: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    "charging_patterns": DatasetSpec(
        slug="charging_patterns",
        title="Electric Vehicle Charging Patterns",
        kaggle_url=(
            "https://www.kaggle.com/datasets/valakhorasani/"
            "electric-vehicle-charging-patterns"
        ),
        license_note="Public Kaggle dataset; verify Kaggle terms before redistribution.",
        primary_csv="ev_charging_patterns.csv",
        expected_files=("ev_charging_patterns.csv",),
        sample_columns=(
            "User ID",
            "Vehicle Model",
            "Battery Capacity (kWh)",
            "Charging Station ID",
            "Charging Start Time",
            "Charging End Time",
            "Energy Consumed (kWh)",
            "Charging Rate (kW)",
            "Charging Cost (USD)",
            "State of Charge (Start %)",
            "State of Charge (End %)",
            "Charger Type",
            "User Type",
        ),
        target_canonical=("vehicles", "charging_events"),
        description=(
            "Per-session charging behaviour: timestamps, energy, rate, cost, "
            "start/end SOC, charger type, and user archetype."
        ),
    ),
    "battery_charging": DatasetSpec(
        slug="battery_charging",
        title="EV Battery Charging Dataset",
        kaggle_url=(
            "https://www.kaggle.com/datasets/programmer3/ev-battery-charging-dataset"
        ),
        license_note="Public Kaggle dataset; verify Kaggle terms before redistribution.",
        primary_csv="nev_battery_charging.csv",
        expected_files=("nev_battery_charging.csv",),
        sample_columns=(
            "timestamp",
            "SOC",
            "SOH",
            "terminal_voltage",
            "battery_current",
            "battery_temp",
            "ambient_temp",
            "charging_efficiency",
            "cycle_degradation",
        ),
        target_canonical=("telemetry",),
        description=(
            "Sequenced battery telemetry: SOC, SOH, voltage/current, temperatures, "
            "aging indicators. Calibrates the battery health hybrid engine."
        ),
    ),
    "station_availability": DatasetSpec(
        slug="station_availability",
        title="EV Charging Station Availability Tracking",
        kaggle_url=(
            "https://www.kaggle.com/datasets/likithagedipudi/"
            "ev-charging-station-availability-tracking"
        ),
        license_note="Public Kaggle dataset; verify Kaggle terms before redistribution.",
        primary_csv="ev_charging_station_data.csv",
        expected_files=("ev_charging_station_data.csv",),
        sample_columns=(
            "timestamp",
            "station_id",
            "network",
            "latitude",
            "longitude",
            "charger_type",
            "power_output_kw",
            "ports_total",
            "ports_available",
            "utilization_rate",
            "station_status",
            "current_price",
        ),
        target_canonical=("station_state",),
        description=(
            "Hourly availability and utilization for US public charging stations: "
            "ports total/available, status, dynamic pricing, local weather."
        ),
    ),
    "germany_charging": DatasetSpec(
        slug="germany_charging",
        title="Electric Vehicle Charging in Germany",
        kaggle_url=(
            "https://www.kaggle.com/datasets/mexwell/electric-vehicle-charging-in-germany"
        ),
        license_note=(
            "Sourced from the German Bundesnetzagentur public station registry; "
            "redistribution restrictions may apply."
        ),
        primary_csv="charging_data.csv",
        expected_files=("charging_data.csv",),
        sample_columns=(
            "betreiber",
            "art_der_ladeeinrichung",
            "anzahl_ladepunkte",
            "anschlussleistung",
            "steckertypen1",
            "p1_kw",
            "kreis_kreisfreie_stadt",
            "ort",
            "postleitzahl",
            "inbetriebnahmedatum",
            "breitengrad",
            "laengengrad",
        ),
        target_canonical=("station_state",),
        description=(
            "Static catalog of public charging stations in Germany (operator, type, "
            "power, connector mix, address, commissioning date). Maps to station_state "
            "at ts=epoch 0 per the static-catalog policy."
        ),
    ),
}


def get_dataset(slug: str) -> DatasetSpec:
    """Return the registered spec for ``slug``, raising ``ValueError`` if unknown."""
    try:
        return REGISTRY[slug]
    except KeyError as exc:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown dataset slug {slug!r}. Known slugs: {known}") from exc


def list_datasets() -> list[DatasetSpec]:
    """Return all registered datasets in stable order (sorted by slug)."""
    return [REGISTRY[k] for k in sorted(REGISTRY)]
