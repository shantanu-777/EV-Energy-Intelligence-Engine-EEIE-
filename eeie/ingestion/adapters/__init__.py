"""Per-dataset adapters that map raw CSVs to canonical EEIE schemas.

Concrete adapters land in Phase 2 (``charging_patterns``, ``battery_charging``)
and Phase 3 (``station_availability``, ``germany_charging``). This package
is an explicit placeholder so imports of ``eeie.ingestion.adapters``
resolve from Phase 1 onwards and tests can introspect it.
"""

from __future__ import annotations

__all__: list[str] = []
