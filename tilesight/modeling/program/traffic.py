"""Read/write-split traffic at every storage level.

Totals such as ``l2_request_bytes`` are *derived* from the read/write parts
and cannot be set independently.  RFO is already part of ``ddr_read_bytes``
and terminal flush / eviction writeback are part of ``ddr_write_bytes``; the
event fields keep their own accounting and are never added again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple

from ..cache.result import CacheTraffic
from ..errors import ModelingValidationError

LEVEL_FIELDS = (
    ("ddr", "ddr_read_bytes", "ddr_write_bytes"),
    ("l2", "l2_read_request_bytes", "l2_write_request_bytes"),
    ("l1_5", "l1_5_read_request_bytes", "l1_5_write_request_bytes"),
    ("smem", "smem_read_bytes", "smem_write_bytes"),
    ("register", "register_read_bytes", "register_write_bytes"),
    ("tmem", "tmem_read_bytes", "tmem_write_bytes"),
)
ON_CHIP_LEVELS = ("smem", "register", "tmem")
VALUE_FIELDS = (
    "payload_read_bytes", "payload_write_bytes", "ddr_read_bytes", "ddr_write_bytes",
    "l2_read_request_bytes", "l2_write_request_bytes", "l1_5_read_request_bytes",
    "l1_5_write_request_bytes", "smem_read_bytes", "smem_write_bytes", "register_read_bytes",
    "register_write_bytes", "tmem_read_bytes", "tmem_write_bytes", "rfo_bytes",
    "dirty_created_bytes", "dirty_eviction_writeback_bytes", "dirty_terminal_flush_bytes",
    "dirty_resident_bytes", "transaction_amplification_bytes",
)
LEGACY_TOTALS = {
    "l2_request_bytes": ("l2_read_request_bytes", "l2_write_request_bytes"),
    "l1_5_request_bytes": ("l1_5_read_request_bytes", "l1_5_write_request_bytes"),
}


def _nonnegative(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("Traffic.%s must be finite and non-negative" % label)
    return result


@dataclass(frozen=True)
class Traffic:
    payload_read_bytes: float = 0.0
    payload_write_bytes: float = 0.0
    ddr_read_bytes: float = 0.0
    ddr_write_bytes: float = 0.0
    l2_read_request_bytes: float = 0.0
    l2_write_request_bytes: float = 0.0
    l1_5_read_request_bytes: float = 0.0
    l1_5_write_request_bytes: float = 0.0
    smem_read_bytes: float = 0.0
    smem_write_bytes: float = 0.0
    register_read_bytes: float = 0.0
    register_write_bytes: float = 0.0
    tmem_read_bytes: float = 0.0
    tmem_write_bytes: float = 0.0
    rfo_bytes: float = 0.0
    dirty_created_bytes: float = 0.0
    dirty_eviction_writeback_bytes: float = 0.0
    dirty_terminal_flush_bytes: float = 0.0
    dirty_resident_bytes: float = 0.0
    transaction_amplification_bytes: float = 0.0
    unknown: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in VALUE_FIELDS:
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        unknown = []
        for item in (self.unknown.items() if isinstance(self.unknown, Mapping) else self.unknown):
            name, reason = item
            if name not in VALUE_FIELDS:
                raise ModelingValidationError("Traffic.unknown names unknown field %r" % name)
            if getattr(self, name) != 0.0:
                raise ModelingValidationError("Traffic.%s is marked unknown but carries a value" % name)
            unknown.append((name, str(reason)))
        object.__setattr__(self, "unknown", tuple(sorted(dict(unknown).items())))
        if self.rfo_bytes > self.ddr_read_bytes + 1e-9 * max(1.0, self.ddr_read_bytes):
            raise ModelingValidationError("Traffic.rfo_bytes must already be included in ddr_read_bytes")
        writeback = self.dirty_eviction_writeback_bytes + self.dirty_terminal_flush_bytes
        if writeback > self.ddr_write_bytes + 1e-9 * max(1.0, self.ddr_write_bytes):
            raise ModelingValidationError("Traffic dirty writeback must already be included in ddr_write_bytes")
        dirty_error = abs(self.dirty_created_bytes - writeback - self.dirty_resident_bytes)
        if dirty_error > 1e-9 * max(self.dirty_created_bytes, 1.0):
            raise ModelingValidationError("dirty traffic violates created = writeback + resident conservation")

    # -- unknown handling -----------------------------------------------------

    @property
    def unknown_fields(self) -> Tuple[str, ...]:
        return tuple(name for name, _reason in self.unknown)

    def is_known(self, name: str) -> bool:
        return name not in self.unknown_fields

    def value(self, name: str) -> Optional[float]:
        """Field value (or derived total), or ``None`` when any part is unknown."""

        if name in LEGACY_TOTALS:
            return self.value_sum(*LEGACY_TOTALS[name])
        if name not in VALUE_FIELDS:
            raise KeyError(name)
        return None if name in self.unknown_fields else getattr(self, name)

    def unknown_reason(self, name: str) -> Optional[str]:
        return dict(self.unknown).get(name)

    def with_unknown(self, names: Any, reason: str) -> "Traffic":
        items = dict(self.unknown)
        values = {name: getattr(self, name) for name in VALUE_FIELDS}
        for name in names:
            items[name] = reason
            values[name] = 0.0
        return Traffic(unknown=tuple(items.items()), **values)

    # -- derived totals -------------------------------------------------------

    @property
    def l2_request_bytes(self) -> Optional[float]:
        """Derived L2 total; ``None`` when either direction is unknown."""

        return self.value_sum("l2_read_request_bytes", "l2_write_request_bytes")

    @property
    def l1_5_request_bytes(self) -> Optional[float]:
        return self.value_sum("l1_5_read_request_bytes", "l1_5_write_request_bytes")

    @property
    def dirty_writeback_bytes(self) -> float:
        return self.dirty_eviction_writeback_bytes + self.dirty_terminal_flush_bytes

    def level_bytes(self, level: str) -> Optional[float]:
        """Read + write bytes of one level; ``None`` when either is unknown."""

        for name, read, write in LEVEL_FIELDS:
            if name == level:
                return self.value_sum(read, write)
        raise KeyError(level)

    def unknown_among(self, names: Any) -> Tuple[str, ...]:
        return tuple(name for name in names if name in self.unknown_fields)

    # -- arithmetic -------------------------------------------------------------

    def add(self, other: "Traffic") -> "Traffic":
        """Field-wise sum; a field unknown on either side stays unknown."""

        unknown = dict(self.unknown)
        unknown.update(dict(other.unknown))
        values = {
            name: 0.0 if name in unknown else getattr(self, name) + getattr(other, name)
            for name in VALUE_FIELDS
        }
        return Traffic(unknown=tuple(unknown.items()), **values)

    def scale(self, factor: float) -> "Traffic":
        return Traffic(unknown=self.unknown, **{name: getattr(self, name) * factor for name in VALUE_FIELDS})

    def close_to(self, other: "Traffic", tolerance: float = 1e-6) -> bool:
        if self.unknown_fields != other.unknown_fields:
            return False
        for name in VALUE_FIELDS:
            x, y = getattr(self, name), getattr(other, name)
            if abs(x - y) > tolerance * max(1.0, abs(x), abs(y)):
                return False
        return True

    def is_zero(self) -> bool:
        return all(getattr(self, name) == 0.0 for name in VALUE_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        data = {name: (None if name in self.unknown_fields else getattr(self, name)) for name in VALUE_FIELDS}
        data["l2_request_bytes"] = self.l2_request_bytes
        data["l1_5_request_bytes"] = self.l1_5_request_bytes
        data["unknown"] = [list(item) for item in self.unknown]
        return data

    def value_sum(self, *names: str) -> Optional[float]:
        if any(name in self.unknown_fields for name in names):
            return None
        return float(sum(getattr(self, name) for name in names))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], mode: Optional[str] = None) -> "Traffic":
        """Import a traffic record.

        Legacy totals (``l2_request_bytes``/``l1_5_request_bytes``) are accepted
        only when their direction can be recovered: either the split fields are
        present and consistent, or ``mode`` (``read``/``write``) names the
        direction.  Otherwise the import is rejected instead of dropping bytes.
        """

        values = {}
        unknown = [tuple(item) for item in data.get("unknown", ())]
        known_unknown = {name for name, _reason in unknown}
        for key, value in data.items():
            if key not in VALUE_FIELDS:
                continue
            if value is None:
                if key not in known_unknown:
                    unknown.append((key, "imported as null without a reason"))
                continue
            values[key] = float(value)
        for total_name, (read_name, write_name) in LEGACY_TOTALS.items():
            total = data.get(total_name)
            if total is None:
                continue
            total = float(total)
            has_split = read_name in values or write_name in values
            if has_split:
                split = values.get(read_name, 0.0) + values.get(write_name, 0.0)
                if abs(split - total) > 1e-9 * max(1.0, total):
                    raise ModelingValidationError(
                        "Traffic.from_dict: %s=%g disagrees with %s+%s=%g"
                        % (total_name, total, read_name, write_name, split)
                    )
            elif total > 0.0:
                if mode == "read":
                    values[read_name] = total
                elif mode == "write":
                    values[write_name] = total
                else:
                    raise ModelingValidationError(
                        "Traffic.from_dict: legacy total %s=%g has no read/write direction; "
                        "pass mode='read'/'write' or re-model with split fields"
                        % (total_name, total)
                    )
        return cls(unknown=tuple(unknown), **values)

    @classmethod
    def from_cache_access(cls, traffic: CacheTraffic, mode: str) -> "Traffic":
        """Split a per-access ``CacheTraffic`` by the access direction."""

        if mode not in ("read", "write", "atomic"):
            raise ModelingValidationError("cache access mode %r has no direction" % mode)
        is_read = mode == "read"
        return cls(
            payload_read_bytes=traffic.payload_read_bytes,
            payload_write_bytes=traffic.payload_write_bytes,
            ddr_read_bytes=traffic.ddr_read_bytes,
            ddr_write_bytes=traffic.ddr_write_bytes,
            l2_read_request_bytes=traffic.l2_request_bytes if is_read else 0.0,
            l2_write_request_bytes=0.0 if is_read else traffic.l2_request_bytes,
            l1_5_read_request_bytes=traffic.l1_5_request_bytes if is_read else 0.0,
            l1_5_write_request_bytes=0.0 if is_read else traffic.l1_5_request_bytes,
            rfo_bytes=traffic.rfo_bytes,
            dirty_created_bytes=traffic.dirty_created_bytes,
            dirty_eviction_writeback_bytes=traffic.dirty_eviction_writeback_bytes,
            dirty_terminal_flush_bytes=traffic.dirty_terminal_flush_bytes,
            dirty_resident_bytes=traffic.dirty_resident_bytes,
            transaction_amplification_bytes=traffic.transaction_amplification_bytes,
        )

    @classmethod
    def on_chip_move(cls, source: str, destination: str, executed_bytes: float) -> "Traffic":
        """Endpoint bytes of one move: source is read, destination is written."""

        values = {}  # type: Dict[str, float]
        if source in ON_CHIP_LEVELS:
            values["%s_read_bytes" % source] = executed_bytes
        if destination in ON_CHIP_LEVELS:
            values["%s_write_bytes" % destination] = executed_bytes
        return cls(**values)

    def with_on_chip_defaults(self, defaults: "Traffic") -> "Traffic":
        """Fill zero on-chip fields from ``defaults`` (declared values win)."""

        values = {name: getattr(self, name) for name in VALUE_FIELDS}
        unknown = dict(self.unknown)
        for level in ON_CHIP_LEVELS:
            for suffix in ("read", "write"):
                name = "%s_%s_bytes" % (level, suffix)
                if values[name] == 0.0 and name not in unknown:
                    values[name] = getattr(defaults, name)
        return Traffic(unknown=tuple(unknown.items()), **values)

    @classmethod
    def compute_internal(cls, reason: str = "compute-internal register/SMEM/TMEM accesses are not derivable from the description") -> "Traffic":
        """Traffic of a compute op: no global traffic, on-chip fields unknown."""

        names = ["%s_%s_bytes" % (level, suffix) for level in ON_CHIP_LEVELS for suffix in ("read", "write")]
        return cls().with_unknown(names, reason)


@dataclass(frozen=True)
class TrafficShare:
    """One op's share of a level/direction total within a denominator scope."""

    field: str
    scope: str
    denominator_id: str
    numerator_bytes: Optional[float]
    denominator_bytes: Optional[float]
    share: Optional[float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def traffic_shares(op_traffic: Traffic, scope: str, denominator_id: str, denominator: Traffic) -> tuple:
    result = []
    for _level, read, write in LEVEL_FIELDS:
        for name in (read, write):
            numerator = op_traffic.value(name)
            total = denominator.value(name)
            if numerator is None:
                result.append(TrafficShare(name, scope, denominator_id, None, total, None,
                                           "numerator unknown: %s" % op_traffic.unknown_reason(name)))
            elif total is None:
                result.append(TrafficShare(name, scope, denominator_id, numerator, None, None,
                                           "denominator unknown in %s: %s" % (denominator_id, denominator.unknown_reason(name))))
            elif total <= 0.0:
                result.append(TrafficShare(name, scope, denominator_id, numerator, total, None,
                                           "denominator is zero: no %s in %s" % (name, denominator_id)))
            else:
                result.append(TrafficShare(name, scope, denominator_id, numerator, total, numerator / total, "ok"))
    return tuple(result)


__all__ = ["LEGACY_TOTALS", "LEVEL_FIELDS", "ON_CHIP_LEVELS", "VALUE_FIELDS", "Traffic", "TrafficShare", "traffic_shares"]
