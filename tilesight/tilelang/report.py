"""Report files for one profile: JSON, a self-contained HTML page, CSV tables and a Perfetto trace.

The JSON payload is the source of truth; HTML and CSV only render it. The
Perfetto trace is TileSight's own export of the one-CTA-per-SM analysis.
"""

from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..modeling.program.perfetto import perfetto_trace
from .api import Profile


def _publish(temporary: str, path: Path) -> None:
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(temporary, 0o666 & ~umask)
    os.replace(temporary, path)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        _publish(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_json(profile: Profile, path: Any) -> Path:
    target = Path(path)
    _atomic_write(target, json.dumps(profile.payload(), indent=1, sort_keys=True, default=str) + "\n")
    return target


# --------------------------------------------------------------------------- CSV

def _csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=str(path.parent), delete=False,
                                     prefix=path.name + ".") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    _publish(stream.name, path)
    return path


def write_tables(payload: Dict[str, Any], directory: Path) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = [_csv(directory / "diagnoses.csv", ("kind", "status", "severity", "scope", "summary"),
                ((d["kind"], d["status"], d["severity"], d["scope"], d["summary"]) for d in payload.get("diagnoses", ())))]
    if "ops" in payload:
        resources = sorted({r for op in payload["ops"] for r in op["service_s"]})
        out.append(_csv(directory / "ops.csv",
                        ["path", "kind", "count", "completion_latency_s", "ddr_read_bytes", "l2_read_request_bytes"]
                        + ["service_%s_s" % r for r in resources],
                        ([op["path"], op["kind"], op["count"], op["latency_s"], op["traffic"]["ddr_read_bytes"],
                          op["traffic"]["l2_read_request_bytes"]] + [op["service_s"].get(r, 0.0) for r in resources]
                         for op in payload["ops"])))
        out.append(_csv(directory / "loops.csv",
                        ("path", "trips", "total_s", "selected_ii_s", "resource_ii_s", "recurrence_ii_s", "credit_ii_s"),
                        ((loop["path"], loop["trips"], loop["total_s"]) + tuple(
                            (loop["ii"] or {}).get(k) for k in ("selected_ii_s", "resource_ii_s", "recurrence_ii_s",
                                                                "credit_ii_s"))
                         for loop in payload["loops"])))
    return out


# --------------------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1f21;--muted:#5f6368;--line:#e2e2df;--card:#fff;--warn:#b3541e;--ok:#2f7d4f;--info:#3b5b8c;--bar:#3b5b8c}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--fg:#e6e6e3;--muted:#a0a4a8;--line:#2c2f33;--card:#1c1f23;--warn:#e58a4e;--ok:#5fbf86;--info:#8fb0e0;--bar:#8fb0e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:14px}.kpi b{display:block;font-size:20px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.chip{display:inline-block;border-radius:10px;padding:0 8px;font-size:12px;border:1px solid currentColor;white-space:nowrap}
.detected.warning{color:var(--warn)}.detected.info{color:var(--info)}.not_detected{color:var(--ok)}.inconclusive,.not_evaluated{color:var(--muted)}
.bar{display:inline-block;height:8px;background:var(--bar);border-radius:2px;vertical-align:middle;margin-right:6px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}ul{margin:4px 0;padding-left:20px}
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _us(seconds: Optional[float]) -> str:
    return "n/a" if seconds is None else "%.2f µs" % (seconds * 1e6)


def _ns(seconds: Optional[float]) -> str:
    return "n/a" if seconds is None else "%.1f ns" % (seconds * 1e9)


def _bytes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return "%.2f %s" % (value / scale, unit)
    return "%.0f B" % value


def _table(header: Sequence[str], rows: Iterable[Sequence[str]], numeric: Sequence[int] = ()) -> str:
    head = "".join("<th>%s</th>" % _e(h) for h in header)
    body = "".join("<tr>%s</tr>" % "".join('<td%s>%s</td>' % (' class="n"' if i in numeric else "", cell)
                                           for i, cell in enumerate(row)) for row in rows)
    return '<div class="card"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (head, body)


def _bars(values: Dict[str, float], n: int = 4) -> str:
    items = list(values.items())[:n]
    return "<br>".join('<span class="bar" style="width:%dpx"></span>%s %.0f%%' % (max(1, int(80 * min(u, 1.0))), _e(r), 100 * u)
                       for r, u in items)


def render_html(payload: Dict[str, Any]) -> str:
    request = payload["request"]
    launch = payload.get("launch", {})
    title = "%s:%s" % (Path(request["file"]).name, request["function"])
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             "<title>TileSight profile</title><style>%s</style></head><body><main>" % _CSS,
             "<h1>%s</h1><div class=\"muted\">%s · target %s · %s</div>"
             % (_e(title), _e(launch.get("symbol", "")), _e(request["target_arch"]), _e(payload["status"]))]
    if payload["status"] != "complete":
        refusal = payload.get("refusal") or {}
        parts.append('<h2>Refused</h2><div class="card"><b>%s</b><p>%s</p></div>'
                     % (_e(refusal.get("code", "")), _e(refusal.get("message", ""))))
    estimate = payload.get("estimate")
    if estimate:
        residency = payload.get("residency", {})
        resident = estimate["resident_ctas_per_sm"]
        bound = ", ".join(residency.get("binding", ()))
        if residency.get("ctas_per_sm") not in (None, resident):
            resident_text = "%d (NCU; TileSight %d, %s)" % (resident, residency["ctas_per_sm"], bound)
        else:
            resident_text = "%d (%s)" % (resident, bound)
        kpis = [("Estimated kernel time", _us(estimate["kernel_body_s"])),
                ("Resident CTAs per SM", resident_text),
                ("Grid × threads", "%s × %d" % (" × ".join(str(v) for v in launch.get("grid", {}).values()), launch.get("threads", 0))),
                ("Waves (one CTA per SM)", str(estimate["waves_single_cta"]))]
        observation = payload.get("observation")
        if observation:
            kpis.insert(1, ("Measured kernel time", _us(observation["kernel_time_s"])))
        parts.append('<div class="kpis">%s</div>' % "".join('<div class="card kpi"><span class="muted">%s</span><b>%s</b></div>'
                                                           % (_e(k), _e(v)) for k, v in kpis))
        if estimate["composition"] != "single_cta":
            parts.append('<p class="muted">%d resident CTAs per SM are composed with TileSight\'s per-SM resource bound; '
                         "if resident CTAs never overlapped different resources the time would be %s.</p>"
                         % (estimate["resident_ctas_per_sm"], _e(_us(estimate["upper_s"]))))
    observation = payload.get("observation")
    if observation:
        parts.append("<h2>Measurement</h2>")
        association = observation["association"]
        if association["status"] != "matched":
            parts.append('<div class="card">The observation does not match this kernel: %s. Nothing was adopted.</div>'
                         % _e("; ".join(association["problems"])))
        else:
            parts.append(_table(("step", "estimate"), ((_e(s["step"]), _us(s["kernel_body_s"])) for s in observation["steps"]),
                                numeric=(1,)))
            parts.append("<p></p>")

            def fmt(quantity: str, value: Any) -> str:
                if value is None:
                    return "n/a"
                if "(s)" in quantity:
                    return _us(value)
                if "bytes" in quantity:
                    return _bytes(value)
                return "%.4g" % value
            parts.append(_table(("quantity", "model", "measured", "model / measured"), (
                (_e(r["quantity"]), fmt(r["quantity"], r["model"]), fmt(r["quantity"], r["measured"]),
                 "n/a" if r["ratio"] is None else "%.2f" % r["ratio"]) for r in observation.get("comparison", ())),
                numeric=(1, 2, 3)))
    if payload.get("diagnoses"):
        parts.append("<h2>Diagnoses</h2>")
        parts.append(_table(("status", "kind", "scope", "finding"), (
            ('<span class="chip %s %s">%s</span>' % (_e(d["status"]), _e(d["severity"]), _e(d["status"].replace("_", " "))),
             _e(d["kind"]), _e(d["scope"]), _e(d["summary"])) for d in payload["diagnoses"])))
    if payload.get("loops"):
        parts.append("<h2>Loops</h2>")
        parts.append(_table(("loop", "trips", "time", "II selected", "resource", "recurrence", "credit", "busy per iteration"), (
            (_e(loop["path"]), _e(loop["trips"]), _us(loop["total_s"]),
             _ns((loop["ii"] or {}).get("selected_ii_s")), _ns((loop["ii"] or {}).get("resource_ii_s")),
             _ns((loop["ii"] or {}).get("recurrence_ii_s")), _ns((loop["ii"] or {}).get("credit_ii_s")),
             _bars({r: s / loop["ii"]["selected_ii_s"] for r, s in sorted(loop["service_per_iteration_s"].items(), key=lambda x: -x[1])})
             if loop["ii"] and loop["ii"].get("selected_ii_s") else "")
            for loop in payload["loops"]), numeric=(1, 2, 3, 4, 5, 6)))
    if payload.get("ops"):
        parts.append("<h2>Operations</h2>")
        parts.append(_table(("operation", "kind", "count", "latency", "service per execution", "DRAM read"), (
            ("<code>%s</code>" % _e(op["path"]), _e(op["kind"]), _e(op["count"]), _ns(op["latency_s"]),
             ", ".join("%s %s" % (_e(r), _ns(s)) for r, s in sorted(op["service_s"].items(), key=lambda x: -x[1])[:3]),
             _bytes(op["traffic"]["ddr_read_bytes"])) for op in payload["ops"]), numeric=(2, 3, 5)))
    if payload.get("traffic"):
        t = payload["traffic"]
        parts.append("<h2>Global traffic</h2>")
        parts.append(_table(("", "read", "write"), (
            ("payload", _bytes(t["payload_read_bytes"]), _bytes(t["payload_write_bytes"])),
            ("L1.5 requests", _bytes(t["l1_5_read_request_bytes"]), _bytes(t["l1_5_write_request_bytes"])),
            ("L2 requests", _bytes(t["l2_read_request_bytes"]), _bytes(t["l2_write_request_bytes"])),
            ("DRAM", _bytes(t["ddr_read_bytes"]), _bytes(t["ddr_write_bytes"]))), numeric=(1, 2)))
    if payload.get("residency"):
        r = payload["residency"]
        parts.append("<h2>Residency</h2>")
        parts.append('<p class="muted">%d CTAs per SM from TileSight\'s declared-storage checks (at most '
                     "max_blocks_per_sm = %d); bound by %s.</p>"
                     % (r["ctas_per_sm"], r["max_blocks_per_sm"], _e(", ".join(r["binding"]))))
        storages = sorted(set(r["storage_per_cta_bytes"]) | set(r["capacity_per_sm_bytes"]))
        parts.append(_table(("storage", "declared per CTA", "capacity per SM"), (
            (_e(k), _bytes(r["storage_per_cta_bytes"].get(k)), _bytes(r["capacity_per_sm_bytes"].get(k)))
            for k in storages), numeric=(1, 2)))
    notes = list(payload.get("notes", ())) + list(payload.get("tilesight_notes", ()))
    if notes:
        parts.append("<h2>Notes</h2><div class=\"card\"><ul>%s</ul></div>" % "".join("<li>%s</li>" % _e(n) for n in notes))
    identity = payload.get("identity")
    if identity:
        rows = [("TileLang", "%s (%s…)" % (dict(identity["compiler"]).get("tilelang_version"),
                                           dict(identity["compiler"]).get("tilelang_fingerprint", "")[:12])),
                ("TileSight", identity.get("tilesight_version") or "n/a"),
                ("Architecture", "%s, %s" % (identity["arch"], identity["arch_profile"])),
                ("Context", json.dumps(identity.get("context") or {})),
                ("Planned IR sha256", identity["planned_ir_sha256"]),
                ("Pass configs", json.dumps(identity["pass_configs"], default=str))]
        parts.append("<h2>Identity</h2>")
        parts.append(_table(("", ""), ((_e(k), "<code>%s</code>" % _e(v)) for k, v in rows)))
    parts.append("</main></body></html>")
    return "".join(parts)


def write_bundle(profile: Profile, directory: Any, *, perfetto: bool = True) -> Dict[str, Path]:
    """report.json, report.html, CSV tables and (for a complete analysis) trace.perfetto.json."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = profile.payload()
    out = {"json": write_json(profile, target / "report.json")}
    _atomic_write(target / "report.html", render_html(payload))
    out["html"] = target / "report.html"
    for path in write_tables(payload, target / "tables"):
        out["table:" + path.stem] = path
    if perfetto and profile.result is not None:
        trace = perfetto_trace(profile.result)
        _atomic_write(target / "trace.perfetto.json", json.dumps(trace))
        out["perfetto"] = target / "trace.perfetto.json"
    return out
