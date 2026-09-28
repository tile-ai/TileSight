"""``python -m tilesight.tilelang``: profile one kernel and write the report.

Exit status: 0 complete, 3 refused (the report names the missing fact),
2 failed (bad arguments or an unexpected error; the JSON records it).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .api import SCHEMA, profile_source_file
from .report import write_bundle, write_json


def _json_object(value: str, option: str) -> Dict[str, Any]:
    text = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        if not path.is_file():
            raise ValueError("%s: file does not exist: %s" % (option, path))
        text = path.read_text(encoding="utf-8")
    try:
        result = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-finite number " + c)))
    except ValueError as error:
        raise ValueError("%s: invalid JSON: %s" % (option, error)) from error
    if not isinstance(result, dict):
        raise ValueError("%s: expected a JSON object" % option)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tilesight.tilelang", description=__doc__.split("\n")[0])
    p.add_argument("entry", metavar="FILE:FUNCTION", help="kernel object or factory in a Python file")
    p.add_argument("--target", required=True, help="CUDA architecture, e.g. sm_90a")
    p.add_argument("--output", required=True, metavar="REPORT.json")
    p.add_argument("--report-dir", metavar="DIR", help="also write report.html, CSV tables and a Perfetto trace")
    p.add_argument("--factory-args", default="{}", metavar="JSON|@FILE")
    p.add_argument("--specialization", default="{}", metavar="JSON|@FILE", help="keyword arguments of get_tir()")
    p.add_argument("--pass-configs", metavar="JSON|@FILE", help="TileLang pass configs (default: the kernel's own)")
    p.add_argument("--bindings", default="{}", metavar="JSON|@FILE",
                   help="shape symbols, integer scalar parameters or declared input contents")
    p.add_argument("--prim-func", help="PrimFunc name when the entry returns an IRModule with several")
    p.add_argument("--arch", default="H200_SXM", help="TileSight architecture class")
    p.add_argument("--arch-profile", default="set_to_microbench", help="method that sets the architecture constants")
    p.add_argument("--observation", metavar="OBSERVATION.json", help="NCU observation for a joint analysis")
    p.add_argument("--overwrite", action="store_true", help="replace existing report files")
    return p


def _check_targets(arguments: argparse.Namespace) -> Optional[str]:
    output = Path(arguments.output).expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        return "%s exists (use --overwrite)" % output
    if arguments.report_dir:
        directory = Path(arguments.report_dir).expanduser().resolve()
        if directory == output or directory in output.parents:
            return "--report-dir must not contain --output"
        if directory.exists() and any(directory.iterdir()) and not arguments.overwrite:
            return "%s is not empty (use --overwrite)" % directory
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = parser()
    arguments = p.parse_args(argv)
    problem = _check_targets(arguments)
    if problem:
        p.error(problem)
    if ":" not in arguments.entry:
        p.error("entry must be FILE:FUNCTION")
    file, function = arguments.entry.rsplit(":", 1)
    output = Path(arguments.output).expanduser()
    try:
        observation = None
        if arguments.observation:
            from .joint import load_observation
            observation = load_observation(arguments.observation)
        profile = profile_source_file(
            file, function,
            factory_args=_json_object(arguments.factory_args, "--factory-args"),
            specialization=_json_object(arguments.specialization, "--specialization"),
            target_arch=arguments.target,
            pass_configs=None if arguments.pass_configs is None else _json_object(arguments.pass_configs, "--pass-configs"),
            bindings=_json_object(arguments.bindings, "--bindings"),
            prim_func_name=arguments.prim_func,
            arch=arguments.arch, arch_profile=arguments.arch_profile)
        if observation is not None and profile.status == "complete":
            from .joint import joint
            profile = joint(profile, observation)
    except Exception as error:  # the JSON report is the only data channel; record the failure there
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"schema": SCHEMA, "status": "failed", "error": {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc().splitlines()[-12:]}}, indent=1) + "\n", encoding="utf-8")
        print("failed: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return 2
    write_json(profile, output)
    if arguments.report_dir:
        write_bundle(profile, arguments.report_dir)
    if profile.status != "complete":
        print("refused: %(code)s: %(message)s" % profile.refusal, file=sys.stderr)
        return 3
    print("%s: %.2f us" % (profile.payload()["launch"]["symbol"], profile.estimate.kernel_body_s * 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
