"""Run the example kernels: model, NCU capture, joint analysis.

    python examples/tilelang/run.py all --output-dir reports/model
    python examples/tilelang/run.py gemm --mode all --device-uuid GPU-... --ncu /path/to/ncu --output-dir reports/gemm
    python examples/tilelang/run.py gemm --mode joint --observation reports/gemm/gemm/capture/ncu/observation.json \
        --output-dir reports/gemm-joint

Cases and their compile options are in ``cases.json``. ``model`` needs TileLang, no
GPU; ``capture`` needs TileLang, PyTorch, an NVIDIA GPU and Nsight Compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

from tilesight.tilelang import KernelRequest, profile_request  # noqa: E402
from tilesight.tilelang.report import write_bundle  # noqa: E402

CATALOG = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))


def request(name: str) -> KernelRequest:
    case = CATALOG["cases"][name]
    return KernelRequest(str(ROOT / case["file"]), case["function"], case["factory_args"], case["specialization"],
                         CATALOG["target_arch"], case["pass_configs"], case["bindings"])


def symbol(name: str) -> str:
    """Device kernel symbol from one lowering (no TileSight, no GPU)."""

    from tilesight.tilelang.capture import capture
    from tilesight.tilelang.entry import resolve
    resolved = resolve(request(name))
    return capture(resolved.prim_func, target_arch=CATALOG["target_arch"],
                   pass_configs=dict(resolved.pass_configs)).device.symbol


def run_capture(name: str, directory: Path, args: argparse.Namespace) -> Path:
    from tilesight.tilelang import ncu
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    case = CATALOG["cases"][name]
    kernel = request(name)
    entry = {"file": kernel.file, "function": kernel.function, "factory_args": kernel.factory_args,
             "specialization": kernel.specialization, "target_arch": kernel.target_arch,
             "pass_configs": kernel.pass_configs, "bindings": kernel.bindings, "compile_flags": case["compile_flags"]}
    (directory / "entry.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
    receipt = directory / "receipt.json"
    capture = CATALOG["capture"]
    req = {"schema_version": ncu.REQUEST_SCHEMA,
           "target": {"symbol": symbol(name), "name_base": "function", "launch_skip": 0},
           "device": {"uuid": args.device_uuid}, "capture": capture,
           "command": [sys.executable, "-m", "tilesight.tilelang.driver", "--entry", str(directory / "entry.json"),
                       "--receipt", str(receipt), "--warmup", str(capture["warmup"])],
           "environment": {"PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])},
           "driver_receipt_path": str(receipt)}
    observation = ncu.capture(req, directory / "ncu", ncu=args.ncu, timeout=args.timeout)
    if observation["status"] == "failed":
        raise RuntimeError("capture failed: %s" % observation["diagnostics"])
    return directory / "ncu" / "observation.json"


def run_model(name: str, directory: Path, args: argparse.Namespace, observation: Path = None) -> dict:
    profile = profile_request(request(name))
    if observation is not None and profile.status == "complete":
        from tilesight.tilelang.joint import joint, load_observation
        profile = joint(profile, load_observation(observation))
    write_bundle(profile, directory)
    payload = profile.payload()
    summary = {"case": name, "status": payload["status"], "output": str(directory)}
    if profile.status == "complete":
        summary["estimate_us"] = round(profile.estimate.kernel_body_s * 1e6, 2)
        if profile.observation:
            summary["measured_us"] = round(profile.observation["kernel_time_s"] * 1e6, 2)
            summary["compile_facts_only_us"] = round(profile.baseline_s * 1e6, 2) if profile.baseline_s else None
    else:
        summary["refusal"] = payload["refusal"]
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", choices=(*CATALOG["cases"], "all"))
    parser.add_argument("--mode", choices=("model", "capture", "joint", "all"), default="model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-uuid", help="GPU-... of the GPU to capture on")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--observation", type=Path, help="existing observation.json for --mode joint (one case)")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    names = list(CATALOG["cases"]) if args.case == "all" else [args.case]
    if args.mode in ("capture", "all") and not (args.device_uuid or "").startswith("GPU-"):
        parser.error("--mode %s needs --device-uuid GPU-..." % args.mode)
    if args.mode == "joint" and (args.observation is None or len(names) != 1):
        parser.error("--mode joint needs one case and --observation")
    status = 0
    for name in names:
        directory = args.output_dir / name
        observation = args.observation
        if args.mode in ("capture", "all"):
            observation = run_capture(name, directory / "capture", args)
            print(json.dumps({"case": name, "observation": str(observation)}), flush=True)
        if args.mode in ("model", "all"):
            summary = run_model(name, directory / "model", args)
            print(json.dumps(summary), flush=True)
            status = max(status, 0 if summary["status"] == "complete" else 3)
        if args.mode in ("joint", "all"):
            summary = run_model(name, directory / "joint", args, observation)
            print(json.dumps(summary), flush=True)
            status = max(status, 0 if summary["status"] == "complete" else 3)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
