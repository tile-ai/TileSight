"""Run one kernel request on the GPU for an NCU capture.

    python -m tilesight.tilelang.driver --entry ENTRY.json --receipt RECEIPT.json --warmup 5

``ENTRY.json`` holds the ``KernelRequest`` fields plus optional ``compile_flags``.
The kernel is compiled by TileLang with the request's target and pass configs,
called ``--warmup`` times, and then once more between ``cudaProfilerStart`` and
``cudaProfilerStop`` (capture with ``profile_from_start=off``). Every buffer
argument is a zero tensor. The receipt records the device and the SHA-256 of
the compiled CUDA source, which the joint analysis compares with the model's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from typing import Any, Dict, List

from .entry import KernelRequest, resolve


def _shape(buffer: Any, bindings: Dict[str, int]) -> List[int]:
    shape = []
    for extent in buffer.shape:
        if hasattr(extent, "value"):
            shape.append(int(extent.value))
        elif str(extent) in bindings:
            shape.append(int(bindings[str(extent)]))
        else:
            raise ValueError("buffer %s has symbolic extent %s; pass it in bindings" % (buffer.name, extent))
    return shape


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--entry", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    import tilelang
    import torch
    from tilelang import tvm

    with open(args.entry, encoding="utf-8") as stream:
        entry = json.load(stream)
    compile_flags = entry.pop("compile_flags", None)
    request = KernelRequest(**entry)
    resolved = resolve(request)
    prim_func = resolved.prim_func
    target = tvm.target.Target({"kind": "cuda", "arch": request.target_arch})
    kernel = tilelang.compile(prim_func, out_idx=[], target=target, pass_configs=dict(resolved.pass_configs),
                              compile_flags=compile_flags)
    source = kernel.get_kernel_source()
    bindings = dict(request.bindings)
    tensors = []
    for param in prim_func.params:
        buffer = prim_func.buffer_map.get(param)
        if buffer is None:
            raise ValueError("scalar parameter %s is not supported by this driver" % param)
        dtype = getattr(torch, str(buffer.dtype))
        tensors.append(torch.zeros(_shape(buffer, bindings), dtype=dtype, device="cuda"))
    for _ in range(args.warmup):
        kernel(*tensors)
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    uuid = str(properties.uuid)
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    clocks = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-gpu=clocks.sm,clocks.mem,driver_version",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    receipt = {
        "device_uuid": uuid, "device_name": properties.name, "sm_count": properties.multi_processor_count,
        "nvidia_smi_clocks_sm_mem_driver": clocks, "tilelang_version": tilelang.__version__,
        "kernel_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "pass_configs": {k: v for k, v in resolved.pass_configs}, "compile_flags": compile_flags,
        "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")} for t in tensors],
        "input_values": "zeros", "warmup": args.warmup, "target_calls": 1,
    }
    with open(args.receipt, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1)
    torch.cuda.cudart().cudaProfilerStart()
    kernel(*tensors)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
