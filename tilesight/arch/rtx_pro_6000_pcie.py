"""RTX PRO 6000 Blackwell PCIe arch entry.

Thin subclass of :class:`B6000` (defined in ``tilesight/arch/b6000.py``).
The parent already encodes the GPU-internal whitepaper parameters (SM count,
tensor TFLOPS, HBM/L2 BW, frequencies) for the Blackwell GB202 desktop class.
This subclass adds:

    * explicit FP8 tensor TFLOPS (Blackwell FP8 dense throughput parity with INT8),
    * PCIe Gen 5 x16 host-link bandwidth (~32 GiB/s unidirectional / ~64 GiB/s bidi),
    * a stable ``RTXPro6000PCIe`` class name + ``get_arch()`` factory for the
      cluster preset and the experiment-side workload registry to consume.

Whitepaper sources (verified 2026-04-26):
    https://www.nvidia.com/en-us/design-visualization/rtx-pro-6000-blackwell/
    NVIDIA Blackwell GB202 product datasheet, rev 1.0.

Use: ``from tilesight.arch.rtx_pro_6000_pcie import RTXPro6000PCIe``.
"""

from .b6000 import B6000


PCIE_GEN5_X16_UNI_BW = 32 * 1024**3
PCIE_GEN5_X16_BI_BW = 64 * 1024**3
PCIE_GEN5_X16_LATENCY_S = 1.0e-6


class RTXPro6000PCIe(B6000):
    """Blackwell GB202 RTX PRO 6000 PCIe edition (desktop / workstation)."""

    def __init__(self) -> None:
        super().__init__()
        self.core = "RTX_PRO_6000_PCIE"
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2
        self.pcie_unidirectional_bw = PCIE_GEN5_X16_UNI_BW
        self.pcie_bidirectional_bw = PCIE_GEN5_X16_BI_BW
        self.pcie_latency = PCIE_GEN5_X16_LATENCY_S
        self.interconnect_type = "pcie5_x16"


def get_arch() -> RTXPro6000PCIe:
    return RTXPro6000PCIe()


if __name__ == "__main__":
    a = RTXPro6000PCIe()
    print(f"sm_count={a.sm_count}  freq={a.max_freq/1e9:.2f}GHz")
    print(f"fp16_tensor_TFLOPS={a.fp16_tensor_flops/1e12:.2f}")
    print(f"fp8_tensor_TFLOPS={a.fp8_tensor_flops/1e12:.2f}")
    print(f"hbm_BW={a.ddr_bandwidth/1e9:.0f} GB/s, hbm_cap={a.ddr_capacity/1024**3:.0f} GiB")
    print(f"l2_BW={a.l2_bandwidth/1e9:.0f} GB/s, l2_cap={a.l2_capacity/1024**2:.0f} MiB")
    print(f"pcie_uni={a.pcie_unidirectional_bw/1024**3:.0f} GiB/s ({a.interconnect_type})")
