# TileLang examples

Five kernels for the TileLang frontend ([`tilesight.tilelang`](../../tilesight/tilelang/README.md)).
`gemm`, `gqa`, `gelu` and `sum` are self-contained copies of TileOps kernels
([LICENSE.TileOps](LICENSE.TileOps)); `gemm_pipelined` is the same GEMM written
with `T.Pipelined`, `T.copy` and `T.gemm`. `cases.json` fixes their shapes and
compile options.

| Example | What it shows |
|---|---|
| `gemm` | hand-written warp-specialized GEMM (TMA producer, WGMMA consumer), 8192³ FP16 |
| `gemm_pipelined` | the same GEMM; the compiler adds the producer warpgroup and TMA |
| `gqa` | warp-specialized grouped-query attention, two consumers synchronized by named barriers |
| `gelu` | element-wise BF16 |
| `sum` | row reduction |

```bash
python examples/tilelang/run.py all --output-dir reports/model
python examples/tilelang/run.py gemm --mode all --device-uuid GPU-... --ncu /path/to/ncu --output-dir reports/gemm
```
