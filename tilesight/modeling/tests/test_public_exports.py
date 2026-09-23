"""Stable top-level exports for the experimental unified API."""


def test_semantic_native_and_resident_exports_are_public():
    import tilesight.modeling as api

    expected = (
        "GeneralGemmOptions",
        "LaunchTopology",
        "LegacyStageBoundary",
        "ResidentScheduleConfig",
        "ResourceScope",
        "SmallTileLatencyConfig",
        "make_elementwise",
        "make_gemm",
        "make_reduce",
        "model_general_gemm",
        "schedule_resident_ctas",
    )
    assert all(hasattr(api, name) for name in expected)


def test_general_gemm_adapter_exports_are_public():
    from tilesight.modeling import adapters

    expected = (
        "BoundGemmCostOracle",
        "GeneralGemmOptions",
        "GeneralGemmResult",
        "bind_general_gemm",
        "gemm_legacy_arch_view",
        "model_general_gemm",
    )
    assert all(hasattr(adapters, name) for name in expected)
