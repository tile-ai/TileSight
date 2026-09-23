"""Unified TileSight interface for TileLang-style tile programs.

Recommended usage::

    from tilesight.modeling import program as sight

    result = sight.analyze(program, arch, options=sight.Options(ii_mode="resource_ii", cache="fast"))
    op = result.op("main/kv/load_K")
    print(op.count, op.traffic_total.ddr_read_bytes)
"""

from .analysis import UnsupportedError, analyze
from .builder import BufferRef, KernelBuilder, TensorRef, TensorSlice, sequential_program, tiles
from .contract import (
    BlockMapper, CutlassSm90, ExplicitOrder, LinearBlockId, PanelSwizzle, register_block_mapper,
    Actor, Buffer, BufferSlots, Carry, Compute, ContractError, Dependency, EventRef,
    FromWorkGroup, Launch, LaunchEdge, Load, Loop, Persistent, Pipeline, Program,
    Projection, ResourceOrder, Sequence, Store, Sync, TileAccess, WorkAxis, WorkGroup,
    WorkRun, WorkUnits, done, start,
)
from .options import Context, LatencyEntry, LatencyTable, Options, TrafficAssumption
from .results import (
    AnalysisResult, CacheAccessReport, CacheReport, CompletionInterval, CostGroup,
    CostGroupKey, Diagnostics, IIComponents, LaunchResult, OpResult, ProgramResult,
    RegionGroupResult, RegionResult, ServiceInterval, TrafficInterval,
    WorkGroupSummary,
)
from .snapshot import program_from_snapshot, program_to_snapshot
from .traffic import Traffic, TrafficShare
from .validation import (
    IDENTITY_FIELDS, Measurement, ValidationCase, ValidationReport, load_baseline_predictions, load_measurements,
    parse_shape_id, validate,
)
from .compare import ComparisonResult, Observation, compare
from .perfetto import export_perfetto, perfetto_trace

__all__ = [
    "BufferRef", "KernelBuilder", "Measurement", "TensorRef", "TensorSlice", "ValidationCase", "ValidationReport",
    "BlockMapper", "CutlassSm90", "ExplicitOrder", "LinearBlockId", "PanelSwizzle", "register_block_mapper",
    "IDENTITY_FIELDS", "load_baseline_predictions", "load_measurements", "parse_shape_id", "sequential_program", "tiles", "validate",
    "Actor", "AnalysisResult", "Buffer", "BufferSlots", "CacheAccessReport", "CacheReport",
    "Carry", "ComparisonResult", "CompletionInterval", "Compute", "Context",
    "ContractError", "CostGroup", "CostGroupKey", "Dependency", "Diagnostics", "EventRef",
    "FromWorkGroup", "IIComponents", "LatencyEntry", "LatencyTable", "Launch",
    "LaunchEdge", "LaunchResult", "Load", "Loop", "Observation", "OpResult", "Options",
    "Persistent", "Pipeline", "Program", "ProgramResult", "Projection",
    "RegionGroupResult", "RegionResult", "ResourceOrder", "Sequence", "ServiceInterval",
    "Store", "Sync", "TileAccess", "Traffic", "TrafficAssumption", "TrafficInterval", "TrafficShare",
    "UnsupportedError", "WorkAxis", "WorkGroup", "WorkGroupSummary", "WorkRun",
    "WorkUnits", "analyze", "compare", "done", "export_perfetto", "perfetto_trace",
    "program_from_snapshot", "program_to_snapshot", "start",
]
