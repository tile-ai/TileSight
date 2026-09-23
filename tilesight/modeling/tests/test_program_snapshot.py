"""Versioned JSON snapshots round-trip every typed example."""

import json

import pytest

from tilesight.modeling import program as sight
from tilesight.modeling.program import examples as ex
from tilesight.modeling.ir import ResourceTiming, Timing
from tilesight.modeling.ops.schema import GemmOpSpec


@pytest.mark.parametrize("builder", [
    lambda: ex.gemm_program(m=512, n=512, k=512),
    lambda: ex.elementwise_add_program(m=1000, n=512),
    lambda: ex.reduce_sum_program(rows=256),
    lambda: ex.rms_norm_program(rows=256),
    lambda: ex.fa3_program(seq_len=512, heads=2),
    lambda: ex.fa3_program(seq_len=512, heads=2, causal=True),
    lambda: ex.flashmla_decode_program(batch=2, kv_len=512),
    lambda: ex.flashmla_decode_program(batch=2, kv_len=512, num_splits=2),
])
def test_round_trip(builder):
    program = builder()
    snapshot = sight.program_to_snapshot(program)
    assert snapshot["schema"] == "tilesight.program/1"
    text = json.dumps(snapshot, sort_keys=True)
    restored = sight.program_from_snapshot(json.loads(text))
    assert restored == program
    assert sight.program_to_snapshot(restored) == snapshot


def test_round_trip_keeps_timing_override_and_persistent_scheduler():
    op = sight.Compute(
        "c", GemmOpSpec(m=8, n=8, k=8, compute_dtype="fp16"), actor="t",
        timing=Timing(10e-9, (ResourceTiming("tensor", 8e-9, 1e-9),)), latency_extra_s=1e-9,
        metadata={"source_line": 12},
    )
    launch = sight.Launch("main", op, (sight.WorkAxis("m", 2),), scheduler=sight.Persistent((((0, 1),), ((1, 2),))),
                       residency=2, grid=(2,))
    program = sight.Program((launch,), name="p", metadata={"origin": "test"})
    restored = sight.program_from_snapshot(sight.program_to_snapshot(program))
    assert restored == program
    assert restored.launches[0].scheduler.cta_count == 2


def test_schema_version_is_checked():
    with pytest.raises(sight.ContractError, match="schema"):
        sight.program_from_snapshot({"schema": "other", "launches": []})
