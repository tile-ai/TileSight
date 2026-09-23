"""Lazy logical work-grid traversal.

Unlike the historical helper, this module does not allocate an
``O(total_tiles)`` coordinate array.  Wave boundaries and coordinate order are
nevertheless byte-for-byte compatible with the legacy panel traversal.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator, Sequence, Tuple

from ..errors import ModelingValidationError
from .ir import ExplicitTraversal, PanelTraversal, RowMajorTraversal, TileGrid, TraversalIR


Coordinate = Tuple[int, ...]


def _panel_coordinates(grid: TileGrid, traversal: PanelTraversal) -> Iterator[Coordinate]:
    grid_m, grid_n = grid.shape
    stride_m = traversal.stride_m
    stride_n = traversal.row_panel
    axis = traversal.raster_axis

    m_start = 0
    n_start = 0
    while m_start < grid_m and n_start < grid_n:
        m_end = min(m_start + stride_m, grid_m)
        n_end = min(n_start + stride_n, grid_n)
        if axis in ("legacy", "along_n"):
            for m in range(m_start, m_end):
                for n in range(n_start, n_end):
                    yield (m, n)
        else:
            for n in range(n_start, n_end):
                for m in range(m_start, m_end):
                    yield (m, n)

        if axis in ("legacy", "along_m"):
            m_start = m_end
            if m_start >= grid_m:
                m_start = 0
                n_start = n_end
        else:
            n_start = n_end
            if n_start >= grid_n:
                n_start = 0
                m_start = m_end


def _row_major_coordinates(shape: Sequence[int]) -> Iterable[Coordinate]:
    return itertools.product(*(range(item) for item in shape))


def iter_coordinates(grid: TileGrid, traversal: TraversalIR) -> Iterable[Coordinate]:
    if isinstance(traversal, PanelTraversal):
        return _panel_coordinates(grid, traversal)
    if isinstance(traversal, RowMajorTraversal):
        return _row_major_coordinates(grid.shape)
    if isinstance(traversal, ExplicitTraversal):
        return iter(traversal.coordinates)
    raise ModelingValidationError("unsupported cache traversal")


def traversal_wave_size(traversal: TraversalIR) -> int:
    if isinstance(traversal, PanelTraversal):
        return traversal.wave_size
    if isinstance(traversal, (RowMajorTraversal, ExplicitTraversal)):
        return traversal.wave_size
    raise ModelingValidationError("unsupported cache traversal")


def traversal_sm_count(traversal: TraversalIR) -> int:
    if isinstance(traversal, (PanelTraversal, RowMajorTraversal, ExplicitTraversal)):
        return traversal.sm_count
    raise ModelingValidationError("unsupported cache traversal")


def iter_waves(grid: TileGrid, traversal: TraversalIR) -> Iterator[Tuple[Coordinate, ...]]:
    coordinates = iter(iter_coordinates(grid, traversal))
    wave_size = traversal_wave_size(traversal)
    while True:
        wave = tuple(itertools.islice(coordinates, wave_size))
        if not wave:
            return
        yield wave

