import numpy as np

from benchmark.shared_warehouse_perception import (
    segment_intersects_rect,
    visible_with_shelves,
)
from benchmark.train_multi_agent_research import SHELVES


def test_segment_rectangle_intersection_detects_blocked_line_of_sight():
    rect = (-8.0, 2.0, -6.0, 8.0)
    assert segment_intersects_rect(np.array([-9.0, 4.0]), np.array([-5.0, 4.0]), rect)
    assert not segment_intersects_rect(np.array([-9.0, 0.0]), np.array([-5.0, 0.0]), rect)


def test_shelf_blocks_human_visibility_but_open_corridor_does_not():
    observer = np.array([-9.0, 4.0], dtype=np.float32)
    hidden = np.array([-5.0, 4.0], dtype=np.float32)
    assert not visible_with_shelves(observer, hidden, SHELVES, max_range=10.0)

    observer = np.array([-9.0, 0.0], dtype=np.float32)
    visible = np.array([-5.0, 0.0], dtype=np.float32)
    assert visible_with_shelves(observer, visible, SHELVES, max_range=10.0)


def test_range_limit_is_part_of_same_visibility_contract():
    assert not visible_with_shelves(
        np.array([0.0, -1.5]), np.array([0.0, -8.0]), SHELVES, max_range=5.0
    )
