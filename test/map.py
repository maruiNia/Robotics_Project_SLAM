import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가

sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from module.MySlam import SlamMap, Obstacle_obj

# 같은 파일에 클래스가 있으면 아래 import는 필요 없고,
# 다른 파일(slammap.py)에 있으면 아래처럼 사용하세요.
# from slammap import SlamMap, Obstacle_obj


class TestObstacleObj(unittest.TestCase):
    def test_create_with_max_corner(self):
        obs = Obstacle_obj(dim=2, min_corner=(1, 2), max_corner=(3, 5))
        self.assertEqual(obs.min_corner, (1, 2))
        self.assertEqual(obs.max_corner, (3, 5))

    def test_create_with_size(self):
        obs = Obstacle_obj(dim=2, min_corner=(1, 2), size=(2, 3))
        self.assertEqual(obs.min_corner, (1, 2))
        self.assertEqual(obs.max_corner, (3, 5))

    def test_auto_fix_swapped_corners(self):
        obs = Obstacle_obj(dim=2, min_corner=(5, 1), max_corner=(2, 4))
        # 자동으로 min/max 정렬 저장
        self.assertEqual(obs.min_corner, (2, 1))
        self.assertEqual(obs.max_corner, (5, 4))

    def test_contains_inclusive(self):
        obs = Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(10, 10))
        self.assertTrue(obs.contains((0, 0)))      # 경계 포함
        self.assertTrue(obs.contains((10, 10)))    # 경계 포함
        self.assertTrue(obs.contains((5, 5)))
        self.assertFalse(obs.contains((-1, 0)))
        self.assertFalse(obs.contains((11, 5)))

    def test_contains_exclusive(self):
        obs = Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(10, 10))
        self.assertFalse(obs.contains((0, 0), inclusive=False))     # 경계 제외
        self.assertFalse(obs.contains((10, 10), inclusive=False))   # 경계 제외
        self.assertTrue(obs.contains((5, 5), inclusive=False))

    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            Obstacle_obj(dim=0, min_corner=(0,), max_corner=(1,))

    def test_invalid_argument_rule(self):
        # max_corner와 size 둘 다 주면 에러
        with self.assertRaises(ValueError):
            Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(1, 1), size=(1, 1))

        # 둘 다 안 줘도 에러
        with self.assertRaises(ValueError):
            Obstacle_obj(dim=2, min_corner=(0, 0))

    def test_dim_mismatch_point(self):
        obs = Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(1, 1))
        with self.assertRaises(ValueError):
            obs.contains((0, 0, 0))  # dim 불일치


class TestSlamMap(unittest.TestCase):
    def test_add_and_get_list(self):
        m = SlamMap(dim=2)
        obs = Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(1, 1))
        m.add(obs)

        lst = m.get_obs_list()
        self.assertEqual(len(lst), 1)
        self.assertIs(lst[0], obs)

    def test_add_dim_mismatch(self):
        m = SlamMap(dim=2)
        obs3 = Obstacle_obj(dim=3, min_corner=(0, 0, 0), max_corner=(1, 1, 1))
        with self.assertRaises(ValueError):
            m.add(obs3)

    def test_is_wall(self):
        m = SlamMap(dim=2)
        obs = Obstacle_obj(dim=2, min_corner=(2, 2), max_corner=(4, 4))
        m.add(obs)

        self.assertTrue(m.is_wall((2, 2)))   # 경계 포함
        self.assertTrue(m.is_wall((3, 3)))
        self.assertFalse(m.is_wall((1, 1)))
        self.assertFalse(m.is_wall((5, 5)))

    def test_is_wall_exclusive(self):
        m = SlamMap(dim=2)
        obs = Obstacle_obj(dim=2, min_corner=(2, 2), max_corner=(4, 4))
        m.add(obs)

        self.assertFalse(m.is_wall((2, 2), inclusive=False))  # 경계 제외
        self.assertTrue(m.is_wall((3, 3), inclusive=False))

    def test_get_map_summary(self):
        m = SlamMap(dim=2, limit=((10, 10), (0, 0)))  # 뒤집혀 들어와도 자동 정렬
        obs1 = Obstacle_obj(dim=2, min_corner=(2, 2), max_corner=(4, 5))
        obs2 = Obstacle_obj(dim=2, min_corner=(6, 1), size=(2, 3))  # (6,1)~(8,4)
        m.add(obs1)
        m.add(obs2)

        info = m.get_map()
        self.assertEqual(info["dim"], 2)
        self.assertEqual(info["limit"], ((0, 0), (10, 10)))
        self.assertEqual(info["obstacle_count"], 2)
        self.assertEqual(info["obstacles_bounds"], ((2, 1), (8, 5)))

    def test_set_dim_blocked_when_obstacles_exist(self):
        m = SlamMap(dim=2)
        m.add(Obstacle_obj(dim=2, min_corner=(0, 0), max_corner=(1, 1)))
        with self.assertRaises(RuntimeError):
            m.set_dim(3)


if __name__ == "__main__":
    unittest.main()
