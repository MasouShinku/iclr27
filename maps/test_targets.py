"""Run: python -m unittest test_targets -v"""
import unittest
from unittest.mock import patch
import numpy as np
from config import Config
from dataset import sequential_target_meshes


class SequentialTargetsTest(unittest.TestCase):
    def test_requeries_on_actual_current_mesh(self):
        initial, middle1, middle2, expert = (object() for _ in range(4))
        geometry = object()
        with patch('dataset.get_sizing_field', side_effect=[np.array([9.]), np.array([7.])]) as sizes, \
             patch('dataset.project_sizing_field', side_effect=[np.array([3.]), np.array([1.])]) as project, \
             patch('dataset.update_mesh', side_effect=[middle1, middle2]) as remesh:
            result = sequential_target_meshes(initial, expert, geometry)
        self.assertEqual(result, [initial, middle1, middle2, expert])
        self.assertIs(sizes.call_args_list[1].args[0], middle1)
        self.assertIs(project.call_args_list[1].args[1], middle1)
        np.testing.assert_allclose(remesh.call_args_list[0].args[1], [7.])
        np.testing.assert_allclose(remesh.call_args_list[1].args[1], [4.])

    def test_default_three_transitions(self):
        self.assertEqual(Config().trajectory_levels, [0, 1, 2, 3])
        self.assertEqual(Config().resolved_inference_steps(), 3)


if __name__ == '__main__':
    unittest.main()
