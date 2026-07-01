import unittest

class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, initial_value: float) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        x = initial_value
        for _ in range(iterations):
            x = x - learning_rate * (2 * x)
        return round(x, 5)

class TestGradientDescent(unittest.TestCase):
    def setUp(self):
        # Initialize the solution object before each test
        self.solution = Solution()

    def test_zero_iterations(self):
        # If no iterations run, the value should remain unchanged.
        self.assertEqual(self.solution.get_minimizer(0, 0.1, 5.0), 5.0)

    def test_single_step(self):
        # Math: x = 5.0 - 0.1 * (2 * 5.0) = 4.0
        self.assertEqual(self.solution.get_minimizer(1, 0.1, 5.0), 4.0)

    def test_positive_convergence(self):
        # With sufficient iterations, it should converge to the minimum (0.0).
        self.assertEqual(self.solution.get_minimizer(100, 0.1, 10.0), 0.0)

    def test_negative_convergence(self):
        # Gradient descent should also find the minimum from a negative starting point.
        self.assertEqual(self.solution.get_minimizer(100, 0.1, -10.0), 0.0)

    def test_overshoot_oscillation(self):
        # A large learning rate (e.g., 0.6) causes the value to oscillate across 0.
        # Step 1: 1.0 - 0.6(2.0) = -0.2
        # Step 2: -0.2 - 0.6(-0.4) = 0.04
        self.assertEqual(self.solution.get_minimizer(2, 0.6, 1.0), 0.04)

if __name__ == '__main__':
    unittest.main()