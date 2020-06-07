import main
import unittest
import numpy as np

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = main.Variable(np.array([2.0]))
        y = main.square(x)
        expected = np.array([4.0])
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = main.Variable(np.array([3.0]))
        y = main.square(x)
        y.backward()
        expected = np.array([6.0])
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = main.Variable(np.random.rand(1))
        y = main.square(x)
        y.backward()
        num_grad = main.numerical_diff(main.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
