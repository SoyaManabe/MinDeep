import unittest
import numpy as np
import mindeep

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = mindeep.Variable(np.array([2.0]))
        y = mindeep.square(x)
        expected = np.array([4.0])
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = mindeep.Variable(np.array([3.0]))
        y = mindeep.square(x)
        y.backward()
        expected = np.array([6.0])
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = mindeep.Variable(np.random.rand(1))
        y = mindeep.square(x)
        y.backward()
        num_grad = mindeep.numerical_diff(mindeep.square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
