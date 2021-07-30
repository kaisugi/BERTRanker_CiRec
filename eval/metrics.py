import numpy as np
import unittest

def recall(r, k=10):
    if 1 in r[:k]:
        return 1.0
    else:
        return 0.0

def reciprocal_rank(r, k=10):
    for i in range(k):
        if r[i] == 1:
            return (1/(i+1))
    
    return 0.0


class TestStringMethods(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(
            recall([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
            1.0
        )

        self.assertEqual(
            recall([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            0.0
        )

    def test_reciprocal_rank(self):
        self.assertEqual(
            reciprocal_rank([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
            1.0
        )

        self.assertEqual(
            reciprocal_rank([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
            0.25
        )

        self.assertEqual(
            reciprocal_rank([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            0.0
        )
 


if __name__ == "__main__":
    unittest.main()