import unittest
import torch
import src.input_generation as ig


class TestGetInputList(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestGetInputList, self).__init__(*args, **kwargs)
        self.A = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        self.A_sym = ig.get_symmetric_adjacency(self.A, self_loop=False)
        self.A_sym_tilde = ig.get_symmetric_adjacency(self.A, self_loop=True)
        self.k = 3
        # Create a sample input matrix X
        self.X = torch.eye(self.A.shape[0], dtype=torch.float32)
        self.input_list = ig.get_input_list(
            self.A_sym, self.A_sym_tilde, self.X, self.k
        )

    def test_length_of_input_list(self):
        # Check if the length of the input list is 2k + 1
        self.assertEqual(len(self.input_list), 2 * self.k + 1)

    def test_powers_of_input(self):
        X1 = self.X
        X2 = self.A_sym @ self.X
        X3 = self.A_sym_tilde @ self.X
        X4 = torch.linalg.matrix_power(self.A_sym, 2) @ self.X
        X5 = torch.linalg.matrix_power(self.A_sym_tilde, 2) @ self.X
        X6 = torch.linalg.matrix_power(self.A_sym, 3) @ self.X
        X7 = torch.linalg.matrix_power(self.A_sym_tilde, 3) @ self.X

        k = 3
        expected_results = [X1, X2, X3, X4, X5, X6, X7]

        for i in range(2 * k + 1):
            if not torch.allclose(self.input_list[i], expected_results[i]):
                print(self.input_list[i], "\n\n", expected_results[i])
            assert torch.allclose(self.input_list[i], expected_results[i])


if __name__ == "__main__":
    unittest.main()
