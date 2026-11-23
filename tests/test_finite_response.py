"""Invariants for the donor-secanted finite-response estimator."""
import unittest

import torch

from ilslens.finite_response import secant_transport


class FiniteResponseTests(unittest.TestCase):
    def test_clean_and_full_donor_endpoints(self):
        mid = torch.tensor([[1., 2.], [3., -1.]])
        end = torch.tensor([[4., -3.], [-1., 1.]])
        for gain in (0., 0.7, 2.):
            torch.testing.assert_close(secant_transport(mid, mid, end, gain), end)
            torch.testing.assert_close(secant_transport(mid * 0, mid, end, gain), end * 0)


    def test_projection_and_degenerate_donor_are_position_specific(self):
        p = torch.tensor([[1., 3.], [2., -2.]])
        mid = torch.tensor([[2., 0.], [0., 0.]])
        end = torch.tensor([[0., 4.], [0., 0.]])
        torch.testing.assert_close(secant_transport(p, mid, end, .5), torch.tensor([[0., 5.], [1., -1.]]))
