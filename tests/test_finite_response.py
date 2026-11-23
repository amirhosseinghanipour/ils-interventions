"""Invariants for the donor-secanted finite-response estimator."""

import unittest

import torch

from ilslens.finite_response import secant_transport


class FiniteResponseTests(unittest.TestCase):
    def test_clean_and_full_donor_endpoints(self):
        mid = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        end = torch.tensor([[4.0, -3.0], [-1.0, 1.0]])
        for gain in (0.0, 0.7, 2.0):
            torch.testing.assert_close(secant_transport(mid, mid, end, gain), end)
            torch.testing.assert_close(secant_transport(mid * 0, mid, end, gain), end * 0)

    def test_projection_and_degenerate_donor_are_position_specific(self):
        p = torch.tensor([[1.0, 3.0], [2.0, -2.0]])
        mid = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        end = torch.tensor([[0.0, 4.0], [0.0, 0.0]])
        torch.testing.assert_close(
            secant_transport(p, mid, end, 0.5), torch.tensor([[0.0, 5.0], [1.0, -1.0]])
        )

    def test_prefix_estimator_reproduces_anchor_and_excludes_future_states(self):
        from ilslens.finite_response import prefix_secants

        torch.manual_seed(712)
        z, y = torch.randn(6, 7), torch.randn(6, 7)
        offset, gain, ridge = prefix_secants(z[3:], z, y, 3)
        torch.testing.assert_close(offset + 0.7 * gain, y[3:])
        patch = torch.randn(1, 7)
        for full, prefix in zip(prefix_secants(patch, z, y, 3), prefix_secants(patch, z[:4], y[:4], 3)):
            torch.testing.assert_close(full, prefix)
