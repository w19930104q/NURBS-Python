"""Tests for the NURBS-Python package.

Requires "pytest" to run.
"""

import pytest

from geomdl import BSpline
from geomdl import NURBS
from geomdl import operations


SAMPLE_SIZE = 5
C_DEGREE = 2
C_CTRLPTS2D = [[1, 1], [2, 1], [2, 2]]
C_CTRLPTS3D = [[1, 1, 0], [2, 1, -1], [2, 2, 0]]
C_KV = [0, 0, 0, 1, 1, 1]

S_DEGREE_U = 2
S_DEGREE_V = 2
S_CTRLPTS = [[0, 0, 0], [0, 1, 0], [0, 2, -3], [1, 0, 6], [1, 1, 0], [1, 2, 0], [2, 0, 0], [2, 1, 0], [2, 2, 3]]
S_KV_U = [0, 0, 0, 1, 1, 1]
S_KV_V = [0, 0, 0, 1, 1, 1]

# Fixed degenerate surface: the u=0 edge is collapsed to the apex point (0, 0, 2)
D_DEGREE_U = 2
D_DEGREE_V = 2
D_CTRLPTS = [
    [0.0, 0.0, 2.0],
    [0.0, 0.0, 2.0],
    [0.0, 0.0, 2.0],  # u=0: degenerate edge (apex)
    [-1.0, 0.0, 1.0],
    [0.0, 0.5, 1.0],
    [1.0, 0.0, 1.0],
    [-2.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [2.0, 0.0, 0.0],
]
D_KV_U = [0, 0, 0, 1, 1, 1]
D_KV_V = [0, 0, 0, 1, 1, 1]

GEOMDL_DELTA = 1e-6


def test_bspline_curve2d_evaluate():
    curve = BSpline.Curve()
    curve.degree = C_DEGREE
    curve.ctrlpts = C_CTRLPTS2D
    curve.knotvector = C_KV
    curve.sample_size = SAMPLE_SIZE

    # Expected output
    res = [[1.0, 1.0], [1.4375, 1.0625], [1.75, 1.25], [1.9375, 1.5625], [2.0, 2.0]]

    assert curve.evalpts == res


def test_bspline_curve3d_evaluate():
    curve = BSpline.Curve()
    curve.degree = C_DEGREE
    curve.ctrlpts = C_CTRLPTS3D
    curve.knotvector = C_KV
    curve.sample_size = SAMPLE_SIZE

    # Expected output
    res = [[1.0, 1.0, 0.0], [1.4375, 1.0625, -0.375], [1.75, 1.25, -0.5], [1.9375, 1.5625, -0.375], [2.0, 2.0, 0.0]]

    assert curve.evalpts == res


def test_bspline_surface_evaluate():
    surf = BSpline.Surface()
    surf.degree_u = S_DEGREE_U
    surf.degree_v = S_DEGREE_V
    surf.set_ctrlpts(S_CTRLPTS, 3, 3)
    surf.knotvector_u = S_KV_U
    surf.knotvector_v = S_KV_V
    surf.sample_size = SAMPLE_SIZE

    # Expected output
    res = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, -0.1875],
        [0.0, 1.0, -0.75],
        [0.0, 1.5, -1.6875],
        [0.0, 2.0, -3.0],
        [0.5, 0.0, 2.25],
        [0.5, 0.5, 1.171875],
        [0.5, 1.0, 0.1875],
        [0.5, 1.5, -0.703125],
        [0.5, 2.0, -1.5],
        [1.0, 0.0, 3.0],
        [1.0, 0.5, 1.6875],
        [1.0, 1.0, 0.75],
        [1.0, 1.5, 0.1875],
        [1.0, 2.0, 0.0],
        [1.5, 0.0, 2.25],
        [1.5, 0.5, 1.359375],
        [1.5, 1.0, 0.9375],
        [1.5, 1.5, 0.984375],
        [1.5, 2.0, 1.5],
        [2.0, 0.0, 0.0],
        [2.0, 0.5, 0.1875],
        [2.0, 1.0, 0.75],
        [2.0, 1.5, 1.6875],
        [2.0, 2.0, 3.0],
    ]

    assert surf.evalpts == res


def _degenerate_bspline_surf():
    """Creates the fixed degenerate B-spline surface (collapsed u=0 edge)."""
    surf = BSpline.Surface()
    surf.degree_u = D_DEGREE_U
    surf.degree_v = D_DEGREE_V
    surf.set_ctrlpts(D_CTRLPTS, 3, 3)
    surf.knotvector_u = D_KV_U
    surf.knotvector_v = D_KV_V
    return surf


def _degenerate_nurbs_surf():
    """Creates the fixed degenerate NURBS surface (collapsed u=1 edge)."""
    surf = NURBS.Surface()
    surf.degree_u = D_DEGREE_U
    surf.degree_v = D_DEGREE_V
    surf.ctrlpts_size_u = 3
    surf.ctrlpts_size_v = 3
    surf.ctrlpts = list(reversed(D_CTRLPTS))
    surf.weights = [1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0]
    surf.knotvector_u = D_KV_U
    surf.knotvector_v = D_KV_V
    return surf


def _dot(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))


def test_bspline_surface_degenerate_edge_evaluate_single():
    """Surface evaluation on the degenerate edge and at the corners of the parametric domain."""
    surf = _degenerate_bspline_surf()

    # The degenerate edge collapses to the apex point
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert surf.evaluate_single([0.0, v]) == pytest.approx([0.0, 0.0, 2.0])

    # Corners of the parametric domain evaluate to the corner control points
    assert surf.evaluate_single([0.0, 0.0]) == pytest.approx([0.0, 0.0, 2.0])
    assert surf.evaluate_single([0.0, 1.0]) == pytest.approx([0.0, 0.0, 2.0])
    assert surf.evaluate_single([1.0, 0.0]) == pytest.approx([-2.0, 0.0, 0.0])
    assert surf.evaluate_single([1.0, 1.0]) == pytest.approx([2.0, 0.0, 0.0])


def test_bspline_surface_degenerate_edge_corner_continuity():
    """Evaluated corner values are continuous with the neighboring evaluated points."""
    surf = _degenerate_bspline_surf()
    eps = 1e-7

    corners = [
        ((0.0, 0.0), [(eps, 0.0), (0.0, eps)]),
        ((0.0, 1.0), [(eps, 1.0), (0.0, 1.0 - eps)]),
        ((1.0, 0.0), [(1.0 - eps, 0.0), (1.0, eps)]),
        ((1.0, 1.0), [(1.0 - eps, 1.0), (1.0, 1.0 - eps)]),
    ]
    for corner, neighbors in corners:
        pt_corner = surf.evaluate_single(list(corner))
        for neighbor in neighbors:
            pt_neighbor = surf.evaluate_single(list(neighbor))
            dist = max(abs(a - b) for a, b in zip(pt_corner, pt_neighbor))
            assert dist < GEOMDL_DELTA


def test_bspline_surface_degenerate_edge_normal():
    """Surface normal on the degenerate edge is consistent with the limit from the interior."""
    surf = _degenerate_bspline_surf()

    # Expected unit normals on the degenerate edge (limit values)
    res = {
        0.0: (0.4082482904638631, -0.8164965809277261, -0.4082482904638631),
        0.5: (0.0, -0.9701425001453319, -0.24253562503633297),
        1.0: (-0.4082482904638631, -0.8164965809277261, -0.4082482904638631),
    }
    for v, expected in res.items():
        pt, normal = operations.normal(surf, [0.0, v])
        # The surface point lies on the degenerate edge
        assert list(pt) == pytest.approx([0.0, 0.0, 2.0])
        # The normal is a unit vector consistent with the limit value
        assert _dot(normal, normal) == pytest.approx(1.0)
        assert list(normal) == pytest.approx(list(expected))

    # The normal on the degenerate edge matches the limit of the interior normals
    eps = 1e-6
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, normal_edge = operations.normal(surf, [0.0, v])
        _, normal_limit = operations.normal(surf, [eps, v])
        assert _dot(normal_edge, normal_limit) == pytest.approx(1.0, abs=1e-4)


def test_bspline_surface_degenerate_edge_tangent():
    """Tangent plane on the degenerate edge is consistent with the surface normal."""
    surf = _degenerate_bspline_surf()

    for v in (0.0, 0.5, 1.0):
        pt, tangent_u, tangent_v = operations.tangent(surf, [0.0, v])
        pt_n, normal = operations.normal(surf, [0.0, v])

        # Tangent and normal evaluations return the same surface point
        assert list(pt) == pytest.approx(list(pt_n))

        # The normal is perpendicular to both tangent directions
        assert _dot(normal, tangent_u) == pytest.approx(0.0, abs=GEOMDL_DELTA)
        assert _dot(normal, tangent_v) == pytest.approx(0.0, abs=GEOMDL_DELTA)


def test_nurbs_surface_degenerate_edge_normal():
    """Normal on the degenerate edge of a rational surface is consistent with the interior limit."""
    surf = _degenerate_nurbs_surf()

    # The degenerate edge (u=1) collapses to the apex point
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        pt, normal = operations.normal(surf, [1.0, v])
        assert list(pt) == pytest.approx([0.0, 0.0, 2.0])
        assert _dot(normal, normal) == pytest.approx(1.0)

    # The normal on the degenerate edge matches the limit of the interior normals
    eps = 1e-6
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, normal_edge = operations.normal(surf, [1.0, v])
        _, normal_limit = operations.normal(surf, [1.0 - eps, v])
        assert _dot(normal_edge, normal_limit) == pytest.approx(1.0, abs=1e-4)


def test_bspline_surface_regular_normal_unchanged():
    """Normal evaluation on a regular surface is not affected by the degenerate edge handling."""
    surf = BSpline.Surface()
    surf.degree_u = S_DEGREE_U
    surf.degree_v = S_DEGREE_V
    surf.set_ctrlpts(S_CTRLPTS, 3, 3)
    surf.knotvector_u = S_KV_U
    surf.knotvector_v = S_KV_V

    for u, v in ((0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0)):
        skl = surf.derivatives(u, v, 1)
        der_u, der_v = skl[1][0], skl[0][1]
        expected = [
            der_u[1] * der_v[2] - der_u[2] * der_v[1],
            der_u[2] * der_v[0] - der_u[0] * der_v[2],
            der_u[0] * der_v[1] - der_u[1] * der_v[0],
        ]
        mag = sum(c**2 for c in expected) ** 0.5
        expected = [c / mag for c in expected]

        pt, normal = operations.normal(surf, [u, v])
        assert list(pt) == pytest.approx(list(skl[0][0]))
        assert list(normal) == pytest.approx(expected)
