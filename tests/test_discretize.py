"""Tests for adaptive curve discretization (geomdl.discretize module).

The tests use a fixed family of B-spline and NURBS curves with deterministic control points and knot vectors,
so that every assertion is fully reproducible. Requires "pytest" to run.
"""

from pytest import fixture, mark, raises

from geomdl import NURBS, BSpline, convert, discretize, operations
from geomdl.exceptions import GeomdlException

GEOMDL_DELTA = 10e-6
TOLERANCE = 10e-4


@fixture
def mixed_curve():
    """Cubic B-spline with a long straight section and a sharply bending section."""
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [[0, 0], [2, 0], [4, 0], [5, 0.2], [5.2, 1], [5.2, 3], [5, 5]]
    curve.knotvector = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
    return curve


@fixture
def quarter_arc():
    """Cubic Bezier approximation of a quarter circle with radius 5."""
    kappa = 0.5522847498
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [[5, 0], [5, 5 * kappa], [5 * kappa, 5], [0, 5]]
    curve.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]
    return curve


@fixture
def straight_line():
    """Degree-1 straight line spanning 10 units with zero curvature."""
    curve = BSpline.Curve()
    curve.degree = 1
    curve.ctrlpts = [[0, 0], [10, 0]]
    curve.knotvector = [0, 0, 1, 1]
    return curve


@fixture
def nurbs_mixed(mixed_curve):
    """Rational version of the mixed curve with non-uniform weights."""
    curve = convert.bspline_to_nurbs(mixed_curve)
    curve.weights = [1.0, 1.0, 1.0, 0.5, 0.8, 1.0, 1.0]
    return curve


# ---------------------------------------------------------------------------
# Structure and reproducibility
# ---------------------------------------------------------------------------


def test_adaptive_returns_endpoints(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    assert result.parameters[0] == 0.0
    assert result.parameters[-1] == 1.0
    assert result.points[0] == mixed_curve.evaluate_single(0.0)
    assert result.points[-1] == mixed_curve.evaluate_single(1.0)
    assert len(result.parameters) == len(result.points)


def test_adaptive_parameters_strictly_increasing(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    for idx in range(1, len(result.parameters)):
        assert result.parameters[idx] > result.parameters[idx - 1]


def test_adaptive_is_deterministic(mixed_curve):
    first = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)
    second = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    assert first.parameters == second.parameters
    assert first.points == second.points
    assert [seg["sagitta"] for seg in first.segments] == [seg["sagitta"] for seg in second.segments]


def test_adaptive_does_not_mutate_curve(mixed_curve):
    delta_before = mixed_curve.delta
    points_before = list(mixed_curve.evalpts)

    discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    assert mixed_curve.delta == delta_before
    assert mixed_curve.evalpts == points_before


def test_adaptive_partial_interval(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, start=0.2, stop=0.8, tolerance=TOLERANCE)

    assert abs(result.parameters[0] - 0.2) < GEOMDL_DELTA
    assert abs(result.parameters[-1] - 0.8) < GEOMDL_DELTA
    assert all(0.2 <= u <= 0.8 for u in result.parameters)


# ---------------------------------------------------------------------------
# Sampling criterion: curvature and arc length adaptivity
# ---------------------------------------------------------------------------


def test_high_curvature_section_is_denser(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    flat = [u for u in result.parameters if u < 0.5]
    bend = [u for u in result.parameters if u >= 0.5]

    # The bend occupies the same parametric extent as the flat section but must contain more samples
    assert len(bend) > len(flat)


def test_tolerance_controls_density(quarter_arc):
    coarse = discretize.adaptive_curve(quarter_arc, tolerance=10e-3)
    fine = discretize.adaptive_curve(quarter_arc, tolerance=10e-6)

    assert len(fine.points) > len(coarse.points)
    assert fine.max_error <= coarse.max_error + GEOMDL_DELTA


def test_every_accepted_segment_respects_tolerance(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)

    assert result.max_error <= TOLERANCE + GEOMDL_DELTA
    for segment in result.segments:
        assert segment["sagitta"] <= TOLERANCE + GEOMDL_DELTA
        assert segment["curvature_residual"] <= 8.0 * TOLERANCE + GEOMDL_DELTA


def test_every_accepted_segment_respects_max_length(straight_line):
    result = discretize.adaptive_curve(straight_line)

    for segment in result.segments:
        assert segment["chord_length"] <= result.max_length * 1.001 + GEOMDL_DELTA


def test_straight_section_is_arc_length_tiled(straight_line):
    result = discretize.adaptive_curve(straight_line)

    # A perfectly straight section must be tiled by a finite number of length-budget segments instead of
    # collapsing into a single chord or being bisected into unnecessarily small fragments.
    assert 9 <= len(result.segments) <= 11
    assert abs(result.arc_length - 10.0) < GEOMDL_DELTA
    for segment in result.segments:
        assert segment["sagitta"] == 0.0


def test_arc_length_criterion_can_be_disabled(straight_line):
    result = discretize.adaptive_curve(straight_line, max_length=float("inf"))

    # Zero curvature plus no length cap means the single endpoint chord is sufficient
    assert len(result.segments) == 1
    assert result.points[0] == [0.0, 0.0]
    assert result.points[-1] == [10.0, 0.0]


def test_arc_length_converges_to_curve_length(quarter_arc):
    quarter_arc.delta = 0.0001
    reference_length = operations.length_curve(quarter_arc)

    coarse = discretize.adaptive_curve(quarter_arc, tolerance=10e-4)
    fine = discretize.adaptive_curve(quarter_arc, tolerance=10e-6)

    # The chord polyline underestimates the arc length and converges as the tolerance is tightened
    assert coarse.arc_length <= reference_length + GEOMDL_DELTA
    assert abs(fine.arc_length - reference_length) < abs(coarse.arc_length - reference_length)


def test_curvature_of_circular_arc(quarter_arc):
    # The Bezier arc approximates a circle of radius 5, hence curvature must be close to 1/5
    curvatures = [discretize.curve_curvature(quarter_arc, idx / 20.0) for idx in range(21)]

    assert max(curvatures) - min(curvatures) < 0.01
    assert abs(sum(curvatures) / len(curvatures) - 0.2) < 0.01


# ---------------------------------------------------------------------------
# Error recomputation
# ---------------------------------------------------------------------------


def test_recompute_matches_reported_error(mixed_curve):
    result = discretize.adaptive_curve(mixed_curve, tolerance=TOLERANCE)
    report = result.recompute()

    assert report["accepted"]
    assert abs(report["max_error"] - result.max_error) < GEOMDL_DELTA
    assert len(report["segments"]) == len(result.segments)
    for reported, recomputed in zip(result.segments, report["segments"]):
        assert abs(reported["sagitta"] - recomputed["sagitta"]) < GEOMDL_DELTA
        assert abs(reported["chord_length"] - recomputed["chord_length"]) < GEOMDL_DELTA


def test_recompute_detects_too_coarse_parameters(quarter_arc):
    # Only the endpoints cannot describe the curved arc within a tight tolerance
    report = discretize.recompute_error(quarter_arc, [0.0, 1.0], tolerance=TOLERANCE)

    assert not report["accepted"]
    assert report["max_error"] > TOLERANCE

    # A dense parameter list passes the same tolerance
    dense = [idx / 500.0 for idx in range(501)]
    dense_report = discretize.recompute_error(quarter_arc, dense, tolerance=TOLERANCE)

    assert dense_report["accepted"]
    assert dense_report["max_error"] <= TOLERANCE + GEOMDL_DELTA


def test_recompute_validates_inputs(mixed_curve):
    with raises(GeomdlException):
        discretize.recompute_error(mixed_curve, [0.5])
    with raises(GeomdlException):
        discretize.recompute_error(mixed_curve, [0.1, 0.1, 0.2])
    with raises(GeomdlException):
        discretize.recompute_error(mixed_curve, [0.0, 0.5, 0.4, 1.0])
    with raises(GeomdlException):
        discretize.recompute_error(mixed_curve, [0.0, 1.0], tolerance=-1.0)


# ---------------------------------------------------------------------------
# Uniform sampling interface is unchanged
# ---------------------------------------------------------------------------


def test_uniform_curve_uses_evenly_spaced_parameters(mixed_curve):
    parameters, points = discretize.uniform_curve(mixed_curve, sample_size=101)

    assert len(parameters) == 101
    assert len(points) == 101
    assert parameters[0] == 0.0
    assert parameters[-1] == 1.0
    for idx in range(1, len(parameters)):
        assert abs((parameters[idx] - parameters[idx - 1]) - 0.01) < GEOMDL_DELTA


def test_uniform_curve_matches_evaluator(mixed_curve):
    parameters, points = discretize.uniform_curve(mixed_curve, sample_size=50)
    reference = mixed_curve.evaluate_list(parameters)

    assert points == reference


def test_uniform_curve_default_sample_size(mixed_curve):
    parameters, _ = discretize.uniform_curve(mixed_curve)

    assert len(parameters) == mixed_curve.sample_size


def test_uniform_curve_validates_arguments(mixed_curve):
    with raises(GeomdlException):
        discretize.uniform_curve(mixed_curve, sample_size=1)
    with raises(GeomdlException):
        discretize.uniform_curve(mixed_curve, start=0.7, stop=0.3)


def test_adaptive_arguments_validated(mixed_curve):
    with raises(GeomdlException):
        discretize.adaptive_curve(mixed_curve, tolerance=0.0)
    with raises(GeomdlException):
        discretize.adaptive_curve(mixed_curve, tolerance=-1.0)
    with raises(GeomdlException):
        discretize.adaptive_curve(mixed_curve, max_length=0.0)
    with raises(GeomdlException):
        discretize.adaptive_curve(mixed_curve, max_depth=0)
    with raises(GeomdlException):
        discretize.adaptive_curve(mixed_curve, start=0.8, stop=0.2)


def test_strict_mode_raises_on_saturation(quarter_arc):
    with raises(GeomdlException):
        discretize.adaptive_curve(quarter_arc, tolerance=10e-12, max_depth=3, strict=True)

    relaxed = discretize.adaptive_curve(quarter_arc, tolerance=10e-12, max_depth=3, strict=False)
    assert relaxed.saturated
    assert relaxed.max_subdivision_depth == 3


# ---------------------------------------------------------------------------
# Rational and 3-dimensional curves
# ---------------------------------------------------------------------------


def test_nurbs_curve_adaptive(nurbs_mixed):
    result = discretize.adaptive_curve(nurbs_mixed, tolerance=TOLERANCE)

    assert result.parameters[0] == 0.0
    assert result.parameters[-1] == 1.0
    assert result.recompute()["accepted"]
    for point in result.points:
        assert len(point) == 2


def test_three_dimensional_curve_adaptive():
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [[0, 0, 0], [5, 0, 2], [10, 0, 4], [10, 5, 6], [5, 10, 8], [0, 10, 10]]
    curve.knotvector = [0, 0, 0, 0, 0.4, 0.6, 1, 1, 1, 1]

    result = discretize.adaptive_curve(curve, tolerance=TOLERANCE)

    assert all(len(point) == 3 for point in result.points)
    assert result.recompute()["accepted"]


def test_three_dimensional_nurbs_curve_adaptive():
    curve = NURBS.Curve()
    curve.degree = 3
    curve.ctrlptsw = [[0, 0, 0, 1], [5, 0, 2, 0.5], [10, 0, 4, 1], [10, 5, 6, 0.8], [5, 10, 8, 1], [0, 10, 10, 1]]
    curve.knotvector = [0, 0, 0, 0, 0.4, 0.6, 1, 1, 1, 1]

    result = discretize.adaptive_curve(curve, tolerance=TOLERANCE)

    assert all(len(point) == 3 for point in result.points)
    assert result.recompute()["accepted"]


@mark.parametrize("tolerance", [10e-3, 10e-4, 10e-5])
def test_fixed_curve_family_assertions(mixed_curve, quarter_arc, straight_line, nurbs_mixed, tolerance):
    for curve in (mixed_curve, quarter_arc, straight_line, nurbs_mixed):
        result = discretize.adaptive_curve(curve, tolerance=tolerance)

        # Shape fidelity: endpoints, monotonicity and independently recomputable error
        assert result.parameters[0] == 0.0
        assert result.parameters[-1] == 1.0
        assert result.parameters == sorted(result.parameters)
        assert result.recompute(tolerance=tolerance)["accepted"]
        assert result.max_error <= tolerance + GEOMDL_DELTA
