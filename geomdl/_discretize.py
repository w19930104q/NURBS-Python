"""
.. module:: _discretize
    :platform: Unix, Windows
    :synopsis: Helper functions for adaptive curve discretization

.. moduleauthor:: Onur R. Bingol <contact@onurbingol.net>

"""

from . import linalg, utilities
from .exceptions import GeomdlException

# Initialize an empty __all__ for controlling imports
__all__ = []

# Default maximum number of midpoint subdivisions per parametric interval
DEFAULT_MAX_DEPTH = 16

# A velocity vector whose magnitude is below this value is treated as a cusp (zero velocity)
_ZERO_VELOCITY = 10e-12

# Relative slack applied to the arc length criterion so that a segment of exactly the budgeted length is
# accepted instead of being fragmented by the next subdivision
_LENGTH_TOLERANCE = 10e-3


def curve_curvature(curve, u):
    """Computes the curvature of the input curve at parameter ``u``.

    Curvature is computed from the first and second order curve derivatives using

    .. math::

        \\kappa(u) = \\frac{\\|C'(u) \\times C''(u)\\|}{\\|C'(u)\\|^3}

    The computation works for planar (2-D) and spatial (3-D) curves. If the first derivative vanishes at ``u``
    (a cusp), the curvature is defined as zero for discretization purposes.

    :param curve: input curve
    :type curve: abstract.Curve
    :param u: parameter value
    :type u: float
    :return: curvature at the parameter
    :rtype: float
    """
    ders = curve.derivatives(u=u, order=2)
    d1 = ders[1]
    d2 = ders[2]

    speed = linalg.vector_magnitude(d1)
    if speed <= _ZERO_VELOCITY:
        return 0.0

    # The cross-product helper accepts 2-D vectors by extending them to 3-D
    cross = linalg.vector_cross(d1, d2)
    return linalg.vector_magnitude(cross) / (speed**3)


def _sagitta(point_start, point_mid, point_end):
    """Computes the sagitta (maximum deviation of the midpoint from the chord) of 3 curve points.

    The sagitta is the perpendicular distance of the curve point evaluated at the parametric midpoint from the
    chord connecting the two segment endpoints. It measures the deviation of the curve segment from the straight
    line approximation and therefore couples curvature and arc length: it grows with both the local curvature and
    the segment length.

    :param point_start: curve point at the segment start parameter
    :type point_start: list, tuple
    :param point_mid: curve point at the segment midpoint parameter
    :type point_mid: list, tuple
    :param point_end: curve point at the segment end parameter
    :type point_end: list, tuple
    :return: sagitta of the segment
    :rtype: float
    """
    chord_vector = linalg.vector_generate(point_start, point_end, normalize=False)
    chord_length = linalg.vector_magnitude(chord_vector)

    # Coincident endpoints (e.g. a closed loop collapsed to a point); use the direct distance
    if chord_length <= _ZERO_VELOCITY:
        return linalg.point_distance(point_start, point_mid)

    # Vector from the start point to the midpoint projected onto the chord
    to_mid = linalg.vector_generate(point_start, point_mid, normalize=False)
    projection = linalg.vector_dot(to_mid, chord_vector) / chord_length

    # Perpendicular component of the start-to-midpoint vector
    projection_vector = linalg.vector_multiply(chord_vector, projection / chord_length)
    perpendicular = linalg.vector_sum(to_mid, projection_vector, coeff=-1.0)
    return linalg.vector_magnitude(perpendicular)


def _auto_max_length(curve):
    """Computes the default maximum segment length from the curve bounding box.

    One tenth of the bounding box diagonal is used as the upper bound on the length of an accepted segment.
    This guarantees that even perfectly straight (zero curvature) curve sections are discretized into a finite
    number of segments instead of being represented by a single long chord.

    :param curve: input curve
    :type curve: abstract.Curve
    :return: default maximum segment length
    :rtype: float
    """
    box = curve.bbox
    diagonal = linalg.point_distance(box[0], box[1])
    if diagonal <= _ZERO_VELOCITY:
        return 1.0
    return diagonal / 10.0


def _validate_interval(curve, start, stop):
    """Validates the evaluation interval of the input curve."""
    if curve._kv_normalize and not utilities.check_params([start, stop]):
        raise GeomdlException("Parameters should be between 0 and 1", data=dict(start=start, stop=stop))
    if stop <= start:
        raise GeomdlException(
            "The stop parameter must be strictly greater than the start parameter", data=dict(start=start, stop=stop)
        )


def _validate_arguments(tolerance, max_length, max_depth):
    """Validates the keyword arguments of the adaptive discretization algorithm."""
    if not isinstance(tolerance, (int, float)) or tolerance <= 0.0:
        raise GeomdlException("Tolerance must be a positive number", data=dict(tolerance=tolerance))
    if max_length is not None and (not isinstance(max_length, (int, float)) or max_length <= 0.0):
        raise GeomdlException(
            "Maximum segment length must be a positive number or None", data=dict(max_length=max_length)
        )
    if not isinstance(max_depth, int) or max_depth < 1:
        raise GeomdlException("Maximum subdivision depth must be a positive integer", data=dict(max_depth=max_depth))


def adaptive_curve(curve, **kwargs):
    """Adaptive discretization of a curve driven by curvature and arc length.

    The curve is discretized through recursive midpoint subdivision. Every parametric segment is accepted only
    when both of the following sampling criteria are satisfied:

    * **Curvature criterion (geometric error):** the sagitta of the segment, i.e. the perpendicular distance of
      the curve point at the parametric midpoint from the chord connecting the segment endpoints, must not exceed
      ``tolerance``. The sagitta grows with both the local curvature and the segment length, so high curvature
      sections are subdivided more densely while straight sections need fewer samples. A curvature-weighted
      residual at the midpoint, :math:`\\kappa \\cdot s^2`, guards against sharp turns located away from the
      midpoint and inflection points whose sagitta vanishes.
    * **Arc length criterion:** the chord length of the segment must not exceed ``max_length``. This couples the
      sampling density to the arc length, so that long straight sections cannot collapse into a single segment.

    Subdivision stops when both criteria hold or when ``max_depth`` midpoint splits have been applied to a
    segment. The algorithm operates on the evaluated curve points and never changes the control points, the knot
    vector or the uniform sampling settings of the input curve.

    Keyword Arguments:
        * ``start``: start parameter of the discretization interval. *Default: first valid knot*
        * ``stop``: stop parameter of the discretization interval. *Default: last valid knot*
        * ``tolerance``: maximum allowed sagitta (geometric deviation) per segment. *Default: 1e-4*
        * ``max_length``: maximum allowed chord length per segment. If ``None``, one tenth of the curve bounding
          box diagonal is used; set to ``float('inf')`` to disable the arc length criterion. *Default: None*
        * ``max_depth``: maximum number of midpoint subdivisions per segment. *Default: 16*
        * ``strict``: if ``True``, raises :class:`.GeomdlException` when a segment reaches ``max_depth`` without
          satisfying the criteria; otherwise the segment is accepted. *Default: False*

    :param curve: input curve
    :type curve: abstract.Curve
    :return: adaptive discretization result
    :rtype: AdaptiveDiscretization
    """
    # Imported here to avoid a circular import at module load time
    from .discretize import AdaptiveDiscretization

    # Keyword arguments
    start = kwargs.get("start", curve.knotvector[curve.degree])
    stop = kwargs.get("stop", curve.knotvector[-(curve.degree + 1)])
    tolerance = float(kwargs.get("tolerance", 10e-5))
    max_length = kwargs.get("max_length", None)
    max_depth = kwargs.get("max_depth", DEFAULT_MAX_DEPTH)
    strict = kwargs.get("strict", False)

    # Input validation
    _validate_interval(curve, start, stop)
    _validate_arguments(tolerance, max_length, max_depth)
    if max_length is None:
        max_length = _auto_max_length(curve)

    # Cache evaluated points and curvature values by parameter to keep subdivision deterministic and repeatable:
    # the same parameter always yields the same curve point during the whole discretization.
    point_cache = dict()
    curvature_cache = dict()

    def point_at(u):
        if u not in point_cache:
            point_cache[u] = curve.evaluate_single(u)
        return point_cache[u]

    def curvature_at(u):
        if u not in curvature_cache:
            curvature_cache[u] = curve_curvature(curve, u)
        return curvature_cache[u]

    def segment_errors(u0, u1, u2, p0, p1, p2):
        """Evaluates the two sampling criteria on the segment [u0, u2].

        Returns a tuple containing the chord length, the midpoint sagitta, the curvature-weighted residual and
        three boolean flags reporting which of the arc length and curvature criteria are satisfied.
        """
        # Arc length criterion: the chord length approximates the arc length of the segment.
        # A relative tolerance absorbs chord-length estimation differences between parent and child segments.
        chord_length = linalg.point_distance(p0, p2)
        length_ok = chord_length <= max_length * (1.0 + _LENGTH_TOLERANCE)

        # Curvature criterion, component 1: midpoint sagitta against the endpoint chord
        sagitta = _sagitta(p0, p1, p2)
        sagitta_ok = sagitta <= tolerance

        # Curvature criterion, component 2: curvature-weighted arc length residual at the midpoint.
        # For a circular arc of curvature kappa the sagitta is approximately kappa * s^2 / 8, hence the
        # residual kappa * s^2 is proportional to the geometric error even when the midpoint sits on an
        # inflection point and its sagitta vanishes.
        curvature_residual = curvature_at(u1) * chord_length**2
        residual_ok = curvature_residual <= 8.0 * tolerance

        return chord_length, sagitta, curvature_residual, length_ok, sagitta_ok, residual_ok

    def split_parameter(u0, u2, p0, p2, length_ok, sagitta_ok, residual_ok):
        """Chooses the parameter at which a failing segment is split.

        When a curvature criterion fails, the parametric midpoint is used so that the local geometric error is
        halved at every subdivision. When only the arc length criterion fails, the split parameter is computed
        from the ratio of the maximum allowed length to the chord length, so that the left segment spans exactly
        one length budget. Repeated application tiles a straight section with segments of approximately
        ``max_length`` instead of generating ever smaller bisected fragments. The ratio is clamped to guarantee
        steady progress and a finite number of subdivisions.
        """
        midpoint = (u0 + u2) / 2.0
        if sagitta_ok and residual_ok and not length_ok:
            chord_length = linalg.point_distance(p0, p2)
            if chord_length > _ZERO_VELOCITY:
                ratio = min(0.9, max(0.1, max_length / chord_length))
                return u0 + ratio * (u2 - u0)
        return midpoint

    # Each stack entry is (u_start, u_end, depth). Segments are processed depth-first and the accepted
    # segments are collected in left-to-right parametric order.
    stack = [(start, stop, 0)]
    segments = []
    max_observed_error = 0.0
    max_observed_depth = 0
    saturated = False

    while stack:
        u0, u2, depth = stack.pop()
        u1 = (u0 + u2) / 2.0
        p0 = point_at(u0)
        p1 = point_at(u1)
        p2 = point_at(u2)

        chord_length, sagitta, curvature_residual, length_ok, sagitta_ok, residual_ok = segment_errors(
            u0, u1, u2, p0, p1, p2
        )
        criteria_met = length_ok and sagitta_ok and residual_ok

        if not criteria_met and depth >= max_depth:
            # The segment could not be refined any further
            saturated = True
            if strict:
                raise GeomdlException(
                    "Adaptive discretization reached the maximum subdivision depth",
                    data=dict(parameter_start=u0, parameter_stop=u2, max_depth=max_depth, tolerance=tolerance),
                )
            criteria_met = True

        if criteria_met:
            segments.append(
                dict(
                    parameter_start=u0,
                    parameter_stop=u2,
                    point_start=p0,
                    point_mid=p1,
                    point_end=p2,
                    chord_length=chord_length,
                    sagitta=sagitta,
                    curvature_residual=curvature_residual,
                    depth=depth,
                )
            )
            max_observed_error = max(max_observed_error, sagitta)
            max_observed_depth = max(max_observed_depth, depth)
        else:
            # Choose the split parameter from the failing criteria and push the right segment first so that
            # the left segment is processed first (LIFO order)
            us = split_parameter(u0, u2, p0, p2, length_ok, sagitta_ok, residual_ok)
            stack.append((us, u2, depth + 1))
            stack.append((u0, us, depth + 1))

    # Build the discretization report
    parameters = [seg["parameter_start"] for seg in segments]
    parameters.append(segments[-1]["parameter_stop"])
    points = [seg["point_start"] for seg in segments]
    points.append(segments[-1]["point_end"])
    arc_length = sum(seg["chord_length"] for seg in segments)

    return AdaptiveDiscretization(
        curve=curve,
        parameters=parameters,
        points=points,
        segments=segments,
        tolerance=tolerance,
        max_length=max_length,
        max_depth=max_depth,
        max_error=max_observed_error,
        arc_length=arc_length,
        max_subdivision_depth=max_observed_depth,
        saturated=saturated,
    )


def uniform_curve(curve, **kwargs):
    """Uniform discretization of a curve over an evenly spaced parameter grid.

    This is a convenience wrapper around the curve's uniform evaluation interface. It does not change the
    behavior of :py:attr:`.abstract.Curve.evalpts`, :py:attr:`.abstract.Curve.delta` and
    :py:attr:`.abstract.Curve.sample_size`; those continue to use evenly spaced parameters as before.

    Keyword Arguments:
        * ``start``: start parameter of the discretization interval. *Default: first valid knot*
        * ``stop``: stop parameter of the discretization interval. *Default: last valid knot*
        * ``sample_size``: number of evenly spaced parameters, including the endpoints. *Default: value computed
          from the current ``sample_size`` property of the curve*

    :param curve: input curve
    :type curve: abstract.Curve
    :return: list of parameters and the corresponding evaluated points
    :rtype: tuple
    """
    start = kwargs.get("start", curve.knotvector[curve.degree])
    stop = kwargs.get("stop", curve.knotvector[-(curve.degree + 1)])
    sample_size = kwargs.get("sample_size", curve.sample_size)

    _validate_interval(curve, start, stop)
    if not isinstance(sample_size, int) or sample_size < 2:
        raise GeomdlException("Sample size must be an integer greater than 1", data=dict(sample_size=sample_size))

    parameters = linalg.linspace(start, stop, sample_size, decimals=curve._precision)
    points = curve.evaluate_list(parameters)
    return parameters, points


def recompute_error(curve, parameters, **kwargs):
    """Recomputes the discretization error of a set of curve parameters from scratch.

    For every consecutive pair of parameters the function evaluates the curve at the endpoints and at the
    parametric midpoint and computes the midpoint sagitta, the chord length and the curvature-weighted residual
    :math:`\\kappa \\cdot s^2`. This provides an independent way of verifying the geometric error reported by
    :func:`.adaptive_curve`; the returned values are computed directly from the curve rather than read from a
    cached discretization.

    Keyword Arguments:
        * ``tolerance``: tolerance used to classify each segment as accepted or rejected. *Default: 1e-4*

    :param curve: input curve
    :type curve: abstract.Curve
    :param parameters: strictly increasing list of curve parameters, including both interval endpoints
    :type parameters: list, tuple
    :return: a report containing the per-segment errors, the maximum sagitta and the total polyline length
    :rtype: dict
    """
    tolerance = float(kwargs.get("tolerance", 10e-5))
    if not isinstance(tolerance, (int, float)) or tolerance <= 0.0:
        raise GeomdlException("Tolerance must be a positive number", data=dict(tolerance=tolerance))
    if not isinstance(parameters, (list, tuple)) or len(parameters) < 2:
        raise GeomdlException("At least two parameters are required", data=dict(num_parameters=len(parameters)))
    for idx in range(1, len(parameters)):
        if parameters[idx] <= parameters[idx - 1]:
            raise GeomdlException(
                "Parameters must be strictly increasing",
                data=dict(index=idx, parameter=parameters[idx], prev_parameter=parameters[idx - 1]),
            )

    segments = []
    total_length = 0.0
    max_sagitta = 0.0
    all_accepted = True

    for idx in range(len(parameters) - 1):
        u0 = parameters[idx]
        u2 = parameters[idx + 1]
        u1 = (u0 + u2) / 2.0

        p0 = curve.evaluate_single(u0)
        p1 = curve.evaluate_single(u1)
        p2 = curve.evaluate_single(u2)

        chord_length = linalg.point_distance(p0, p2)
        sagitta = _sagitta(p0, p1, p2)
        curvature_residual = curve_curvature(curve, u1) * chord_length**2
        accepted = sagitta <= tolerance and curvature_residual <= 8.0 * tolerance

        total_length += chord_length
        max_sagitta = max(max_sagitta, sagitta)
        all_accepted = all_accepted and accepted

        segments.append(
            dict(
                parameter_start=u0,
                parameter_stop=u2,
                chord_length=chord_length,
                sagitta=sagitta,
                curvature_residual=curvature_residual,
                accepted=accepted,
            )
        )

    return dict(
        segments=segments,
        max_error=max_sagitta,
        arc_length=total_length,
        tolerance=tolerance,
        accepted=all_accepted,
    )
