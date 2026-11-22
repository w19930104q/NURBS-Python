"""
.. module:: _operations
    :platform: Unix, Windows
    :synopsis: Helper functions for operations module

.. moduleauthor:: Onur R. Bingol <contact@onurbingol.net>

"""

from . import linalg, helpers
from .exceptions import GeomdlException


# Initialize an empty __all__ for controlling imports
__all__ = []

# Relative tolerance for detecting vanishing 1st-order partial derivatives on degenerate surface positions
_DEGENERATE_DERIV_TOL = 1e-12


def tangent_curve_single(obj, u, normalize):
    """Evaluates the curve tangent vector at the given parameter value.

    The output returns a list containing the starting point (i.e. origin) of the vector and the vector itself.

    :param obj: input curve
    :type obj: abstract.Curve
    :param u: parameter
    :type u: float
    :param normalize: if True, the returned vector is converted to a unit vector
    :type normalize: bool
    :return: a list containing "point" and "vector" pairs
    :rtype: tuple
    """
    # 1st derivative of the curve gives the tangent
    ders = obj.derivatives(u, 1)

    point = ders[0]
    vector = linalg.vector_normalize(ders[1]) if normalize else ders[1]

    return tuple(point), tuple(vector)


def tangent_curve_single_list(obj, param_list, normalize):
    """Evaluates the curve tangent vectors at the given list of parameter values.

    :param obj: input curve
    :type obj: abstract.Curve
    :param param_list: parameter list
    :type param_list: list or tuple
    :param normalize: if True, the returned vector is converted to a unit vector
    :type normalize: bool
    :return: a list containing "point" and "vector" pairs
    :rtype: tuple
    """
    ret_vector = []
    for param in param_list:
        temp = tangent_curve_single(obj, param, normalize)
        ret_vector.append(temp)
    return tuple(ret_vector)


def _boundary_limit_sign(param, domain_start, domain_end):
    """Returns the sign of the one-sided limit direction at the given parametric position.

    A degenerate boundary of the parametric domain can only be approached from the inside of the domain.
    Therefore, limit directions approaching the lower boundary get a positive sign and limit directions
    approaching the upper boundary get a negative sign.

    :param param: parameter value
    :type param: float
    :param domain_start: start of the parametric domain
    :type domain_start: float
    :param domain_end: end of the parametric domain
    :type domain_end: float
    :return: 1.0 if the position is closer to the start of the domain, -1.0 otherwise
    :rtype: float
    """
    return 1.0 if abs(param - domain_start) <= abs(param - domain_end) else -1.0


def _surface_tangent_frame(obj, uv):
    """Evaluates the surface point and the u- and v-tangent vectors at the given (u, v) parameter pair.

    On degenerate parametric positions, e.g. collapsed (degenerate) edges where one or both of the 1st-order
    partial derivatives vanish, the vanishing derivative vectors are replaced by their limit directions
    computed from the 2nd-order partial derivatives. This keeps the tangent frame, and therefore the surface
    normal, consistent with the limiting values approached from the regular neighborhood of the degenerate
    position, including the corners of the parametric domain.

    :param obj: input surface
    :type obj: abstract.Surface
    :param uv: (u,v) parameter pair
    :type uv: list or tuple
    :return: surface point, derivative w.r.t. u and derivative w.r.t. v (or their limit directions)
    :rtype: tuple
    """
    # Tangent vectors are the 1st-order partial derivatives of the surface
    skl = obj.derivatives(uv[0], uv[1], 1)

    point = skl[0][0]
    der_u = skl[1][0]
    der_v = skl[0][1]

    # Check if the 1st-order partial derivatives vanish. The magnitudes are compared relatively, so that
    # regular surfaces of any geometric scale keep working with the 1st-order derivatives. A vanishing
    # derivative may not be exactly zero due to floating-point round-off; in that case normalizing the
    # residual noise would result in an arbitrary tangent direction.
    mag_u = linalg.vector_magnitude(der_u)
    mag_v = linalg.vector_magnitude(der_v)
    zero_u = mag_u <= _DEGENERATE_DERIV_TOL * mag_v
    zero_v = mag_v <= _DEGENERATE_DERIV_TOL * mag_u

    # Regular parametric position: no limit analysis is required
    if not zero_u and not zero_v:
        return point, der_u, der_v

    # Degenerate parametric position: compute the 2nd-order partial derivatives
    skl2 = obj.derivatives(uv[0], uv[1], 2)
    der_uu = skl2[2][0]
    der_vv = skl2[0][2]
    der_uv = skl2[1][1]

    # Limit directions which may replace the vanishing 1st-order partial derivatives.
    # On a collapsed edge, the 1st-order partial derivative along the edge direction approaches
    # the mixed 2nd-order derivative, e.g. dS/dv(u0 + h, v) ~= h * d2S/dudv(u0, v) as h -> 0.
    # The sign of a limit direction flips on the upper boundary of the parametric domain,
    # since the degenerate boundary can only be approached from the inside, i.e. h < 0.
    sign_u = _boundary_limit_sign(uv[0], obj.knotvector_u[obj.degree_u], obj.knotvector_u[-(obj.degree_u + 1)])
    sign_v = _boundary_limit_sign(uv[1], obj.knotvector_v[obj.degree_v], obj.knotvector_v[-(obj.degree_v + 1)])
    subs_u = (
        [der_u]
        if not zero_u
        else [[s * d for d in der] for s, der in ((sign_u, der_uu), (sign_v, der_uv))]
    )
    subs_v = (
        [der_v]
        if not zero_v
        else [[s * d for d in der] for s, der in ((sign_u, der_uv), (sign_v, der_vv))]
    )

    # Pick the first pair of tangent vectors which spans a well-defined tangent plane
    for vec_u in subs_u:
        if linalg.vector_magnitude(vec_u) == 0.0:
            continue
        for vec_v in subs_v:
            if linalg.vector_magnitude(vec_v) == 0.0:
                continue
            if linalg.vector_magnitude(linalg.vector_cross(vec_u, vec_v)) > 0.0:
                return point, vec_u, vec_v

    # The tangent plane is not defined at this position: return the vanishing derivatives
    return point, der_u, der_v


def tangent_surface_single(obj, uv, normalize):
    """Evaluates the surface tangent vector at the given (u,v) parameter pair.

    The output returns a list containing the starting point (i.e., origin) of the vector and the vectors themselves.

    :param obj: input surface
    :type obj: abstract.Surface
    :param uv: (u,v) parameter pair
    :type uv: list or tuple
    :param normalize: if True, the returned tangent vector is converted to a unit vector
    :type normalize: bool
    :return: A list in the order of "surface point", "derivative w.r.t. u" and "derivative w.r.t. v"
    :rtype: list
    """
    # Tangent is the 1st derivative of the surface (or its limit direction on degenerate positions)
    point, der_u, der_v = _surface_tangent_frame(obj, uv)

    vector_u = linalg.vector_normalize(der_u) if normalize else der_u
    vector_v = linalg.vector_normalize(der_v) if normalize else der_v

    return tuple(point), tuple(vector_u), tuple(vector_v)


def tangent_surface_single_list(obj, param_list, normalize):
    """Evaluates the surface tangent vectors at the given list of parameter values.

    :param obj: input surface
    :type obj: abstract.Surface
    :param param_list: parameter list
    :type param_list: list or tuple
    :param normalize: if True, the returned vector is converted to a unit vector
    :type normalize: bool
    :return: a list containing "point" and "vector" pairs
    :rtype: tuple
    """
    ret_vector = []
    for param in param_list:
        temp = tangent_surface_single(obj, param, normalize)
        ret_vector.append(temp)
    return tuple(ret_vector)


def normal_surface_single(obj, uv, normalize):
    """Evaluates the surface normal vector at the given (u, v) parameter pair.

    The output returns a list containing the starting point (i.e. origin) of the vector and the vector itself.

    :param obj: input surface
    :type obj: abstract.Surface
    :param uv: (u,v) parameter pair
    :type uv: list or tuple
    :param normalize: if True, the returned normal vector is converted to a unit vector
    :type normalize: bool
    :return: a list in the order of "surface point" and "normal vector"
    :rtype: list
    """
    # Take the 1st derivative of the surface (or its limit direction on degenerate positions)
    point, der_u, der_v = _surface_tangent_frame(obj, uv)

    vector = linalg.vector_cross(der_u, der_v)
    vector = linalg.vector_normalize(vector) if normalize else vector

    return tuple(point), tuple(vector)


def normal_surface_single_list(obj, param_list, normalize):
    """Evaluates the surface normal vectors at the given list of parameter values.

    :param obj: input surface
    :type obj: abstract.Surface
    :param param_list: parameter list
    :type param_list: list or tuple
    :param normalize: if True, the returned vector is converted to a unit vector
    :type normalize: bool
    :return: a list containing "point" and "vector" pairs
    :rtype: tuple
    """
    ret_vector = []
    for param in param_list:
        temp = normal_surface_single(obj, param, normalize)
        ret_vector.append(temp)
    return tuple(ret_vector)


def find_ctrlpts_curve(t, curve, **kwargs):
    """Finds the control points involved in the evaluation of the curve point defined by the input parameter.

    This function uses a modified version of the algorithm *A3.1 CurvePoint* from The NURBS Book by Piegl & Tiller.

    :param t: parameter
    :type t: float
    :param curve: input curve object
    :type curve: abstract.Curve
    :return: 1-dimensional control points array
    :rtype: list
    """
    # Get keyword arguments
    span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    # Find spans and the constant index
    span = span_func(curve.degree, curve.knotvector, len(curve.ctrlpts), t)
    idx = span - curve.degree

    # Find control points involved in evaluation of the curve point at the input parameter
    curve_ctrlpts = [() for _ in range(curve.degree + 1)]
    for i in range(0, curve.degree + 1):
        curve_ctrlpts[i] = curve.ctrlpts[idx + i]

    # Return control points array
    return curve_ctrlpts


def find_ctrlpts_surface(t_u, t_v, surf, **kwargs):
    """Finds the control points involved in the evaluation of the surface point defined by the input parameter pair.

    This function uses a modified version of the algorithm *A3.5 SurfacePoint* from The NURBS Book by Piegl & Tiller.

    :param t_u: parameter on the u-direction
    :type t_u: float
    :param t_v: parameter on the v-direction
    :type t_v: float
    :param surf: input surface
    :type surf: abstract.Surface
    :return: 2-dimensional control points array
    :rtype: list
    """
    # Get keyword arguments
    span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    # Find spans
    span_u = span_func(surf.degree_u, surf.knotvector_u, surf.ctrlpts_size_u, t_u)
    span_v = span_func(surf.degree_v, surf.knotvector_v, surf.ctrlpts_size_v, t_v)

    # Constant indices
    idx_u = span_u - surf.degree_u
    idx_v = span_v - surf.degree_v

    # Find control points involved in evaluation of the surface point at the input parameter pair (u, v)
    surf_ctrlpts = [[] for _ in range(surf.degree_u + 1)]
    for k in range(surf.degree_u + 1):
        temp = [() for _ in range(surf.degree_v + 1)]
        for l in range(surf.degree_v + 1):
            temp[l] = surf.ctrlpts2d[idx_u + k][idx_v + l]
        surf_ctrlpts[k] = temp

    # Return 2-dimensional control points array
    return surf_ctrlpts


def link_curves(*args, **kwargs):
    """Links the input curves together.

    The end control point of the curve k has to be the same with the start control point of the curve k + 1.

    Keyword Arguments:
        * ``tol``: tolerance value for checking equality. *Default: 10e-8*
        * ``validate``: flag to enable input validation. *Default: False*

    :return: a tuple containing knot vector, control points, weights vector and knots
    """
    # Get keyword arguments
    tol = kwargs.get("tol", 10e-8)
    validate = kwargs.get("validate", False)

    # Validate input
    if validate:
        for idx in range(len(args) - 1):
            if linalg.point_distance(args[idx].ctrlpts[-1], args[idx + 1].ctrlpts[0]) > tol:
                raise GeomdlException("Curve #" + str(idx) + " and Curve #" + str(idx + 1) + " don't touch each other")

    kv = []  # new knot vector
    cpts = []  # new control points array
    wgts = []  # new weights array
    kv_connected = []  # superfluous knots to be removed
    pdomain_end = 0

    # Loop though the curves
    for arg in args:
        # Process knot vectors
        if not kv:
            kv += list(
                arg.knotvector[: -(arg.degree + 1)]
            )  # get rid of the last superfluous knot to maintain split curve notation
            cpts += list(arg.ctrlpts)
            # Process control points
            if arg.rational:
                wgts += list(arg.weights)
            else:
                tmp_w = [1.0 for _ in range(arg.ctrlpts_size)]
                wgts += tmp_w
        else:
            tmp_kv = [pdomain_end + k for k in arg.knotvector[1 : -(arg.degree + 1)]]
            kv += tmp_kv
            cpts += list(arg.ctrlpts[1:])
            # Process control points
            if arg.rational:
                wgts += list(arg.weights[1:])
            else:
                tmp_w = [1.0 for _ in range(arg.ctrlpts_size - 1)]
                wgts += tmp_w

        pdomain_end += arg.knotvector[-1]
        kv_connected.append(pdomain_end)

    # Fix curve by appending the last knot to the end
    kv += [pdomain_end for _ in range(arg.degree + 1)]
    # Remove the last knot from knot insertion list
    kv_connected.pop()

    return kv, cpts, wgts, kv_connected
