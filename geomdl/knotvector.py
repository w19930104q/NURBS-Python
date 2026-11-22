"""
.. module:: knotvector
    :platform: Unix, Windows
    :synopsis: Provides utility functions related to knot vector generation and validation

.. moduleauthor:: Onur R. Bingol <contact@onurbingol.net>

"""

from .linalg import linspace
from .exceptions import GeomdlException
from ._utilities import export


@export
def generate(degree, num_ctrlpts, **kwargs):
    """Generates an equally spaced knot vector.

    It uses the following equality to generate knot vector: :math:`m = n + p + 1`

    where;

    * :math:`p`, degree
    * :math:`n + 1`, number of control points
    * :math:`m + 1`, number of knots

    Keyword Arguments:

        * ``clamped``: Flag to choose from clamped or unclamped knot vector options. *Default: True*

    :param degree: degree
    :type degree: int
    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :return: knot vector
    :rtype: list
    """
    if degree == 0 or num_ctrlpts == 0:
        raise ValueError("Input values should be different than zero.")

    # Get keyword arguments
    clamped = kwargs.get("clamped", True)

    # Number of repetitions at the start and end of the array
    num_repeat = degree

    # Number of knots in the middle
    num_segments = num_ctrlpts - (degree + 1)

    if not clamped:
        # No repetitions at the start and end
        num_repeat = 0
        # Should conform the rule: m = n + p + 1
        num_segments = degree + num_ctrlpts - 1

    # First knots
    knot_vector = [0.0 for _ in range(0, num_repeat)]

    # Middle knots
    knot_vector += linspace(0.0, 1.0, num_segments + 2)

    # Last knots
    knot_vector += [1.0 for _ in range(0, num_repeat)]

    # Return auto-generated knot vector
    return knot_vector


@export
def normalize(knot_vector, decimals=18):
    """Normalizes the input knot vector to [0, 1] domain.

    :param knot_vector: knot vector to be normalized
    :type knot_vector: list, tuple
    :param decimals: rounding number
    :type decimals: int
    :return: normalized knot vector
    :rtype: list
    """
    try:
        if knot_vector is None or len(knot_vector) == 0:
            raise ValueError("Input knot vector cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Knot vector must be a list or tuple")
    except Exception:
        raise

    first_knot = float(knot_vector[0])
    last_knot = float(knot_vector[-1])
    denominator = last_knot - first_knot

    knot_vector_out = [
        float(("{:." + str(decimals) + "f}").format((float(kv) - first_knot) / denominator)) for kv in knot_vector
    ]

    return knot_vector_out


@export
def check(degree, knot_vector, num_ctrlpts):
    """Checks the validity of the input knot vector.

    The following rules are enforced (please refer to The NURBS Book (2nd Edition), p.50 for details):

    * the knot vector length must conform to the equality :math:`m = n + p + 1`
    * knots must be in non-decreasing (ascending) order
    * multiplicity of the first and the last knots must not exceed ``degree + 1``
    * multiplicity of the interior knots must not exceed ``degree + 1``

    Invalid input raises :class:`.GeomdlException` with a ``data`` attribute describing the offending
    knot(s), so that the problem can be located inside the input knot vector.

    :param degree: degree of the curve or the surface
    :type degree: int
    :param knot_vector: knot vector to be checked
    :type knot_vector: list, tuple
    :param num_ctrlpts: number of control points
    :type num_ctrlpts: int
    :return: True if the knot vector is valid
    :rtype: bool
    """
    try:
        if knot_vector is None or len(knot_vector) == 0:
            raise ValueError("Input knot vector cannot be empty")
    except TypeError as e:
        print("An error occurred: {}".format(e.args[-1]))
        raise TypeError("Knot vector must be a list or tuple")
    except Exception:
        raise

    # Check the formula; m = p + n + 1
    if len(knot_vector) != degree + num_ctrlpts + 1:
        raise GeomdlException(
            "Knot vector length does not conform to the rule m = p + n + 1",
            data=dict(degree=degree, num_ctrlpts=num_ctrlpts, expected=degree + num_ctrlpts + 1, actual=len(knot_vector)),
        )

    # Check ascending (non-decreasing) order
    for idx in range(1, len(knot_vector)):
        if knot_vector[idx] < knot_vector[idx - 1]:
            raise GeomdlException(
                "Knot vector must be in non-decreasing order",
                data=dict(index=idx, knot=knot_vector[idx], prev_knot=knot_vector[idx - 1]),
            )

    # Maximum allowed multiplicity for any knot value
    max_mult = degree + 1

    # Check the repetition (multiplicity) of the first knot
    start_mult = 1
    while start_mult < len(knot_vector) and knot_vector[start_mult] == knot_vector[0]:
        start_mult += 1
    if start_mult > max_mult:
        raise GeomdlException(
            "Multiplicity of the first knot exceeds the allowed maximum",
            data=dict(knot=knot_vector[0], multiplicity=start_mult, max_multiplicity=max_mult),
        )

    # Check the repetition (multiplicity) of the last knot
    end_mult = 1
    while end_mult < len(knot_vector) and knot_vector[-(end_mult + 1)] == knot_vector[-1]:
        end_mult += 1
    if end_mult > max_mult:
        raise GeomdlException(
            "Multiplicity of the last knot exceeds the allowed maximum",
            data=dict(knot=knot_vector[-1], multiplicity=end_mult, max_multiplicity=max_mult),
        )

    # Check the multiplicity of the interior knots
    idx = start_mult
    interior_end = len(knot_vector) - end_mult
    while idx < interior_end:
        mult = 1
        while idx + mult < interior_end and knot_vector[idx + mult] == knot_vector[idx]:
            mult += 1
        if mult > max_mult:
            raise GeomdlException(
                "Multiplicity of the interior knot at index " + str(idx) + " exceeds the allowed maximum",
                data=dict(knot=knot_vector[idx], multiplicity=mult, max_multiplicity=max_mult, index=idx),
            )
        idx += mult

    return True
