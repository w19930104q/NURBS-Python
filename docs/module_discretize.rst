Curve Discretization
^^^^^^^^^^^^^^^^^^^^

The ``discretize`` module provides adaptive discretization algorithms for curves. Unlike uniform sampling,
which evaluates the curve at evenly spaced parameters, adaptive discretization concentrates the samples where
the curve needs them: high curvature sections receive more samples while long straight sections receive fewer.

The sampling criterion combines two independent measures:

* **Curvature / geometric error:** the midpoint *sagitta* of every segment, i.e. the perpendicular distance of
  the curve point at the parametric midpoint from the chord connecting the segment endpoints, must not exceed a
  user-defined tolerance. A curvature-weighted residual :math:`\kappa \cdot s^2` additionally guards sharp turns
  and inflection points.
* **Arc length:** the chord length of every segment must not exceed a maximum segment length, so that the
  sampling density stays coupled to the arc length even on perfectly straight curve sections.

The following example illustrates the usage of the adaptive discretization:

.. code-block:: python
    :linenos:

    from geomdl import BSpline
    from geomdl import discretize

    # Create a curve instance
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [[0, 0], [2, 0], [4, 0], [5, 0.2], [5.2, 1], [5.2, 3], [5, 5]]
    curve.knotvector = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]

    # Adaptive discretization with a geometric tolerance of 1e-4
    result = discretize.adaptive_curve(curve, tolerance=10e-5)

    # Adaptively placed parameters and evaluated points
    parameters = result.parameters
    points = result.points

    # Independently recompute and verify the discretization error
    report = result.recompute()
    assert report["accepted"]

Uniform sampling remains available through :py:attr:`.abstract.Curve.evalpts`,
:py:attr:`.abstract.Curve.delta` and :py:attr:`.abstract.Curve.sample_size` exactly as before; the
``discretize`` module never changes those settings. A convenience wrapper is provided for completeness:

.. code-block:: python
    :linenos:

    parameters, points = discretize.uniform_curve(curve, sample_size=100)

Class Reference
===============

Adaptive Discretization
-----------------------

.. autoclass:: geomdl.discretize.AdaptiveDiscretization
    :members:
    :undoc-members:
    :show-inheritance:

Function Reference
==================

.. autofunction:: geomdl.discretize.adaptive_curve

.. autofunction:: geomdl.discretize.uniform_curve

.. autofunction:: geomdl.discretize.recompute_error

.. autofunction:: geomdl.discretize.curve_curvature
