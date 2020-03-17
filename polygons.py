from matplotlib import path
import numpy as np
from scipy.ndimage import filters


def GenerateConvexPolygon(
    n_vertices, min_segment_angle, scale, raster_dim, subpixel_res, shift_to_mean=False
):
    """Returns a random convex polygon with constraints on the minimum angle size.

  An ellipse is split into 'n_vertices' segments and a vertex is picked in
  each segment. Vertices are then selected within each segment such that the
  minimum segment angle (vertex A, zero, vertex B) is respected.

  Args:
    n_vertices: Integer > 2, number of vertices of the polygon.
    min_segment_angle: Float, minimum segment angle (in degrees).
    scale: Float in (0, 1], downscaling factor for the generated polygon.
    raster_dim: Integer > 1, size of the square grid of pixels to return.
    subpixel_res: Integer, each cell of the raster is split into subpixels.
    shift_to_mean: Boolean, whether to move the center of mass to the center
                   of the raster.
  """
    segment_angle = 360.0 / n_vertices
    max_angle = segment_angle - min_segment_angle / 2.0
    grid_length = raster_dim * scale
    angles = []
    for i in range(n_vertices):
        offset_angle = np.random.rand() * max_angle
        angles.append(i * segment_angle + min_segment_angle / 2.0 + offset_angle)
    angles_in_radians = np.array(angles) * np.pi / 180.0
    # Corresponding vertices are mapped to [0, 1]^2.
    x = np.column_stack((np.cos(angles_in_radians), np.sin(angles_in_radians)))
    x = (x + 1.0) / 2.0

    # Randomly rotate the vertices.
    theta = np.radians(np.random.rand() * 2 * np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.matrix([[cos, -sin], [sin, cos]])
    x = np.dot(x, rot)

    # Corresponding vertices are mapped to [0, raster_dim]^2
    vertices = x * grid_length

    # If required, move the centre of mass to [raster_dim/2, raster_dim/2]^2.
    if shift_to_mean:
        vertices += raster_dim / 2.0 - vertices.mean(axis=0)

    # Create the bitmask by dividing each cell into subpixels.
    r, dim = subpixel_res, raster_dim
    poly_as_path = path.Path(vertices * r)
    grid_x, grid_y = np.mgrid[0 : dim * r, 0 : dim * r]
    flattened_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    mask = poly_as_path.contains_points(flattened_grid).reshape(dim * r, dim * r)
    mask = np.array(~mask, dtype=np.float32)
    mask = filters.convolve(mask, np.ones((r, r)), mode="constant", cval=1.0)
    return mask[::r, ::r] / (r * r)


def GenerateDataset(
    n_instances,
    n_vertices,
    min_segment_angle,
    scale,
    raster_dim,
    subpixel_res,
    shift_to_mean,
    seed=0,
):
    """Returns a data set of random convex polygons with angle constraints.

  Args:
    n_instances: Integer, number of polygons to generate.
    n_vertices: Integer > 2, number of vertices of the polygon.
    min_segment_angle: Float, minimum angle between two vertices (in degrees).
    scale: Float in (0, 1], downscaling factor for the generated polygon.
    raster_dim: Integer > 1, size of the square grid of pixels to return.
    subpixel_res: Each pixel is split into a grid of size [subpixel_res,
                  subpixel_res] and each subpixel is checked when computing
                  whether the pixel belongs to the polygon.
    shift_to_mean: Boolean, whether to move the center of mass to the center
                   of the raster.
    seed: Random seed to ensure reproducibility.
  Raises:
    ValueError: If the number of vertices is too small, the min angle is too
                large, or scale is not within (0, 1].
  """
    if n_vertices < 3:
        raise ValueError("Need more than 2 vertices.")
    if min_segment_angle > 360.0 / n_vertices:
        raise ValueError("The minimum segment angle is infeasible.")
    if scale < 0 or scale > 1:
        raise ValueError("Scale must be within (0, 1]")
    if raster_dim <= 1:
        raise ValueError("Raster sidelength has to be greater than 1.")
    np.random.seed(seed)
    x = np.zeros((n_instances, raster_dim, raster_dim))
    y = np.ones(n_instances, dtype=np.int8) * n_vertices
    for i in range(n_instances):
        x[i] = GenerateConvexPolygon(
            n_vertices=n_vertices,
            min_segment_angle=min_segment_angle,
            scale=scale,
            raster_dim=raster_dim,
            subpixel_res=subpixel_res,
            shift_to_mean=shift_to_mean,
        )
    ids = np.random.permutation(x.shape[0])
    return x[ids], y[ids]


