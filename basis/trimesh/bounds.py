import numpy as np
import time

from .util import unitize, transformation_2D
from .constants import log
from .grouping import group_vectors
from .points import transform_points, project_to_plane
from .geometry import rotation_2D_to_3D

try:
    from scipy.spatial import ConvexHull
except ImportError:
    log.warning('Scipy import failed!')


def oriented_bounds_2D(points):
    '''
    Find an oriented bounding box for a set of 2D points.

    Arguments
    ----------
    points: (n,2) float, 2D points
    
    Returns
    ----------
    transform: (3,3) float, homogenous 2D transformation matrix to move the input set of 
               points to the FIRST QUADRANT, so no value is negative. 
    rectangle: (2,) float, size of extents once input points are transformed by transform
    '''
    c = ConvexHull(np.asanyarray(points))
    # (n,2,3) line segments
    hull = c.points[c.simplices]
    # (3,n) points on the hull to check against
    dot_test = c.points[c.vertices].reshape((-1, 2)).T
    edge_vectors = unitize(np.diff(hull, axis=1).reshape((-1, 2)))
    perp_vectors = np.fliplr(edge_vectors) * [-1.0, 1.0]
    bounds = np.zeros((len(edge_vectors), 4))
    for i, edge, perp in zip(range(len(edge_vectors)),
                             edge_vectors,
                             perp_vectors):
        x = np.dot(edge, dot_test)
        y = np.dot(perp, dot_test)
        bounds[i] = [x.min(), y.min(), x.max(), y.max()]

    extents = np.diff(bounds.reshape((-1, 2, 2)), axis=1).reshape((-1, 2))
    area = np.product(extents, axis=1)
    area_min = area.argmin()

    offset = -bounds[area_min][0:2]
    theta = np.arctan2(*edge_vectors[area_min][::-1])

    transform = transformation_2D(offset, theta)
    rectangle = extents[area_min]

    return transform, rectangle


def oriented_bounds(mesh, angle_tol=1e-6):
    '''
    Find the oriented bounding box for a Trimesh 

    Arguments
    ----------
    mesh: Trimesh object
    angle_tol: float, angle in radians that OBB can be away from minimum volume
               solution. Even with large values the returned extents will cover
               the mesh albeit with larger than minimal volume. 
               Larger values may experience substantial speedups. 
               Acceptable values are floats >= 0.0.
               The default is small (1e-6) but non-zero.

    Returns
    ----------
    to_origin: (4,4) float, transformation matrix which will move the center of the
               bounding box of the input mesh to the origin. 
    extents: (3,) float, the extents of the mesh once transformed with to_origin
    '''
    # this version of the cached convex hull has normals pointing in 
    # arbitrary directions (straight from qhull)
    # using this avoids having to compute the expensive corrected normals
    # that mesh.convex_hull uses since normal directions don't matter here
    hull = mesh._convex_hull_raw
    vectors = group_vectors(hull.face_normals,
                            angle=angle_tol,
                            include_negative=True)[0]
    min_volume = np.inf
    tic = time.time()
    for i, normal in enumerate(vectors):
        projected, to_3D = project_to_plane(hull.vertices,
                                            plane_normal=normal,
                                            return_planar=False,
                                            return_transform=True)
        height = projected[:, 2].ptp()
        rotation_2D, box = oriented_bounds_2D(projected[:, 0:2])
        volume = np.product(box) * height
        if volume < min_volume:
            min_volume = volume
            rotation_2D[0:2, 2] = 0.0
            rotation_Z = rotation_2D_to_3D(rotation_2D)
            to_2D = np.linalg.inv(to_3D)
            extents = np.append(box, height)
    to_origin = np.dot(rotation_Z, to_2D)
    transformed = transform_points(hull.vertices, to_origin)
    box_center = (transformed.min(axis=0) + transformed.ptp(axis=0) * .5)
    to_origin[0:3, 3] = -box_center

    log.debug('oriented_bounds checked %d vectors in %0.4fs',
              len(vectors),
              time.time() - tic)
    return to_origin, extents


from .geometry import plane_transform
from .util import vector_to_spherical, is_shape, grid_linspace
from .transformations import translation_matrix, transform_points, spherical_matrix
from .convex import hull_points
from .nsphere import minimum_nsphere
from time import time as now
from scipy import optimize
def minimum_cylinder(obj, sample_count=6, angle_tol=.001):
    """
    Find the approximate minimum volume cylinder which contains
    a mesh or a a list of points.
    Samples a hemisphere then uses scipy.optimize to pick the
    final orientation of the cylinder.
    A nice discussion about better ways to implement this is here:
    https://www.staff.uni-mainz.de/schoemer/publications/ALGO00.pdf
    Parameters
    ----------
    obj : trimesh.Trimesh, or (n, 3) float
      Mesh object or points in space
    sample_count : int
      How densely should we sample the hemisphere.
      Angular spacing is 180 degrees / this number
    Returns
    ----------
    result : dict
      With keys:
        'radius'    : float, radius of cylinder
        'height'    : float, height of cylinder
        'transform' : (4,4) float, transform from the origin
                      to centered cylinder
    """

    def volume_from_angles(spherical, return_data=False):
        """
        Takes spherical coordinates and calculates the volume
        of a cylinder along that vector
        Parameters
        ---------
        spherical : (2,) float
           Theta and phi
        return_data : bool
           Flag for returned
        Returns
        --------
        if return_data:
            transform ((4,4) float)
            radius (float)
            height (float)
        else:
            volume (float)
        """
        to_2D = spherical_matrix(*spherical,
                                 axes='rxyz')
        projected = transform_points(hull,
                                     matrix=to_2D)
        height = projected[:, 2].ptp()

        try:
            center_2D, radius = minimum_nsphere(projected[:, :2])
        except BaseException:
            # in degenerate cases return as infinite volume
            return np.inf

        volume = np.pi * height * (radius ** 2)
        if return_data:
            center_3D = np.append(center_2D, projected[
                                             :, 2].min() + (height * .5))
            transform = np.dot(np.linalg.inv(to_2D),
                               translation_matrix(center_3D))
            return transform, radius, height
        return volume

    # we've been passed a mesh with radial symmetry
    # use center mass and symmetry axis and go home early
    if hasattr(obj, 'symmetry') and obj.symmetry == 'radial':
        # find our origin
        if obj.is_watertight:
            # set origin to center of mass
            origin = obj.center_mass
        else:
            # convex hull should be watertight
            origin = obj.convex_hull.center_mass
        # will align symmetry axis with Z and move origin to zero
        to_2D = plane_transform(
            origin=origin,
            normal=obj.symmetry_axis)
        # transform vertices to plane to check
        on_plane = transform_points(
            obj.vertices, to_2D)
        # cylinder height is overall Z span
        height = on_plane[:, 2].ptp()
        # center mass is correct on plane, but position
        # along symmetry axis may be wrong so slide it
        slide = translation_matrix(
            [0, 0, (height / 2.0) - on_plane[:, 2].max()])
        to_2D = np.dot(slide, to_2D)
        # radius is maximum radius
        radius = (on_plane[:, :2] ** 2).sum(axis=1).max() ** 0.5
        # save kwargs
        result = {'height': height,
                  'radius': radius,
                  'transform': np.linalg.inv(to_2D)}
        return result

    # get the points on the convex hull of the result
    hull = hull_points(obj)
    if not is_shape(hull, (-1, 3)):
        raise ValueError('Input must be reducable to 3D points!')

    # sample a hemisphere so local hill climbing can do its thing
    samples = grid_linspace([[0, 0], [np.pi, np.pi]], sample_count)

    # if it's rotationally symmetric the bounding cylinder
    # is almost certainly along one of the PCI vectors
    if hasattr(obj, 'principal_inertia_vectors'):
        # add the principal inertia vectors if we have a mesh
        samples = np.vstack(
            (samples,
             vector_to_spherical(obj.principal_inertia_vectors)))

    tic = [now()]
    # the projected volume at each sample
    volumes = np.array([volume_from_angles(i) for i in samples])
    # the best vector in (2,) spherical coordinates
    best = samples[volumes.argmin()]
    tic.append(now())

    # since we already explored the global space, set the bounds to be
    # just around the sample that had the lowest volume
    step = 2 * np.pi / sample_count
    bounds = [(best[0] - step, best[0] + step),
              (best[1] - step, best[1] + step)]
    # run the local optimization
    r = optimize.minimize(volume_from_angles,
                          best,
                          tol=angle_tol,
                          method='SLSQP',
                          bounds=bounds)

    tic.append(now())
    log.debug('Performed search in %f and minimize in %f', *np.diff(tic))

    # actually chunk the information about the cylinder
    transform, radius, height = volume_from_angles(r['x'], return_data=True)
    result = {'transform': transform,
              'radius': radius,
              'height': height}
    return result
