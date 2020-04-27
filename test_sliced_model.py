import numpy as np
from math import comb
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import NearestNeighbors
import networkx as nx


def set_axes_equal(ax):
    """ Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input:
            ax: a matplotlib axis, e.g., as output from mplot3d.Axes3D(fig)
    """
    # Define Current Axes Limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate Scaling
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    # Set New Axes Limits
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def triangle_area_multi(v1, v2, v3):
    """ Compute area of triangles in face-vertex mesh format

        Inputs:
            v1, v2, v3: (N, 3) ndarrays.
            - vi represents the xyz coordinate of one triangle vertex
            - v1[n], v2[n], v3[n] are the 3 vertices of the nth triangle

        Outputs:
            total surface area of all triangles in mesh
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def generate_pt_cloud(v1, v2, v3, n):
    # Mesh Sampling - Sample n mesh faces.  Ensure mesh faces with larger areas are more likely to be sampled.
    areas = triangle_area_multi(v1, v2, v3)
    probabilities = areas / areas.sum()
    weighted_rand_ind = np.random.choice(range(len(areas)), size=n, p=probabilities)
    v1 = v1[weighted_rand_ind]
    v2 = v2[weighted_rand_ind]
    v3 = v3[weighted_rand_ind]

    # Mesh sampling (continued) - Choose points inside the sampled mesh faces to form the point cloud
    # Barycentric Coordinates
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1
    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    w = 1 - (u + v)
    res = (v1 * u) + (v2 * v) + (v3 * w)
    return res


def getneighbors(vec, low, high):
    """ Return indices of points inside local bin

        Inputs:
            vec: vector of values
            low: lower bound of local bin
            high: upper bound of local bin

        Outputs:
            indices of values in vec between the lower and upper bounds
    """
    return np.where(np.logical_and(vec >= low, vec <= high))


def get_nearest_neighbors(ycoords, zcoords):
    """ Return indices of points inside local bin

        Found this algorithm online somewhere

        Inputs:
            ycoords: vector of y coordinates
            zcoords: vector of z coordinates

        Outputs:
            ordered yz pairs in nearest neighbor configuration
    """
    points = np.c_[ycoords, zcoords]
    clf = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points)
    g = clf.kneighbors_graph()
    t = nx.from_scipy_sparse_matrix(g)
    paths = [list(nx.dfs_preorder_nodes(t, ind)) for ind in range(len(points))]
    mindist = np.inf
    minidx = 0
    for ind in range(len(points)):
        p = paths[ind]         # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = ind
    opt_order = paths[minidx]
    yy = ycoords[opt_order]
    zz = zcoords[opt_order]
    return yy, zz


def bernstein_poly(uval, k, n):
    """ Return n-th degree bernstein polynomial value for u with index k

        Inputs:
            uval - local parameter coordinate, ranges from 0 to 1
            k - order index, ranges from 0 to n
            n - degree of polynomial, equal to number of control points - 1

        Outputs:
            The n-th degree bernstein polynomial value for u with index k
    """
    return comb(n, k) * (uval ** k) * (1 - uval) ** (n - k)


def bezier_curve(cp, num_plot_points):
    """ Return points for bezier curve

        Inputs:
            cp - 2D (3 x "number of control points") array of control points
            num_plot_points - number of discrete points to draw

        Outputs:
            curve: 2D (3 x num_plot_points) array of points that represent the curve
    """
    num_ctrl_pts = cp.shape[1]
    u_vec = np.linspace(0, 1, num_plot_points)
    bernstein_values = np.zeros((num_plot_points, num_ctrl_pts))
    for i in range(0, num_plot_points):
        for j in range(0, num_ctrl_pts):
            bernstein_values[i, j] = bernstein_poly(u_vec[i], j, num_ctrl_pts - 1)
    curve = np.zeros((3, num_plot_points))
    for a in range(0, 3):
        coordinate = np.zeros((num_plot_points,))
        for b in range(0, num_plot_points):
            coordinate[b] = np.dot(bernstein_values[b, :], cp[a, :])
        curve[a, :] = coordinate
    return curve


# Load first STL model from file
my_mesh = mesh.Mesh.from_file('Arm_NoHand.stl')
numpoints = 1000
result = generate_pt_cloud(my_mesh.v0, my_mesh.v1, my_mesh.v2, numpoints)

# Find min and max values along x-axis
result = result[np.lexsort((result[:, 2], result[:, 1], result[:, 0]))]
xmin = np.min(result[:, 0])
xmax = np.max(result[:, 0])

# Create vector of x coordinates of slices along model
slc, dx = np.linspace(xmin, xmax, num=25, retstep=True)

# Create figure to visualize 'sliced' model
fig3 = plt.figure()
ax3 = mplot3d.Axes3D(fig3)

# Iterate through slices
for i in range(len(slc)):
    # Extract x coordinate of current slice
    xcoord = slc[i]
    # Bin point cloud data within a distance from current x coordinate
    if i == 0:
        lbound = xcoord
        ubound = xcoord + dx/2
    elif i == len(slc):
        lbound = xcoord - dx/2
        ubound = xcoord
    else:
        lbound = xcoord - dx/2
        ubound = xcoord + dx/2
    # Get indices of points inside current bin
    pt_idx = getneighbors(result[:, 0], lbound, ubound)
    # Project 5 points onto YZ plane at current x coordinate
    x_points = np.ones(len(pt_idx[0])) * xcoord
    y_points = result[pt_idx[0], 1]
    z_points = result[pt_idx[0], 2]
    # Plot current '2D' scatter
    ax3.scatter(x_points, y_points, z_points, c='k', marker='o', s=1)
    # Arrange points in nearest-neighbor configuration
    y_cp, z_cp = get_nearest_neighbors(y_points, z_points)
    x_cp = np.ones(y_cp.shape[0]) * xcoord
    # Construct Control Points
    c_p = np.vstack((x_cp, y_cp, z_cp))
    # Construct Bezier Curve
    bez_curve = bezier_curve(c_p, 20)
    # Plot Bezier Curve
    ax3.plot3D(bez_curve[0, :], bez_curve[1, :], bez_curve[2, :])

set_axes_equal(ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
plt.show()
