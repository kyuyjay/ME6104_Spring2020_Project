import numpy as np
from math import comb
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Plotting Helpers
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


# Generate num number of discrete points
def gen_mesh(num):
    u = np.empty(num)
    for i in range(num):
        u[i] = i * (1 / (num - 1))
    return u


# Generate Bernstein coefficients for a set of discrete points for one k
def bernstein_poly(u, k, n):
    poly = np.empty(np.size(u))
    for i, j in enumerate(u):
        poly[i] = comb(n, k) * (j ** k) * (1 - j) ** (n - k)
    return poly


# Determine surface using num_u * num_v set of discrete points and cp control points
def bezier_surface(cpp, num_u, num_v):
    u = gen_mesh(num_u)
    v = gen_mesh(num_v)
    # dim = np.size(cpp, 0)
    # n_u = np.size(cpp, 1)
    # n_v = np.size(cpp, 2)
    dim = cpp.shape[2]
    n_u = cpp.shape[0]
    n_v = cpp.shape[1]
    poly = np.zeros((dim, num_u, num_v))
    for coord in range(dim):
        for i in range(n_u):
            for j in range(n_v):
                # Degree of polynomial one less than number of control points
                poly[coord] = poly[coord] + np.outer(bernstein_poly(u, i, n_u - 1), bernstein_poly(v, j, n_v - 1)) * \
                              cpp[i, j, coord]
    return poly


# Add surface to figure
def plot(cpp, num_u=100, num_v=100):
    print("Plotting Surface...")
    fig = plt.figure()
    fig.suptitle("Surface Generated by Genetic Algorithm")
    surface = bezier_surface(cpp, num_u, num_v)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X=surface[0], Y=surface[1], Z=surface[2])
    set_axes_equal(ax)
    plt.show()


# Point Cloud Helpers
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


# 3D Control Points from Stacked Convex Hulls
def create_surf(cloud, numslices):
    # Find min and max values along x-axis
    cloud = cloud[np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0]))]
    xmin = np.min(result[:, 0])
    xmax = np.max(result[:, 0])

    # Create vector of x coordinates of slices along model
    slc, dx = np.linspace(xmin, xmax, num=numslices, retstep=True)

    # Hulls
    hulls = []

    # Iterate through slices
    for i in range(len(slc)):
        # Extract x coordinate of current slice
        xcoord = slc[i]
        # Bin point cloud data within a distance from current x coordinate
        if i == 0:
            lbound = xcoord
            ubound = xcoord + dx / 2
        elif i == len(slc):
            lbound = xcoord - dx / 2
            ubound = xcoord
        else:
            lbound = xcoord - dx / 2
            ubound = xcoord + dx / 2
        # Get indices of points inside current bin
        pt_idx = getneighbors(cloud[:, 0], lbound, ubound)
        # Project points onto YZ plane at current x coordinate
        y_points = cloud[pt_idx[0], 1]
        z_points = cloud[pt_idx[0], 2]
        # Construct Convex Hull of Current Slice
        pnts = np.append(y_points, z_points)
        pnts = np.reshape(pnts, (-1, 2), order='F')
        current_hull = convex_hull(pnts)
        # Construct 3D Version of Convex Hull
        x_cp = np.ones(current_hull.shape[0]) * xcoord
        y_cp = current_hull[:, 0]
        z_cp = current_hull[:, 1]
        c_hull = np.append(np.append(x_cp, y_cp, axis=0), z_cp, axis=0)
        c_hull = np.reshape(c_hull, (-1, 3), order='F')
        # Store Convex Hull of Current Slice
        hulls.append(c_hull.tolist())
    return hulls


def pad_surf(surface):
    maxlength = 0
    for i in range(0, len(surface)):
        if len(surface[i]) > maxlength:
            maxlength = len(surface[i])
    padded_surf = []
    for i in range(0, len(surface)):
        cp = surface[i]
        cp_padded = pad_vec(cp, maxlength)
        padded_surf.append(cp_padded)
    return padded_surf


def pad_vec(p, final_length):
    p = np.asarray(p, dtype='float')
    for i in range(0, final_length):
        if p.shape[0] >= final_length:
            break
        idx = np.random.choice(np.arange(p.shape[0] - 1))
        midpoint = (p[idx, :] + p[idx + 1, :])/2
        p = np.insert(p, idx + 1, midpoint, axis=0)
    return p


# 2D Convex Hull Helpers
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


def orientation(p, q, r):
    val = (q[1] - p[1])*(r[0] - q[0]) - (q[0] - p[0])*(r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def convex_hull(pts):
    n = pts.shape[0]
    if n < 3:
        return np.empty([1, 2])
    # Find leftmost point in set of points
    pt_l = 0
    for i in range(1, pts.shape[0]):
        if pts[i, 0] < pts[pt_l, 0]:
            pt_l = i
    '''
    Start from leftmost point, keep moving counterclockwise until reach the start point 
    again. This loop runs O(h) times where h is number of points in result or output.
    '''
    vertex = pts[pt_l, :]
    pt_p = pt_l
    count = 1
    while True:
        # Add current point to result
        if count == 1:
            hull = vertex
        else:
            hull = np.append(hull, pts[pt_p, :], axis=0)
        # print('Hull ' + str(count) + ':\n')
        # print(hull)
        '''
        Search for a point 'q' such that  
        orientation(p, x, q) is counterclockwise  
        for all points 'x'. The idea is to keep  
        track of last visited most counterclock- 
        wise point in q. If any point 'i' is more  
        counterclock-wise than q, then update q.
        '''
        pt_q = (pt_p + 1) % n
        for i in range(0, n):
            # If i is more counterclockwise than current q, then update q
            if orientation(pts[pt_p, :], pts[i, :], pts[pt_q, :]) == 2:
                pt_q = i
        pt_p = pt_q
        count += 1
        if pt_p == pt_l:
            break
    hull = np.insert(hull, len(hull), pts[pt_l, :])
    hull = np.reshape(hull, (-1, 2))
    # hull = np.insert(hull, n, pts[pt_l, :], axis=0)
    return hull


# Bezier Helper Functions
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
my_mesh = mesh.Mesh.from_file('Arm_NoHand.STL')
npoints = 1000
nslices = 50
result = generate_pt_cloud(my_mesh.v0, my_mesh.v1, my_mesh.v2, npoints)

# Generate Control Point Surface
surf_hulls = create_surf(result, nslices)
# print(surf_hulls)
surf_cp = pad_surf(surf_hulls)
# print(surf_cp)

# Plot Bezier Surface
control_surface = np.asarray(surf_cp, dtype='float')
# print(control_surface)
plot(control_surface)
