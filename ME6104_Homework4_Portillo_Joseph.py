import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Inputs:
#   u - local parameter coordinate, ranges from 0 to 1
#   k - order index, ranges from 0 to n
#   n - degree of polynomial, equal to number of control points - 1
# Outputs:
#   The n-th degree bernstein polynomial value for u with index k
def bernstein_poly(u, k, n):
    return scipy.special.comb(n, k) * (u ** k) * ((1 - u) ** (n - k))


# Inputs:
#   cp - 2D (3 x "number of control points") array of control points
#   num_plot_points - number of discrete points to draw
# Outputs:
#   curve - 2D (3 x num_plot_points) array of points that represent the curve
def bezier_curve(cp, num_plot_points):
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


# Inputs:
#   cpp - 3D (3 x "# of control points in u direction" x "# of control points in v direction") array of control points
#   num_u - number of discrete points to draw in u direction
#   num_v - number of discrete points to draw in v direction
# Outputs:
#   surface - 3D (3 x num_u x num_v) array of points that represent the surface
def bezier_surface(cpp, num_u, num_v):
    num_ctrl_pts_u = cpp.shape[1]
    num_ctrl_pts_v = cpp.shape[2]
    u_vec = np.linspace(0, 1, num_u)
    v_vec = np.linspace(0, 1, num_v)
    bernstein_values_u = np.zeros((num_u, num_ctrl_pts_u))
    for i in range(0, num_u):
        for j in range(0, num_ctrl_pts_u):
            bernstein_values_u[i, j] = bernstein_poly(u_vec[i], j, num_ctrl_pts_u - 1)
    bernstein_values_v = np.zeros((num_ctrl_pts_v, num_v))
    for i in range(0, num_v):
        for j in range(0, num_ctrl_pts_v):
            bernstein_values_v[j, i] = bernstein_poly(v_vec[i], j, num_ctrl_pts_v - 1)
    surface = np.zeros((3, num_u, num_v))
    for a in range(0, 3):
        grid = np.zeros((num_u, num_v))
        for b in range(0, num_u):
            for c in range(0, num_v):
                grid[b, c] = np.matmul((np.matmul(bernstein_values_u[b, :], cpp[a, :, :])), bernstein_values_v[:, c])
        surface[a, :, :] = grid
    return surface


# Plot Bezier Curve
c_p = np.array([[3, 4, 6, 7.2, 11, 14], [10, 7, 6, 7.5, 7, 6], [1, 2, 3, 3.5, 2, 1]])
bez_curve = bezier_curve(c_p, 100)
fig1 = plt.figure()
ax1 = mplot3d.Axes3D(fig1)
X = bez_curve[0, :]
Y = bez_curve[1, :]
Z = bez_curve[2, :]
ax1.plot3D(X, Y, Z)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.savefig('bezier_curve.png')
plt.show()


# Plot Bezier Surface
c_pp = np.array([[[1, 3, 6, 8], [1, 3, 6, 8], [1, 3, 6, 8], [1, 3, 6, 8]],
                 [[20, 21, 22, 23], [17, 17, 17, 17], [14, 14, 14, 14], [11, 11, 11, 11]],
                 [[2, 5, 4, 3], [2, 6, 5, 5], [2, 6, 5, 4], [2, 3, 4, 3]]])
bez_surf = bezier_surface(c_pp, 100, 100)
fig2 = plt.figure()
ax2 = mplot3d.Axes3D(fig2)
for u in range(0, 100):
    current_curve = bez_surf[:, u, :]
    X = current_curve[0, :]
    Y = current_curve[1, :]
    Z = current_curve[2, :]
    ax2.plot3D(X, Y, Z)
for v in range(0, 100):
    current_curve = bez_surf[:, :, v]
    X = current_curve[0, :]
    Y = current_curve[1, :]
    Z = current_curve[2, :]
    ax2.plot3D(X, Y, Z)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
plt.savefig('bezier_surface.png')
plt.show()
