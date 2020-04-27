import numpy as np
import pandas as pd
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# def set_axes_equal(ax):
#     """ Make axes of 3D plot have equal scale so that spheres appear as spheres,
#         cubes as cubes, etc..  This is one possible solution to Matplotlib's
#         ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
#
#         Input:
#             ax: a matplotlib axis, e.g., as output from mplot3d.Axes3D(fig)
#     """
#     # Define Current Axes Limits
#     x_limits = ax.get_xlim3d()
#     y_limits = ax.get_ylim3d()
#     z_limits = ax.get_zlim3d()
#
#     # Calculate Scaling
#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)
#
#     # The plot bounding box is a sphere in the sense of the infinity
#     # norm, hence I call half the max range the plot radius.
#     plot_radius = 0.5*max([x_range, y_range, z_range])
#
#     # Set New Axes Limits
#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def triangle_area_multi(v1, v2, v3):
    """ Compute area of triangles in face-vertex mesh format

        Inputs:
            v1, v2, v3: (N, 3) ndarrays.
            - vi represents the xyz coordinate of one triangle vertex
            - v1[n], v2[n], v3[n] are the 3 vertices of the nth triangle
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
    return res.astype(np.float32)


def convert(filename):
    # Load mesh from STL file
    my_mesh = mesh.Mesh.from_file(filename)
    size = 3
    numpoints = 100

    # # Get XYZ coordinates of all vertices and delete duplicate (shared) vertices
    # v_with_dupes = np.vstack((my_mesh.v0, my_mesh.v1, my_mesh.v2))
    # v_unique = np.unique(v_with_dupes, axis=0)

    # # Create a new plot for raw mesh data
    # figure = plt.figure()
    # ax1 = mplot3d.Axes3D(figure)

    # # Add the points from the STL mesh model to the plot
    # x_raw = v_unique[:, 0]
    # y_raw = v_unique[:, 1]
    # z_raw = v_unique[:, 2]
    # ax1.scatter(x_raw, y_raw, z_raw, c='b', marker='o', s=size)

    # # Auto scale the plot and label axes
    # set_axes_equal(ax1)
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')

    # # Show the plot to the screen
    # plt.savefig('stl_cloud_raw.png')
    # plt.show()

    # Generate point cloud from mesh data
    result_xyz = generate_pt_cloud(my_mesh.v0, my_mesh.v1, my_mesh.v2, numpoints)

    # # Save data as pandas DataFrame (not necessary, but data is presented more cleanly)
    # result = pd.DataFrame()
    # result["x"] = result_xyz[:, 0]
    # result["y"] = result_xyz[:, 1]
    # result["z"] = result_xyz[:, 2]

    # # Create new plot for generated point cloud
    # fig2 = plt.figure()
    # ax2 = mplot3d.Axes3D(fig2)

    # # Add the points from the cloud to the plot
    # x = result_xyz[:, 0]
    # y = result_xyz[:, 1]
    # z = result_xyz[:, 2]
    # ax2.scatter(x, y, z, c='r', marker='o', s=size)

    # # Auto scale the plot and label axes
    # set_axes_equal(ax2)
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')

    # # Show the plot to the screen
    # plt.savefig('stl_cloud_' + str(numpoints) + 'points_' + str(size) + 'markersize.png')
    # plt.show()

    # Return pointcloud
    return result_xyz

