import sys
import random
import numpy as np
from math import comb

# Generate N x N number of control points from existing pointcloud points
def gen_points(N, pointcloud):
    cp = np.zeros((3,N,N))
    points = np.zeros((3,N*N))
    for i in range(N):
        for j in range(N):
            point = random.choice(pointcloud)
            # point[0] = point[0] + random.uniform(-5,5)
            # point[1] = point[1] + random.uniform(-5,5)
            # point[2] = point[2] + random.uniform(-5,5)
            points[0,i*N + j] = point[0]
            points[1,i*N + j] = point[1]
            points[2,i*N + j] = point[2]
    points = points[:,np.argsort(points[0,:])]
    for i in range(N):
        row = points[:,i*N:(i+1)*N]
        row = row[:,np.argsort(row[1,:])]
        cp[:,i,:] = row
    return cp

# Generate a control point from existing pointcloud point
def gen_point(pointcloud):
    point = random.choice(pointcloud)
    x = point[0] + random.uniform(-5,5)
    y = point[1] + random.uniform(-5,5)
    z = point[2] + random.uniform(-5,5)
    return (x, y, z)

def sort_cp(N,cp):
    points = np.zeros((3,N*N))
    for d in range(3):
        points[d] = cp[d].flatten()[np.argsort(cp[0].flatten())]
    for i in range(N):
        row = points[:,i*N:(i+1)*N]
        row = row[:,np.argsort(row[1,:])]
        cp[:,i,:] = row
    return cp

# Calculate Bernstein coefficient
def bernstein_poly(u, k, n):
    poly = comb(n, k) * (u ** k) * (1 - u) ** (n - k)
    return poly

# Construct a bezier point given u and v parameters and N x N control points
def bezier_point(cp, u, v):
    N = len(cp)
    point = np.zeros(3)
    for i in range(N):
        for j in range(N):
            point[0] = point[0] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[0, i, j])
            point[1] = point[1] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[1, i, j])
            point[2] = point[2] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[2, i, j])
    return point
