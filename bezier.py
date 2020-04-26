import sys
import random
import numpy as np

from math import comb

def gen_points(N, max_, min_):
    cp = np.zeros((3,N,N))
    for i in range(N):
        for j in range(N):
            cp[0,i,j] = random.uniform(max_[0] - 1, min_[0] + 1)
            cp[1,i,j] = random.uniform(max_[1] - 1, min_[1] + 1)
            cp[2,i,j] = random.uniform(max_[2] - 1, min_[2] + 1)
    return cp
            
def gen_point(max_, min_):
    x = random.uniform(max_[0] - 1, min_[0] + 1)
    y = random.uniform(max_[1] - 1, min_[1] + 1)
    z = random.uniform(max_[2] - 1, min_[2] + 1)
    return (x, y, z)

def bernstein_poly(u, k, n):
    poly = comb(n,k) * (u**k) * (1-u)**(n-k)
    return poly

def bezier_point(cp, u, v):
    N = len(cp)
    point = np.zeros(3)
    for i in range(N):
        for j in range(N):
            point[0] = point[0] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[0,i,j])
            point[1] = point[1] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[1,i,j])
            point[2] = point[2] + (bernstein_poly(u, i, N) * bernstein_poly(v, j, N) * cp[2,i,j])
    return point

