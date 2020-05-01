# ME6104 Computer-Aided Design Project
#
# Main Driver
# End to end pipeline to perform surface extractions on pointclouds derived from an STL.


import sys
import argparse
import pointcloud_reader as pc
# import test_sliced_model as pc
import genetic
import vis

# Build up command line interface
parser = argparse.ArgumentParser(description="Genetic Algorithm to recover surfaces from pointclouds")
parser.add_argument("filename", help="STL File")
parser.add_argument("-pt", "--points", type=int, default=100, help="Number of points in the pointcloud")
parser.add_argument("-N", type=int, default=10, help="Number of Bezier control points on one curve")
parser.add_argument("-c", "--cutoff", type=int, default=600, help="Number of seconds to run for")
parser.add_argument("-g", "--gen", type=int, default=10000, help="Number of generations to run for" )
parser.add_argument("-p", "--pop", type=int, default=100, help="Population of genepool" )
parser.add_argument("-e", "--elite", type=float, default=0.2, help="Elitism factor" )
parser.add_argument("-m", "--mutate", type=float, default=0.2, help="Probability of mutation" )
parser.add_argument("-s", "--silent", type=bool, default=False, help="True if no visualization needed")
args = parser.parse_args()

# Step 1 - Convert STL file to pointcloud (simulates 3D printer)
pointcloud = pc.convert(args.filename, args.points)

# Step 2 - Use Genetic Algorithm to transform pointcloud to array of Bezier control points
opt = genetic.genetic(pointcloud, args.N, args.cutoff, args.gen, args.pop, args.elite, args.mutate)
cp = opt.evolve()

# Step 3 - Visualize Bezier surface
if not args.silent:
    vis.plot(cp)
