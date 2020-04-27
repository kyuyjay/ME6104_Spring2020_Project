import sys
import argparse
import pointcloud_reader as pc
import genetic
import vis

# Build up command line interface
parser = argparse.ArgumentParser(description="Genetic Algorithm to recover surfaces from pointclouds")
parser.add_argument("filename", help="STL File")
parser.add_argument("-N", type=int, default=30, help="Number of Bezier control points on one curve")
parser.add_argument("-c", "--cutoff", type=int, default=600, help="Number of seconds to run for")
parser.add_argument("-g", "--gen", type=int, default=10000, help="Number of generations to run for" )
parser.add_argument("-p", "--pop", type=int, default=500, help="Population of genepool" )
parser.add_argument("-e", "--elite", type=float, default=0.2, help="Elitism factor" )
parser.add_argument("-m", "--mutate", type=float, default=0.2, help="Probability of mutation" )
args = parser.parse_args()

# Step 1 - Convert STL file to pointcloud (simulates 3D printer)
pointcloud = pc.convert(sys.argv[1])

# Step 2 - Use Genetic Algorithm to transform pointcloud to array of Bezier control points
opt = genetic.genetic(pointcloud, args.N, args.cutoff, args.gen, args.pop, args.elite, args.mutate)
cp = opt.evolve()

# Step 3 - Visualize Bezier surface
vis.plot(cp)
