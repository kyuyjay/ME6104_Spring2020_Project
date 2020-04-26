import sys
import time
import math
import random
import numpy as np
import pandas as pd
import bezier as bz
import vis

class genetic:
    # Main driver class to hold algorithm parameters
    def __init__(self,params,pointcloud,cutoff=600,pop=10,elite=0.2,mutate=0.3):
        self.mating_pool = []
        self.weights = []
        self.cutoff = cutoff

        self.hyp = {
                "N": params["N"],
                "POPULATION": pop,
                "ELITIST FACTOR": elite,
                "MUTATION": mutate,
                "GENERATIONS": 10000
                }
        self.hyp["ELITISM"] = min(1,int(round(self.hyp["ELITIST FACTOR"] * self.hyp["POPULATION"])))
        print("Hyperparameters")
        for param in self.hyp:
            print("{}: {}".format(param,self.hyp[param]))

        self.pointcloud = pointcloud
        self.points = len(pointcloud)
        self.max = np.max(pointcloud, axis=1) 
        self.min = np.min(pointcloud, axis=1) 

    # Each gene represents a control point
    class gene:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        # Calculate the distance from this location to another location
        def distance(self,dest):
            return np.linalg.norm([dest.x - self.x,dest.y - self.y])

    # Each DNA is a collection of N x N Bezier points that form a Bezier surface
    class DNA:
        def __init__(self,N,points,max_,min_):
            self.N = N
            self.strand = bz.gen_points(N, max_, min_)
            self.u = np.random.rand(points)
            self.v = np.random.rand(points)

        # Calculate the least squares difference between this surface and the given pointcloud
        def fitness(self, pointcloud):
            self.rmse = 0
            for i,point in enumerate(pointcloud):
                approx = bz.bezier_point(self.strand, self.u[i], self.v[i])
                self.rmse = np.linalg.norm((point - approx),2)
            return

        # Mutates each gene by swapping gene positions in the DNA with the given probability
        def mutate(self,N,MUTATION,points,max_,min_):
            for i in range(self.N):
                for j in range(self.N):
                    chance = random.random()
                    if chance <= MUTATION: 
                        point = bz.gen_point(max_,min_)
                        for d in range(3):
                            self.strand[d,i,j] = point[d]
            for i in range(points):
                chance = random.random()
                if chance <= MUTATION:
                    self.u[i] = random.random()
                    self.v[i] = random.random()
            return

    # Create the initial population from the gene pool by generating DNA
    def populate(self):
        for i in range(self.hyp["POPULATION"]):
            self.mating_pool.append(self.DNA(self.hyp["N"],self.points,self.max,self.min))
            self.mating_pool[i].fitness(self.pointcloud)
            self.weights.append(1 / self.mating_pool[i].rmse)
        self.mating_pool, self.weights = (list(x) for x in zip(*sorted(zip(self.mating_pool,self.weights),key=lambda x: x[0].rmse)))
        return

    # Select 2 parents from the population with a weighted probability
    def select(self):
        parents = random.choices(self.mating_pool,weights=self.weights,k=2)
        return parents
 
    # Mate two DNA by selecting different portions of them while preserving the validity of the solution
    def crossbreed(self,host,partner):
        N = self.hyp["N"]
        half = math.floor(N/2)
        child = self.DNA(self.hyp["N"],self.points,self.max,self.min)
        for i in range(3):
            child.strand[i, 0:half, :] = host.strand[i, 0:half, :]
            child.strand[i, half+1:N-1, :] = partner.strand[i, half+1:N-1, :]
        half = math.floor(self.points/2)
        child.u[0:half] = host.u[0:half]
        child.u[half+1:] = partner.u[half+1:]
        child.v[0:half] = host.v[0:half]
        child.v[half+1:] = partner.v[half+1:]
        return child

    # Driver method to conduct selection and mating for the whole population
    def survive(self):
        N = self.hyp["N"]
        POPULATION = self.hyp["POPULATION"]
        ELITISM = self.hyp["ELITISM"]
        MUTATION = self.hyp["MUTATION"]
        next_gen = self.mating_pool[0:ELITISM]
        for i in range(POPULATION - ELITISM):
            parents = self.select()
            child = self.crossbreed(parents[0],parents[1])
            child.mutate(N,MUTATION,self.points,self.max,self.min)
            next_gen.append(child)
        self.weights.clear()
        for DNA in next_gen:
            DNA.fitness(self.pointcloud)
            self.weights.append(1 / DNA.rmse)
        next_gen, self.weights = (list(x) for x in zip(*sorted(zip(next_gen,self.weights),key=lambda x: x[0].rmse)))
        print(next_gen)
        print(self.weights)
        self.mating_pool = next_gen
        return

    # Main driver program to run the whole algorithm for a fixed number of generation or until timeout
    def evolve(self):
        start_time = time.time()
        self.populate()
        curr_min = sys.maxsize
        unchanged = 0
        for i in range(self.hyp["GENERATIONS"]):
            print("Generation " + str(i))
            self.survive()
            if self.mating_pool[0].rmse < curr_min:
                unchanged = 0
                curr_min = self.mating_pool[0].rmse
                best_DNA = self.mating_pool[0]
            unchanged = unchanged + 1
            print("Best RMSE " + str(self.mating_pool[0].rmse))
            if unchanged > 1000:
                break
            if self.cutoff is not None:
                if (time.time() - start_time) > self.cutoff:
                    break
        vis.plot(self.mating_pool[0].strand)
        return 
