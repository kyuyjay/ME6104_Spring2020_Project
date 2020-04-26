import sys
import pointcloud_reader as pc
import genetic


pointcloud = pc.convert(sys.argv[1])
params = {
        "N": int(sys.argv[2])
        }
        
opt = genetic.genetic(params, pointcloud)
opt.evolve()
