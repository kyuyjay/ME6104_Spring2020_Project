# Recovering Bezier Surfaces from Pointclouds

End to end pipeline to convert pointclouds to Bezier surfaces. Optional STL to pointcloud reader included.

## Installation
No installation needed. 

Requirements
- Python 3.8.2
- numpy 1.18.1
- numpy-stl 2.10.1
- matplotlib 3.1.2

Written on Vim 8.0

## Usage
```
python3 driver.py [-h] [-pt POINTS] [-N N] [-c CUTOFF] [-g GEN] [-p POP] [-e ELITE] [-m MUTATE] [-s SILENT] filename 
```
positional arguments:
  filename              
    STL file location

optional arguments:
    -h, --help  show this help message and exit
    -pt POINTS, --points POINTS
        Number of points in the pointcloud
    -N N                  
        Number of Bezier control points on one curve
    -c CUTOFF, --cutoff CUTOFF
        Number of seconds to run for
    -g GEN, --gen GEN     
        Number of generations to run for
    -p POP, --pop POP     
        Population of genepool
    -e ELITE, --elite ELITE
        Elitism factor
    -m MUTATE, --mutate MUTATE
        Probability of mutation
    -s SILENT, --silent SILENT
        True if no visualization needed
