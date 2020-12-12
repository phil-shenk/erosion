import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def closestPoint(point, points):
    indexOfClosest = distance.cdist(point, points).argmin()
    return points[indexOfClosest]

def calculateVoronoi(width,length,ypoints,xpoints):
    points = []
    for i in range(0,ypoints):
        for j in range(0,xpoints):
            r1 = np.random.random_sample()*2-1
            r2 = np.random.random_sample()*2-1
            points.append([i+r1,j+r2])

    grid = np.zeros((width,length))
    xarr = np.linspace(0,xpoints,width)
    yarr = np.linspace(0,ypoints,length)
    for i in range(len(xarr)):
        for j in range(len(yarr)):
            point = np.array([[xarr[i],yarr[j]]])
            cp = [closestPoint(point,points)]
            grid[i][j] = distance.cdist(point,cp)

    return np.array(grid)
        
