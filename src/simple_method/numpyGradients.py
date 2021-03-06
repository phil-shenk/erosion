import numpy as np
import matplotlib.pyplot as plt
import voronoi as v

import time

#TODO: make mavavi optional with cmd line flag
from mayavi import mlab

def calculateGradient(grid):
    grad = np.gradient(grid)
    return grad

width  = 200
length = 200
gravity = 10.0

xs = np.linspace(-np.pi/2,np.pi/2,width)
ys = np.linspace(-np.pi/2,np.pi/2,length)
xx, yy = np.meshgrid(xs,ys)
#grid = xx**2 + yy**2
grid = v.calculateVoronoi(width,length,2,2)
grad = calculateGradient(grid)

# just tuning these for now instead of more well-thought-out gravity things
gradientMultiplier = 0.9
momentumMultiplier = 0.9

maxCapacity = 0.03
capacityConstant = 2.0

def changeGrid(x,y,dsediment):
    x = int(x)
    y = int(y)

    # if point is within grid
    if not (x < 0 or x >= width or y<0 or y>=length):
        neighbors = np.empty(9)
        neighbors[:] = np.NaN
        validcount = 0
        localsum = 0.0

        #TODO: check if I can do this step faster with numpy functions

        # iterate over neighbors, use them to calculate mean if they are valid
        for i in range(-1,2):
            if i+x>=0 and i+x < width:
                for j in range(-1,2):
                    if j+y>=0 and j+y<length:
                        # (i+x, j+y) is a valid neighbor, increment validcount and 
                        validcount += 1
                        localsum += grid[i+x][j+y]

        # divide the sediment among the valid tiles
        ds = dsediment / float(validcount)
        # calculate the avg height of the valid tiles
        avg = localsum / float(validcount)
        # now iterate over neighbors to update their values
        for i in range(-1,2):
            if i+x>=0 and i+x < width:
                for j in range(-1,2):
                    if j+y>=0 and j+y<length:
                        # (i+x, j+y) is a valid tile, set its value to the local avg + the appropriate chunk of sediment
                        #grid[i+x][j+y] = avg + ds # spread evenly over neighbors #TODO: try other vals
                        grid[i+x][j+y] = (grid[i+x][j+y] + avg)/2.0 + ds
def getGrid(x,y):
    if (x < 0 or x >= width or y<0 or y>=length):
        return 0.0
    else:
        return grid[x][y]

def getNegativeGradient(x,y):
    global grad
    gx = -1*grad[0][int(x)][int(y)]
    gy = -1*grad[1][int(x)][int(y)]

    #"""artificially clip gradient if it's too big
    clipval = 0.02
    if np.abs(gx)>clipval:
        gx = clipval*np.sign(gx)
    if np.abs(gy)>clipval:
        gy = clipval*np.sign(gy)

    return gx,gy

def stepDrop(x,y,vx,vy,liquid,sediment):
    """
    increment the simulation for the given drop
    vx,vy = previous dx,dy (prev velocity, dx/tile, used to calculate momentum)
    """
    prev_speed = np.sqrt(vx**2 + vy**2)
    
    dzdx = 0
    dzdy = 0
    # get grid NEGATIVE gradient at drop position ** w.r.t. GRID ** (rise / 1 grid unit 'run')
    dzdx, dzdy = getNegativeGradient(x,y)

    # move drop according to gradient #TODO: incorporate momentum
    dx = dzdx*gravity + vx*momentumMultiplier
    dy = dzdy*gravity + vy*momentumMultiplier

    new_speed = np.sqrt(dx**2 + dy**2)

    # max slope = magnitude of gradient
    slope = np.sqrt(dzdx**2 + dzdy**2)

    capacity  = slope*new_speed*liquid*capacityConstant

    dsediment = capacity - sediment
    
    # change position for next drop
    prev_x = x
    prev_y = y
    x += dx
    y += dy
        
    depositSediment(x,y,-1*dsediment)
    
    sediment += dsediment

    return x,y,dx,dy,liquid,sediment

def depositSediment(x,y,sediment):
    x = int(x)
    y = int(y)
    changeGrid(x,y,sediment)
    
numdrops = 5000
s = mlab.surf(grid, warp_scale="auto")
@mlab.animate(delay=10)
def anim():
    for i in range(numdrops): 
        global grad # only for numpy gradients
        global grid
        x = np.random.randint(width-1)
        y = np.random.randint(length-1)
        dx = (np.random.ranf()*2.0 - 1.0) / 10.0
        dy = (np.random.ranf()*2.0 - 1.0) / 10.0
        liquid = 1.0
        sediment = 0.0
        # #TODO: definitely should just calculate gradient locally instead of using whole grid..
        grad = calculateGradient(grid)
        #for j in range(100):
        j=0
        meas = 100.0
        # maintain history of steps to determine stop time
        xstephist = np.array([0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0])
        ystephist = np.array([100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0])

        # step the drop until stop
        while j<200 and meas > 3.0:
            xstephist[j%10] = x
            ystephist[j%10] = y
            meas = np.ptp(xstephist)**2+np.ptp(ystephist)**2

            j+=1
            
            x,y,dx,dy,liquid,sediment = stepDrop(x,y,dx,dy,liquid,sediment)
            buffer = 3
            
            if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
                break
            
        # put all sediment where the walker finishes
        depositSediment(x,y,sediment)
        
        # every 50 drops
        if(i%100==0):
            print(i)
            # update mesh vertices for 3d rendering
            s.mlab_source.scalars = grid
            yield
        if i == numdrops-1:
            end = time.time()
            print("ELAPSED:",end-start)

start = time.time()
anim()
mlab.show()
