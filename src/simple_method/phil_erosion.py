import numpy as np
import matplotlib.pyplot as plt
import voronoi as v
import cv2

import time

#TODO: make mavavi optional with cmd line flag
from mayavi import mlab

def calculateGradient(grid):
    grad = np.gradient(grid)
    return grad

width  = 400
length = 400
gravity = 10.0
capacityConstant = 2.0
momentumMultiplier = 0.9

use_numpy_gradients = False
showdrops = False

xs = np.linspace(-np.pi/2,np.pi/2,width)
ys = np.linspace(-np.pi/2,np.pi/2,length)
xx, yy = np.meshgrid(xs,ys)
#grid = xx**2 + yy**2
mapsrc = None
mapsrc = "heightmap1.png"
if mapsrc == None:
    grid = -1.0*v.calculateVoronoi(width,length,5,5)
else:
    grid = cv2.imread("../../heightmaps/"+mapsrc, cv2.IMREAD_GRAYSCALE)
    grid = grid * (1.0/256.0)
prev_grid = grid.copy()

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
                        # set em all to the avg, kindof like a smearing brush (** this has little artefacts when the drop changes direction.. ** #TODO improve that)
                        grid[i+x][j+y] = avg + ds

def getGrid(x,y):
    if (x < 0 or x >= width or y<0 or y>=length):
        return 0.0
    else:
        return prev_grid[x][y]

def localNegativeGradient(x,y):
    """
    find gradient at given point using surrounding 8 adjacent cells \n
    z1 | z2 | z3 \n
    z4 | z5 | z6 \n
    z7 | z8 | z9 \n
    im gonna try something like this: \n
    dz/dx = 2*(z6-z4) + z3 - z1 + z9 - z7 \n
    dz/dy = 2*(z2-z8) + z3 - z9 + z1 - z7 \n
    not using np.gradient because I think that does the whole grid.. could run it on just the 3x3 subgrid though
    """
    #TODO: handle edges better
    #TODO: actually i think i should just kill drops when they're near the edge

    # i'm sure there's a more elegant way to do this
    ix = int(x)
    iy = int(y)
    z1 = getGrid(ix-1,iy-1)
    z2 = getGrid(ix,iy-1)
    z3 = getGrid(ix+1,iy-1)
    z4 = getGrid(ix-1,iy)
    #z5 = getGrid(ix,iy)
    z6 = getGrid(ix+1,iy)
    z7 = getGrid(ix-1,iy+1)
    z8 = getGrid(ix,iy+1)
    z9 = getGrid(ix+1,iy+1)
    #z1,z2,z3,z4,z6,z7,z8,z9)
    dzdx = -1.0*(2*(z6-z4) + z3 - z1 + z9 - z7) / 8.0
    dzdy = -1.0*(2*(z2-z8) + z3 - z9 + z1 - z7) / 8.0

    # clip em
    clipval = 0.02
    if np.abs(dzdx)>clipval:
        dzdx = clipval*np.sign(dzdx)
    if np.abs(dzdy)>clipval:
        dzdy = clipval*np.sign(dzdy)

    #print("loc",dzdx,dzdy)
    return dzdx, -1.0*dzdy

def localNumpyNegativeGradient(x,y):
    if(int(x) <= 0 or int(x) >= width-1 or int(y)<=0 or int(y)>=length-1):
        return 0.0,0.0
    localgrid = np.array(prev_grid[int(x)-1:int(x)+2, int(y)-1:int(y)+2])
    localgrad = np.gradient(localgrid)
    dzdx = -1.0*localgrad[0][1][1]
    dzdy = -1.0*localgrad[1][1][1]
    return dzdx,dzdy

def getNegativeGradient(x,y):
    global grad
    gx = -1*grad[0][int(x)][int(y)]
    gy = -1*grad[1][int(x)][int(y)]
    #"""artificially clip gradient
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
    if(use_numpy_gradients):
        dzdx, dzdy = getNegativeGradient(x,y)
    else:
        dzdx, dzdy = localNegativeGradient(x,y)
        #dzdx, dzdy = localNumpyNegativeGradient(x,y)

    # move drop according to gradient #TODO: incorporate momentum
    dx = dzdx*gravity + vx*momentumMultiplier
    dy = dzdy*gravity + vy*momentumMultiplier

    new_speed = np.sqrt(dx**2 + dy**2)

    #print(prev_speed,new_speed)

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
    
numdrops = 100000
copyRate = 10
s = mlab.surf(grid, warp_scale=25.0)
@mlab.animate(delay=10)
def anim():
    for i in range(numdrops): 
        global grad # only for numpy gradients
        global grid
        global prev_grid
        x = np.random.randint(width-1)
        y = np.random.randint(length-1)
        dx = (np.random.ranf()*2.0 - 1.0) / 10.0
        dy = (np.random.ranf()*2.0 - 1.0) / 10.0
        liquid = 1.0
        sediment = 0.0
        # #TODO: definitely should just calculate gradient locally instead of using whole grid..
        # or build up a list of modifications, and apply ONLy those after each drop
        if(use_numpy_gradients):
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
            
            if(showdrops):
                s.mlab_source.scalars = grid
                yield

            if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
                break

        # put all sediment where the walker finishes
        depositSediment(x,y,sediment)
        
        #hmm... maybe just save previous gradient locally too? #TODO
        # only copy every n drops
        if(i%copyRate==0):
            prev_grid = grid.copy()
        
        
        # every 50 drops
        if(i%100==0):
            print(i)
            # update mesh vertices for 3d rendering
            s.mlab_source.scalars = grid
            yield
        if i == numdrops-1:
            end = time.time()
            print("ELAPSED:",end-start)
            cv2.imwrite("../../heightmaps/eroded_"+str(numdrops)+"_"+mapsrc, grid*256.0)
 
start = time.time()
anim()
mlab.show()

