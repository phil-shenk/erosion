import numpy as np
import matplotlib.pyplot as plt
import voronoi as v
from mayavi import mlab


def calculateGradient(grid):
    grad = np.gradient(grid)
    return grad

width  = 200
length = 200

xs = np.linspace(-np.pi/2,np.pi/2,width)
ys = np.linspace(-np.pi/2,np.pi/2,length)
xx, yy = np.meshgrid(xs,ys)
#grid = xx**2 + yy**2
grid = v.calculateVoronoi(width,length,2,2)
grad = calculateGradient(grid)

# just tuning these for now instead of more well-thought-out gravity things
gradientMultiplier = 0.5
momentumMultiplier = 0.5

minSlope = 0.04
capacityConstant = 8.8

sedimentAccumulationMultiplier = 3.442
sedimentDepositionMultiplier   = 3.442

def changeGrid(x,y,dsediment):
    if not (int(x) < 0 or int(x) >= width or int(y)<0 or int(y)>=length):
        grid[int(x)][int(y)] += dsediment

def getGrid(x,y):
    if (int(x) < 0 or int(x) >= width or int(y)<0 or int(y)>=length):
        return 0
    else:
        return grid[int(x)][int(y)]

def getGradient(x,y):
    global grad
    gx = -1*grad[0][int(x)][int(y)]
    gy = -1*grad[1][int(x)][int(y)]
    if np.abs(gx)>10:
        gx = 10*np.sign(gx)
    if np.abs(gy)>10:
        gy = 10*np.sign(gy)

    return gx,gy

def stepDrop(x,y,dx,dy,liquid,sediment):
    """
    increment the simulation for the given drop
    """
    speed = np.sqrt(dx**2 + dy**2)
    # get grid gradient at drop position ** w.r.t. GRID ** (rise / 1 grid unit 'run')
    dzdx, dzdy = getGradient(x,y)
    # max slope = magnitude of gradient
    slope = np.sqrt(dzdx**2 + dzdy**2)
    
    capacity  = max(slope,minSlope)*speed*liquid*capacityConstant
    dsediment = sediment - capacity

    sediment += dsediment

    # remove mass from gridaccording to how much sediment was removed
    changeGrid(x,y,-1*dsediment)
    
    # move drop according to gradient AND momentum
    dx = -dzdx #xGradientContribution + xMomentumContribution
    dy = -dzdy #yGradientContribution + yMomentumContribution
    
    x += dx
    y += dy

    # update momentum
    xMomentumContribution = dx * momentumMultiplier
    yMomentumContribution = dy * momentumMultiplier

    # deposit sediment according to momentum
    # dsediment = sedimentDepositionMultiplier*(xMomentumContribution**2 + yMomentumContribution**2)
    # deposit exactly the same amount lmao
    dsediment = dsediment
    sediment -= dsediment

    # add that sediment mass back to the grid
    changeGrid(x,y,dsediment)

    return x,y,dx,dy,liquid,sediment


# compute gradient between each drop (drops will only affect gradient of next drop, not itself)

def testDrop100Steps():
    global grad
    x = np.random.randint(width-1)
    y = np.random.randint(length-1)
    dx = (np.random.ranf()*2 - 1) / 10
    dy = (np.random.ranf()*2 - 1) / 10
    liquid = 1.0
    sediment = 0.0
    grad = calculateGradient(grid)
    #for i in range(100):
    i=0
    
    # stop condition kinda arbitrary..
    while i<1000 and ((dx**2+dy**2)>0.0001):
        i+=1
        x,y,dx,dy,liquid,sediment = stepDrop(x,y,dx,dy,liquid,sediment)
        buffer = 3
        
        if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
            break

    # put all sediment where the walker finishes
    changeGrid(x,y,sediment)

for i in range(1000):
    testDrop100Steps()
    if i%100 == 0:
        print(i)
        #plt.clf()
        #plt.imshow(grid, cmap="gist_earth", origin="lower", vmin=0.0,vmax=2.0)
        #plt.colorbar()
        
        #plt.pause(0.001)
        #plt.savefig(str(i)+".png")
        x,y = np.meshgrid(grid)
        mlab.surf(x,y,grid)
        mlab.show()
#plt.show()