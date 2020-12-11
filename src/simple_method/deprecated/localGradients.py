import numpy as np
import matplotlib.pyplot as plt

width  = 100
length = 100

# just tuning these for now instead of more well-thought-out gravity things
gradientMultiplier = 1.0
momentumMultiplier = 0.0

sedimentAccumulationMultiplier = 0.0
sedimentDepositionMultiplier   = 0.0

#grid = np.ones((width,length))
xs = np.linspace(-np.pi/2,np.pi/2,100)
ys = np.linspace(-np.pi/2,np.pi/2,100)
xx, yy = np.meshgrid(xs,ys)
grid = xx**2 + yy**2

def changeGrid(x,y,dsediment):
    if not (int(x) < 0 or int(x) >= width or int(y)<0 or int(y)>=length):
        grid[int(x)][int(y)] += dsediment

def getGrid(x,y):
    if (int(x) < 0 or int(x) >= width or int(y)<0 or int(y)>=length):
        return 0
    else:
        return grid[int(x)][int(y)]
    

def localGradient(x,y):
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

    dzdx = 2*(z6-z4) + z3 - z1 + z9 - z7
    dzdy = 2*(z2-z8) + z3 - z9 + z1 - z7
    return dzdx, dzdy

# place 'raindrop' at random point
# pick up some quantity of sediment from point
# maybe sediment amount should depend on steepness? or momentum?
# walk raindrop in downward-ish direction (but also taking momentum into account)
# deposit sediment (if shallow enough? slow enough?)
#    deposition wikipedia: when enough KE of the fluid is lost, sediment is deposited

# what to keep track of for each particle:
#  - momentum
#  - amount of sediment
#  - amount of liquid

def testDrop100Steps():
    x = np.random.randint(width-1)
    y = np.random.randint(length-1)
    dx = 0.0#(np.random.ranf()*2 - 1) / 100
    dy = 0.0#(np.random.ranf()*2 - 1) / 100
    liquid = 1.0
    sediment = 0.0
    for i in range(100):
        x,y,dx,dy,liquid,sediment = stepDrop(x,y,dx,dy,liquid,sediment)
        buffer = 3
        if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
            break
        plt.clf()
        plt.imshow(grid, cmap="gray", origin="lower", vmin=0,vmax=1.0)
        plt.colorbar()
        plt.arrow(x,y,dx*5,dy*5, width=2)
        #plt.scatter(y,x)
        plt.pause(0.0001)
    

def stepDrop(x,y,dx,dy,liquid,sediment):
    """
    increment the simulation for the given drop
    """
    # get grid gradient at drop position
    dzdx, dzdy = localGradient(x,y)
    xGradientContribution = dzdx*gradientMultiplier
    yGradientContribution = dzdy*gradientMultiplier
    xMomentumContribution = dx * momentumMultiplier
    yMomentumContribution = dy * momentumMultiplier
    #print(xGradientContribution,yGradientContribution,xMomentumContribution,yMomentumContribution)
    
    # pick up sediment according to slope and momentum.
    #TODO: make this conserve mass..
    dsediment = sedimentAccumulationMultiplier*(xGradientContribution**2 + yGradientContribution**2 + xMomentumContribution**2 + yMomentumContribution**2)
    sediment += dsediment

    # remove mass from gridaccording to how much sediment was removed
    changeGrid(x,y,-1*dsediment)
    
    # move drop according to gradient AND momentum
    dx = xGradientContribution + xMomentumContribution
    dy = yGradientContribution + yMomentumContribution
    
    x += dx
    y += dy

    # update momentum
    xMomentumContribution = dx * momentumMultiplier
    yMomentumContribution = dy * momentumMultiplier

    # deposit sediment according to momentum
    dsediment = sedimentDepositionMultiplier*(xMomentumContribution**2 + yMomentumContribution**2)
    sediment -= dsediment

    # add that sediment mass back to the grid
    changeGrid(x,y,dsediment)

    return x,y,dx,dy,liquid,sediment


# try later:
# split raindrops into smaller volumes that go in different directions
# gather different types/hardness of sediments for different layers of heightmap
# have 'momentum' actually depend on mass, not just velocity
for i in range(2):
    testDrop100Steps()
plt.show()
