import numpy as np
import matplotlib.pyplot as plt
import voronoi as v
import cv2
import multiprocessing
import time

#TODO: make mavavi optional with cmd line flag
from mayavi import mlab

def calculateGradient(grid):
    grad = np.gradient(grid)
    return grad



def depositSediment(x,y,dsediment, prev_frame, difference):
    """
    x,y = grid position to deposit/erode sediment
    sediment = amount by which to change the height
    prev_frame = numpy array reference, for reading only. calculations are based on this
    difference = numpy array reference, for output. the difference that should be applied to calculate the next frame 
    """
    x = int(x)
    y = int(y)
    width,length = prev_frame.shape

    # if point is within grid
    if not (x < 0 or x >= width or y<0 or y>=length):
        # divide the sediment among the valid tiles
        difference[x][y] = dsediment
                        
    # could return difference, but it's passed by reference so the original is modified I think that might add calls
    #return difference
def depositSedimentSemiFancy(x,y,dsediment, prev_frame, difference):
    """
    x,y = grid position to deposit/erode sediment
    sediment = amount by which to change the height
    prev_frame = numpy array reference, for reading only. calculations are based on this
    difference = numpy array reference, for output. the difference that should be applied to calculate the next frame 
    """
    x = int(x)
    y = int(y)
    width,length = prev_frame.shape

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
                        localsum += prev_frame[i+x][j+y]

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
                        ##TODO: this is a bit convoluted.. ideally I won't need the prev_frame at all in depositSediment
                        difference[i+x][j+y] = avg + ds - prev_frame[i+x][j+y]
                        
    # could return difference, but it's passed by reference so the original is modified I think that might add calls
    #return difference

def depositSedimentFancy(x,y,dsediment, prev_frame, difference):
    """
    x,y = grid position to deposit/erode sediment
    sediment = amount by which to change the height
    prev_frame = numpy array reference, for reading only. calculations are based on this
    difference = numpy array reference, for output. the difference that should be applied to calculate the next frame 
    """
    x = int(x)
    y = int(y)
    width,length = prev_frame.shape

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
                        localsum += prev_frame[i+x][j+y]

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
                        ##TODO: this is a bit convoluted.. ideally I won't need the prev_frame at all in depositSediment
                        difference[i+x][j+y] = avg + ds - prev_frame[i+x][j+y]
                        
    # could return difference, but it's passed by reference so the original is modified I think that might add calls
    #return difference

def getGrid(x,y,prev_frame):
    if (x < 0 or x >= prev_frame.shape[0] or y<0 or y>=prev_frame.shape[1]):
        return 0.0
    else:
        return prev_frame[x][y]

def localNegativeGradient(x,y, prev_frame):
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
    z1 = getGrid(ix-1,iy-1, prev_frame)
    z2 = getGrid(ix,iy-1, prev_frame)
    z3 = getGrid(ix+1,iy-1, prev_frame)
    z4 = getGrid(ix-1,iy, prev_frame)
    #z5 = getGrid(ix,iy, prev_frame)
    z6 = getGrid(ix+1,iy, prev_frame)
    z7 = getGrid(ix-1,iy+1, prev_frame)
    z8 = getGrid(ix,iy+1, prev_frame)
    z9 = getGrid(ix+1,iy+1, prev_frame)
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

def stepDrop(x,y,vx,vy,liquid,sediment, prev_frame, difference):
    """
    increment the simulation for the given drop
    vx,vy = previous dx,dy (prev velocity, dx/tile, used to calculate momentum)
    """
    prev_speed = np.sqrt(vx**2 + vy**2)
    
    gravity = 10.0
    capacityConstant = 2.0
    momentumMultiplier = 0.9

    dzdx = 0
    dzdy = 0

    # get grid NEGATIVE gradient at drop position ** w.r.t. GRID ** (rise / 1 grid unit 'run')
    dzdx, dzdy = localNegativeGradient(x,y, prev_frame)
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
    
    depositSedimentFancy(x,y,-1*dsediment, prev_frame, difference)
    
    sediment += dsediment

    return x,y,dx,dy,liquid,sediment

# strategy 1: just make a whole new difference array for each process to return (i expect to be slow)
def simulateDropArraydiffs(prev_frame):
    width,length = prev_frame.shape

    difference = np.zeros((width,length))

    x = np.random.randint(width-1)
    y = np.random.randint(length-1)
    dx = (np.random.ranf()*2.0 - 1.0) / 10.0
    dy = (np.random.ranf()*2.0 - 1.0) / 10.0
    liquid = 1.0
    sediment = 0.0

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
        
        x,y,dx,dy,liquid,sediment = stepDrop(x,y,dx,dy,liquid,sediment, prev_frame, difference)
        buffer = 3

        if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
            break
    # put all sediment where the walker finishes
    depositSedimentFancy(x,y,sediment, prev_frame, difference)
    return difference

# strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
def simulateDropDictdiffs():
    return

def main():
    width  = 400
    length = 400
    gravity = 10.0
    capacityConstant = 2.0
    momentumMultiplier = 0.9

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

    numFrames = 25
    numDropsPerFrame = 2000
    numdrops = numFrames * numDropsPerFrame
    """
    s = mlab.surf(grid, warp_scale=25.0)
    @mlab.animate(delay=10)
    def anim():
        # for each frame (serially), 
        # we'll calculate a bunch of drops on the previous frame (in parallel)
        # and combine the results to use as the next frame
        for i in range(numFrames):
            global grad # only for numpy gradients
            global grid
            global prev_grid

            pool = multiprocessing.Pool()
            result = pool.map(simulateDropArraydiffs, prev_grid)
            print(result)
            grid += result
            # strategy 1: just make a whole new difference array for each process to return (i expect to be slow)
            #simulateDropArraydiffs(prev_frame)

            # strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
            #simulateDropDictdiffs(prev_frame)

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
    """

    start = time.time()

    for i in range(numFrames):
        prev_grids = []
        #pool code
        """
        # horrifying:
        for drop in range(numDropsPerFrame):
            # maybe ill get lucky and this just does it by reference..
            prev_grids.append(prev_grid)
        
        print("POOLING:",len(prev_grids))
        pool = multiprocessing.Pool(2)
        
        print("REALLy POOLING:")
        results = pool.map(simulateDropArraydiffs, prev_grids)
        
        pool.close()
        print("pool closed")
        pool.join()
        print("pool joined")
        
        #print("res:\n",sum(results))
        grid += sum(results)
        
        # strategy 1: just make a whole new difference array for each process to return (i expect to be slow)
        #simulateDropArraydiffs(prev_frame)

        # strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
        #simulateDropDictdiffs(prev_frame)
        """
        #non pool code for profiling:
        for drop in range(numDropsPerFrame):
            result = simulateDropArraydiffs(prev_grid)
            grid += result

        prev_grid = grid.copy()
        
        # every 50 drops
        if(i%100==0):
            print(i)

        if i == numFrames-1:
            end = time.time()
            print("ELAPSED:",end-start)
            output_filename = "../../heightmaps/mp_eroded_"+str(numdrops)+"_"+mapsrc
            cv2.imwrite(output_filename, grid*256.0)
            print('wrote to',output_filename)

    #end = time.time()
    #print("ELAPSED :::", end-start)
    #anim()
    #mlab.show()

if __name__ == '__main__':
    main()
    #freeze_support()