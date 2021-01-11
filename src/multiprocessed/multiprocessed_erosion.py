import numpy as np
import matplotlib.pyplot as plt
import voronoi as v
import cv2
import time

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process, Pool

#TODO: make mavavi optional with cmd line flag
#from mayavi import mlab

def calculateGradient(grid):
    grad = np.gradient(grid)
    return grad

def depositSedimentFancyDirect(x,y,dsediment, shared_array):
    x = int(x)
    y = int(y)
    width,length = shared_array.shape

    # if point is within grid
    if not (x < 0 or x >= width or y<0 or y>=length):
            
        smoothSpread = True
        if smoothSpread:
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
                            localsum += shared_array[i+x][j+y]

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
                            # set em all to the avg
                            #shared_array[i+x][j+y] = avg + ds

                            # just add the ds portion
                            shared_array[i+x][j+y] = shared_array[i+x][j+y] + ds
        else:
            shared_array[x][y] = shared_array[x][y] + dsediment


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

    return dzdx, -1.0*dzdy

  

def stepDropDirect(x,y,vx,vy,liquid,sediment, shared_array):
    """
    increment the simulation for the given drop
    vx,vy = previous dx,dy (prev velocity, dx/tile, used to calculate momentum)
    """
    prev_speed = np.sqrt(vx**2 + vy**2)
    
    gravity = 10.0
    capacityConstant = 0.5
    momentumMultiplier = 0.9

    dzdx = 0
    dzdy = 0

    # get grid NEGATIVE gradient at drop position ** w.r.t. GRID ** (rise / 1 grid unit 'run')
    dzdx, dzdy = localNegativeGradient(x,y, shared_array)
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
    
    depositSedimentFancyDirect(x,y,-1*dsediment, shared_array)
    
    sediment += dsediment

    return x,y,dx,dy,liquid,sediment

# strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
def simulateDropDictdiffs():
    return

# strategy 3: write effect of drops directly to the shared array and hope for the best
# #TODO: benchmark against dictionary approach
def simulateDropRiskyWrite(shared_array):
    width,length = shared_array.shape

    #difference = np.zeros((width,length))

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
        
        x,y,dx,dy,liquid,sediment = stepDropDirect(x,y,dx,dy,liquid,sediment, shared_array)
        buffer = 3

        if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
            break
    # put all sediment where the walker finishes
    depositSedimentFancyDirect(x,y,sediment, shared_array)
    return

def work_function(shm_name, shm_shape, shm_dtype, numDropsPerFrame):
    # locate shared memory block by name
    shm = SharedMemory(shm_name)
    # Create the np.ndarray from the buffer of the shared memory
    np_array = np.ndarray(shape=shm_shape, dtype=shm_dtype, buffer=shm.buf)
    
    for drop in range(numDropsPerFrame):
        simulateDropRiskyWrite(np_array)
        if(drop%100) == 0:
            print('drop',drop)
    return 'jazz'

def main():
    width  = 400
    length = 400
    gravity = 10.0 # 10.0
    capacityConstant = 2.0
    momentumMultiplier = 0.9 # 0.9

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

    shape, dtype = grid.shape, grid.dtype
    print(shape)
    name = ''
    # create a section of shared memory of the same size as the grid array
    with SharedMemoryManager() as smm:
        #shm = shared_memory.SharedMemory(create=True, size=init_grid.nbytes)
        shm = smm.SharedMemory(size=grid.nbytes)
        name = shm.name
        
        # create another ndarray of the same shape & type as grid, backed by shared memory
        shared_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # copy the initialized data into the shared area of memory
        np.copyto(shared_array, grid)

    numprocesses = 6
    numFrames = 1
    numDropsPerFrame = 100000
    numdrops = numFrames * numprocesses * numDropsPerFrame
    
    start = time.time()

    # frames don't need to exist if we're reading and writing to the same shared array... #TODO remove this for loop
    for i in range(numFrames):
        
        # strategy 1: just make a whole new difference array for each process to return (i expect to be slow)
        #simulateDropArraydiffs(prev_frame)

        # strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
        #simulateDropDictdiffs(prev_frame)
        
        #non pool code for profiling:
        #for drop in range(numDropsPerFrame):
        #    result = simulateDropArraydiffs(prev_grid)
        #    grid += result

        #prev_grid = grid.copy()

        # simulate N (6?) drops with a ProcessPoolExecutor
        #with ProcessPoolExecutor(cpu_count()) as exe:
        
        with ProcessPoolExecutor(numprocesses) as exe:
            fs = [exe.submit(work_function, name, shape, dtype, numDropsPerFrame) for _ in range(numprocesses)]
            for future in as_completed(fs):
                #print('as_completed yeeted,',fs)
                #print('res:',future.result())
                pass
            
        # every 10 frames
        if(i%10==0):
            shm = SharedMemory(name)
            # Create the np.ndarray from the buffer of the shared memory
            grid = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            print('frame #:',i)
            print(grid)

        # export on last frame
        if i == numFrames-1:
            shm = SharedMemory(name)
            # Create the np.ndarray from the buffer of the shared memory
            grid = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
            
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