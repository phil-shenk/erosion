import numpy as np
import time

#TODO make this a class so I can initialize parameters

def depositSedimentInplace(x,y,dsediment, shared_array):
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
#                            shared_array[i+x][j+y] = avg + ds

                            # just add the ds portion
                            #shared_array[i+x][j+y] = shared_array[i+x][j+y] + ds

                            # set it to avg of avg and previous 
                            shared_array[i+x][j+y] = ((avg + shared_array[i+x][j+y])/2.0) + ds
        else:
            shared_array[x][y] = shared_array[x][y] + dsediment

def getHeight(x,y,array):
    if (x < 0 or x >= array.shape[0] or y<0 or y>=array.shape[1]):
        return 0.0
    else:
        return array[x][y]

def localNegativeGradient(x,y, array):
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
    #TODO: actually i think i should just kill drops when they're near the edge, which i do anyway (at edge) and it would erase some of the checks

    # i'm sure there's a more elegant way to do this
    ix = int(x)
    iy = int(y)
    z1 = getHeight(ix-1,iy-1, array)
    z2 = getHeight(ix,iy-1, array)
    z3 = getHeight(ix+1,iy-1, array)
    z4 = getHeight(ix-1,iy, array)
    #z5 = getHeight(ix,iy, array)
    z6 = getHeight(ix+1,iy, array)
    z7 = getHeight(ix-1,iy+1, array)
    z8 = getHeight(ix,iy+1, array)
    z9 = getHeight(ix+1,iy+1, array)
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

def stepDropInplace(x,y,vx,vy,liquid,sediment, shared_array):
    """
    increment the simulation for the given drop
    vx,vy = previous dx,dy (prev velocity, dx/tile, used to calculate momentum)
    """
    prev_speed = np.sqrt(vx**2 + vy**2)
    
    # to see in animation
    #time.sleep(0.001)

    gravity = 20.0
    capacityConstant = 3.0
    momentumMultiplier = 0.1

    dzdx = 0
    dzdy = 0

    # get grid NEGATIVE gradient at drop position ** w.r.t. GRID ** (rise / 1 grid unit 'run')
    dzdx, dzdy = localNegativeGradient(x,y, shared_array)

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
    #x += dx
    #y += dy

    # move an entire grid step (independent of speed) to avoid hops / redundant 
    # to do this, how about I just get a normalized dv and get the x and y of that
    ndx = dx / new_speed
    ndy = dy / new_speed
    x += ndx
    y += ndy

    depositSedimentInplace(x,y,-1*dsediment, shared_array)
    
    sediment += dsediment

    # not returning normalized ndx,ndy since we want to preserve concept of speed
    return x,y,dx,dy,liquid,sediment

# strategy 2: make a dictionary of the changed values {(x,y): difference}, return the dictionary, then combine them all
# TODO: try

# strategy 3: write effect of drops directly to the shared array and hope for the best
# #TODO: benchmark against dictionary approach
def simulateDropInplace(shared_array):
    width,length = shared_array.shape

    x = np.random.randint(width-1)
    y = np.random.randint(length-1)
    dx = (np.random.ranf()*2.0 - 1.0) / 10.0
    dy = (np.random.ranf()*2.0 - 1.0) / 10.0
    liquid = 1.0
    sediment = 0.0

    j=0
    meas = 100.0

    # #TODO shrink these arrays (or use different measure) to improve performance
    # maintain history of steps to determine stop time
    xstephist = np.array([0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0])
    ystephist = np.array([100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0,100.0,0.0])

    # step the drop until stop
    while j<200 and meas > 3.0:
        xstephist[j%10] = x
        ystephist[j%10] = y
        meas = np.ptp(xstephist)**2+np.ptp(ystephist)**2
        
        j+=1
        
        x,y,dx,dy,liquid,sediment = stepDropInplace(x,y,dx,dy,liquid,sediment, shared_array)
        buffer = 3

        if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
            break
    # put all sediment where the walker finishes
    depositSedimentInplace(x,y,sediment, shared_array)
    return