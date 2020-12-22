import numpy as np
from mayavi import mlab
import voronoi as v
import erosion_functions as e


shared_array = -1.0*v.calculateVoronoi(200,200,2,2)

numdrops = 1000
s = mlab.surf(shared_array, color=(1,1,1), warp_scale=50.0)

@mlab.animate(delay=100)
def anim():
    for i in range(numdrops):
        #e.simulateDropInplace(grid)
        # to visualize @ drop lvl, need to be able to yield:
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
        while j<200:#and meas > 3.0:
            xstephist[j%10] = x
            ystephist[j%10] = y
            meas = np.ptp(xstephist)**2+np.ptp(ystephist)**2
            j+=1
            
            x,y,dx,dy,liquid,sediment = e.stepDropInplace(x,y,dx,dy,liquid,sediment, shared_array)
            buffer = 3
            
            s.mlab_source.scalars = shared_array
            yield

            if (int(x) < 0+buffer or int(x) >= width-buffer or int(y)<0+buffer or int(y)>=length-buffer):
                break
        # put all sediment where the walker finishes
        e.depositSedimentInplace(x,y,sediment, shared_array)
        

anim()
mlab.show()