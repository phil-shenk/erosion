import numpy as np

def simdrop(arrr):
    arrr += 0.1
    return arrr

arrray = np.zeros(100)
print("sent",arrray)
simdrop(arrray)
print("then",arrray)

# so it DOES pass them by value
# nice
