import numpy as np
cimport numpy as np

cdef unsigned long long int maxval
cdef unsigned long long int toal

cdef int k 
cdef np.ndarray arr
cdef int total 

def test(int total):
    maxval = 100
    arr = np.arange(maxval)

    for k in arr:
        total = total + k
        print(total)