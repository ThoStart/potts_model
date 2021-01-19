import vc
import timeit
import pickle
import numpy as np

temperatures = [0.01, 0.5, 1, 1.5, 2.0, 2.5, 5]# np.arange(0.01)
states = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
samples = np.arange(0, 10,1)

start = timeit.default_timer()
for state in states:
    states_exp = [state]
    Pottsruns10k10x10st = vc.perform_tests(temperatures, states_exp, samples)
    filename = f'Pottsruns10k10x10st{state}'
    print('experiment one')
    outfile = open(filename,'wb')
    pickle.dump(Pottsruns10k10x10st, outfile)
    print('pickle one done')
    outfile.close()

stop = timeit.default_timer()
print('Time: ', stop - start) 


