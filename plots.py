import vc
import timeit
import pickle
import numpy as np
import matplotlib.pyplot as plt
import vc
# open pickle file
temperatures = [0.01, 0.5, 1, 1.5, 2.0, 2.5, 5]# np.arange(0.01)
states = [2, 3, 4, 5, 6, 7, 8, 9, 10]
samples = np.arange(0,5,1)

lattice_size = 10
timesteps = 100
i1 = lattice_size**2 + 1
i2 = lattice_size**2 * timesteps - 1

plt.figure(1)
for state in states:
    filename = f'Pottsruns10k10x10st{state}'
    infile = open(filename,'rb')
    runs = pickle.load(infile)
    infile.close()
    l = []
    k = []
    for temp in temperatures:
        fh = 0
        sh = 0
        satisfaction = 0
        for run in samples:
#                 fh += runs[f'run{state,temp,run}'][i1][i2]
            sh += runs[f'run{state,temp,run}'][i1+1][i2]
            satisfaction += runs[f'run{state,temp,run}'][i1+2][i2]
        l.append(sh/len(samples)/200)
        k.append(satisfaction/len(samples)/lattice_size**2)
    plt.plot(temperatures, l, label=f'{state} states')
#         plt.plot(temperatures, k, label=f'{state}ver2')

plt.ylabel('Phase')
plt.xlabel('Temperature')
plt.title('Potts encoding')
plt.legend()
plt.savefig('Potts 1')

temperatures = [0.01, 0.5, 1, 1.5, 2.0, 2.5, 5]# np.arange(0.01)
states = [2, 3, 4, 5, 6, 7, 8, 9, 10]
samples = np.arange(0,5,1)

lattice_size = 10
timesteps = 100
i1 = lattice_size**2 + 1
i2 = lattice_size**2 * timesteps - 1

plt.figure(2)
global_satisfactions = []
for state in states:
    filename = f'Pottsruns10k10x10st{state}'
    infile = open(filename,'rb')
    runs = pickle.load(infile)
    infile.close()
    G_init, J, system_hamiltonian, A, fm = vc.simulate(1, state)
    global_statisfactions = []
    global_satisfactions_err = []
    for temp in temperatures:
        fh = 0
        sh = 0
        satisfaction = 0
        avg = []
        for run in samples:
            avg.append(vc.check(np.array(runs[f'run{state,temp,run}'].iloc[-1, :100]), A, J)[-1])
        global_statisfactions.append(np.mean(avg))
        global_satisfactions_err.append(np.std(avg))
#         print(global_statisfactions)
    # plt.plot(temperatures, global_statisfactions, label=f'{state} states')
    plt.errorbar(temperatures, global_statisfactions, yerr=global_satisfactions_err, label=f'{state} states', uplims=True, lolims=True)
plt.ylabel('Number of finished VCs')
plt.xlabel('Temperature')
plt.title('Potts encoding')
plt.legend()
plt.savefig('Potts2')