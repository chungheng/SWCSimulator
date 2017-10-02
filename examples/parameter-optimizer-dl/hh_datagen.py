import matplotlib
from model import *
from swcsim import *
import matplotlib.pyplot as plt
import numpy as np
from StringIO import StringIO
import re
import random
from scipy.io import savemat
import pickle
import multiprocessing


def make_data(m):
    #print(m)
    swcString = \
    "id type x y z radius parent_id\n" \
    "0  2    3 1 0 1      -1\n" \
    "1  1    2 1 0 1       0\n" \
    "2  1    1 2 0 1       1\n" \
    "3  1    0 2 0 1       2\n" \
    "4  1    1 0 0 1       1\n" \
    "5  1    0 0 0 1       4\n"
    dataString = StringIO(re.sub(' +',' ',swcString))
    np.random.seed()
    dt = 1e-5
    dur = 0.5
    t = np.arange(0,dur,dt)

    neuron = None
    dataString = StringIO(re.sub(' +',' ',swcString))
    neuron = SWCNeuron(dataString, swcKwargs={'delimiter':' '}, ActiveModel = HodgkinHuxley())

    i_n = 0
    #print(neuron.graph.nodes(data = True))
    for i in range(len(neuron.graph.nodes(data = True))):
        #print(neuron.graph.nodes(data = True)[i][1]['model'].__class__.__name__)
        if neuron.graph.nodes(data = True)[i][1]['model'].__class__.__name__ == 'HodgkinHuxley':
            neuron.graph.nodes(data = True)[i][1]['params'] = neuron.graph.nodes(data = True)[i][1]['model'].Params(
                                                            E_Na=50. + np.random.random() * 60. - 30.,
                                                            E_K=-77. + np.random.random() * 60. - 30.,
                                                            E_l=-54.387 + np.random.random() - 30.,
                                                            g_Na=120. + np.random.random() - 30.,
                                                            g_K=36. + np.random.random() * 40. - 20.,
                                                            g_l=0.3 + np.random.random() * 0.4 - 0.2,)
            i_n += len(list(neuron.graph.nodes(data = True)[i][1]['params']))
            p = list(neuron.graph.nodes(data = True)[i][1]['params'])

    print((neuron.graph.nodes(data = True)[0][1]['params']))
    V = np.zeros((neuron.graph.number_of_nodes(), len(t)))
    I = np.zeros((neuron.graph.number_of_edges(), len(t)))
    P = np.zeros((i_n, len(t)))

    stimulus = 2000.0 * np.random.poisson(lam=0.25, size=t.shape)

    sim = neuron.create_simulator(dt, [3,5], stimulus, steps=len(stimulus))

    for i, _ in enumerate(sim):
        for j,n in enumerate(neuron.graph.nodes_iter()):
            V[j,i] = neuron.graph.node[n]['states'].V
        for j, (u,v,d) in enumerate(neuron.graph.edges_iter(data=True)):
            I[j,i] = d['I']
        P[:,i] = np.array(p)
    return [V, I, P]

    #return 0

if __name__ == "__main__":
    pool = multiprocessing.Pool(40)
    M = 200
    print(M)
    out = pool.map(make_data, range(M))
    print('Processing complete...')
    print(out)
    pickle.dump( out, open( "hh_data.p", "wb" ), 2 )
