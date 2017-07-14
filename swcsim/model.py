"""
Abstract Neuron Class
"""
from abc import ABCMeta, abstractmethod
from math import exp
from recordclass import recordclass

class Neuron(object):
    """
    Base Neuron Class

    Attributes:
    -----------
    Params: a recordclass
        Parameters

    States:

    Methods:
    --------

    Notes:
    ------
    """
    __metaclass__ = ABCMeta
    States = recordclass('states', ('V', 'I'))
    Params = recordclass('params', ())
    initStates = States(V=-65.,I=0.)

    @classmethod
    @abstractmethod
    def ode(cls, states, params=None):
        """
        Define the set of ODEs of the state variables.

        Parameters:
        -----------
        states: an instance of the recordclass 'States'
            State variables at a given time step.

        params: an instance of the recordclass 'Params'
            Parameters used in the ODE. If None, the default parameters defined
            in 'defaultParams' will be used.

        Returns:
        --------
        gradStates : an instance of the recordclass 'States'
            The derivative of the state variables.

        Notes:
        ------
        """
        params = params or cls.defaultParams
        pass

    @classmethod
    def update(cls, dt, states, params=None):
        """
        Update the state varibles of the neuron model.

        This function first calls the ode() to compute the derivative of
        the state variables, and then uses the forward Euler method to update
        the state variables. The current fed into ode() is the sum of
        external current and the synaptic current.

        Parameters:
        -----------
        dt : float
            Time step of the numerical method.

        states: an instance of the recordclass 'States'
            State variables at a given time step.

        params: an instance of the recordclass 'Params'
            Parameters used in the ODE. If None, the default parameters defined
            in 'defaultParams' will be used.

        Note:
        -----
        The forward Euler method can be replaced by other methods with better
        accuracy, such as Runge-Kutta.

        """
        params = params or cls.defaultParams
        gradStates = cls.ode(states, params)
        statesDict = states._asdict()
        for k,v in statesDict.items():
            statesDict[k] = v + dt*getattr(gradStates, k)
        return cls.States(**statesDict)

class PassiveModel(Neuron):
    Params = recordclass('params', ('g','E','C'))
    States = recordclass('states', ('V','I'))
    initStates = States(V=-50., I=0.0)
    defaultParams = Params(g=3., E=-44.387, C=0.001)

    @classmethod
    def ode(cls, states, params=None):
        params = params or cls.defaultParams
        gradStates = cls.States(*[0.]*len(cls.States._fields))

        gradStates.V = 1./params.C*(-params.g*(states.V-params.E)+states.I)
        return gradStates

class IAF(Neuron):
    """
    Integrate-and-Fire Neuron
    """
    Params = recordclass('params', ('bias','kappa','delta', 'reset'))
    States = recordclass('states', ('V','I'))
    initStates = States(V=-65., I=0.0)
    defaultParams = Params(bias=1.0, kappa=0.01, delta=-60., reset=-70.)

    @classmethod
    def ode(cls, states, params=None):
        params = params or cls.defaultParams
        gradStates = cls.States(*[0.]*len(cls.States._fields))

        gradStates.V = (params.bias+states.I) / params.kappa
        return gradStates

    @classmethod
    def update(cls, dt, states, params=None):
        params = params or cls.defaultParams
        newStates = super(IAF, cls).update(dt, states, params)
        if newStates.V > params.delta:
            newStates.V = params.reset
        return newStates

class HodgkinHuxley(Neuron):
    """
    Hodgkin Huxley Neuron Model
    """
    Params = recordclass('params', ('E_Na','E_K','E_l','g_Na','g_K','g_l'))
    States = recordclass('states', ('V','I','n','m','h'))
    initStates = States(V=-65., n=0.0, m=0.0, h=1.0, I=0.)
    defaultParams = Params(
        E_Na=50.,
        E_K=-77.,
        E_l=-54.387,
        g_Na=120.,
        g_K=36.,
        g_l=0.3)

    @classmethod
    def ode(cls, states, params=None):
        params = params or cls.defaultParams
        gradStates = cls.States(*[0.]*len(cls.States._fields))

        # compute deristates.Vatistates.Ve of N
        a = exp(-(states.V+55.)/10.)-1.
        if a == 0.:
            gradStates.n = (1-states.n)*0.1 - \
                 states.n*(0.125*exp(-(states.V+65.)/80.))
        else:
            gradStates.n = (1-states.n)*(-0.01*(states.V+55.)/a) - \
                 states.n*(0.125*exp(-(states.V+65.)/80.))

        # compute deristates.Vatistates.Ve of M
        a = exp(-(states.V+40.)/10.) - 1.
        if a == 0.:
            gradStates.m = (1-states.m) - \
                 states.m*(4.*exp(-(states.V+65.)/18.))
        else:
            gradStates.m = (1-states.m)*(-0.1*(states.V+40.)/a) - \
                 states.m*(4.*exp(-(states.V+65.)/18.))

        # compute deristates.Vatistates.Ve of H
        gradStates.h = (1.-states.h)*(0.07*exp(-(states.V+65.)/20.)) - \
             states.h/(1.+exp(-(states.V+35.)/10.))

        # compute deristates.Vatistates.Ve of states.V
        gradStates.V = states.I \
            - params.g_K*(states.n**4)*(states.V-params.E_K) \
            - params.g_Na*(states.m**3)*states.h*(states.V-params.E_Na) \
            - params.g_l*(states.V-params.E_l)

        return gradStates

    @classmethod
    def update(cls, dt, states, params=None):
        newStates = super(HodgkinHuxley, cls).update(1e3*dt, states, params)
        return newStates

class FitzHughNagumo(Neuron):
    """
    FitzHugh-Nagumo Neuron
    """
    Params = recordclass('params', ())
    States = recordclass('states', ('V','I','W'))
    initStates = States(V=-65., W=0., I=0.)
    defaultParams = Params()

    @classmethod
    def ode(cls, states, params=None):
        params = params or cls.defaultParams
        gradStates = cls.States(*[0.]*len(cls.States._fields))

        gradStates.V = states.V - states.V**3/3. - states.W + states.I
        gradStates.W = 0.08*(states.V+0.7-0.8*states.W)
        return gradStates.W

if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dt = 1e-5
    stimulusStart = 0.05
    dur = 0.2
    t = np.arange(0, dur, dt)
    stimulus = np.zeros_like(t)
    stimulus[t > stimulusStart] = 20

    models = (PassiveModel, HodgkinHuxley, IAF)
    states = {k:k.initStates._replace() for k in models}
    vTraces = {k:np.zeros_like(t) for k in models}

    for i, s in enumerate(stimulus):
        for m in models:
            states[m] = states[m]._replace(I=s)
            states[m] = m.update(dt, states[m])
            vTraces[m][i] = states[m].V

    fig, axes = plt.subplots(len(models), 1, figsize=(8,8))
    for ax, m in zip(axes, models):
        ax.plot(t,vTraces[m])
        ax.set_xlabel('Time, [s]')
        ax.set_ylabel('Voltage, [mV]')
        ax.set_title(m.__name__)

    plt.tight_layout()
    plt.savefig('hhn.png', dpi=300)
