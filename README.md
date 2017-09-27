# SWC Simulator

A Python simulator for neuron models specified in the
[SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
format.

## Installation

First clone the repository to a local directory, and navigate to the directory. Install the package with the following command,

```
python setup.py install
```

## Usage

#### Load SWC
The first step is to load a SWC file,

```python
from swcsim import SWCNeuron
neuron = SWCNeuron('neuron.swc')
```
The SWC file is converted to a `Networkx` graph, in which each node represents a segment in the original SWC file, and each edge represents the link between two segments. A node has three required attributes:

* `model`: a python class implementing a neural computational model, ex. Hodgkin-Huxley model.
* `params`: parameters of the computational model. If not specified, the default set of parameter will be used.
* `states`: states variables of the computational model. `states` must have at least two fields, `I` and `V`.

#### Setup Simulator
The second step is to create a simulator by calling `create_simulator`,

```python
simulator = neuron.create_simulator(
    dt,
    group_1, stimulus_1,
    group_2, stimulus_2,
    ...,
    steps = 1000)
```

The function `create_simulator` takes variable pairs of `group_i` and `stimulus_i` as positional arguments. `group_i` is a set or an iterator of identifiers of nodes that receive input from `stimulus_i`. `stimulus_i` is a python iterator (ex. a `numpy.ndarray` or a generator) that will be directly injected into nodes as external current. `create_simulator` also takes keyword arguments `steps` or `durations` that corresponds to number of steps to run.

The function `create_simulator` creates a generator that updates all nodes and yields its caller `SWCNeuron` instance at each iteration.

#### Run Simulator
Once the simulator is created by calling `create_simulator`, running simulation is as simple as looping through the resultant generator,

```python
simulator = neuron.create_simulator(...)
for i,_ in enumerate(simulator):
    # record voltage value for a particular node[x]
    V[i] = neuron.graph.node[x]['states'].V
```

## Contributors
This repository is maintained by following people:

* [Chung-Heng Yeh](http://www.bionet.ee.columbia.edu/people)
* [Mehmet Kerem Turkcan](http://www.bionet.ee.columbia.edu/people)
* [Tingkai (Thomas) Liu](http://www.bionet.ee.columbia.edu/people)
