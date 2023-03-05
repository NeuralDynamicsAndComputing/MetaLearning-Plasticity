Meta-Learning Biologically Plausible Plasticity Rules with Random Feedback Pathways
======================================

This repository contains a PyTorch implementation of the meta-learning approach for developing interpretable, biologically plausible plasticity rules that improve online learning performance with fixed random feedback connections. 

The motivation behind this approach stems from the issue of understanding the relationship between backpropagation and synaptic plasticity in the brain, given the lack of empirical evidence supporting the existence of symmetric backward connectivity. A well-known alternative, Random Feedback Alignment, relies on fixed, random backward connections to propagate errors backward. However, this approach learns slowly and performs poorly with deeper models or online learning. 

Our solution to this problem is developing a meta-learning approach that discovers plasticity rules that improve the online training of deep models in the low data regime. Our approach shows promising results and highlights the potential of meta-learning to discover effective, interpretable learning rules that satisfy biological constraints.

## Getting Started

### Dependencies

The following packages and their respective versions were used during the development of this implementation:

* Python (3.7.7)
* PyTorch (1.11.0.dev20211018)
* git (3.1.24)
* NumPy (1.19.2)
* gzip
* PIL (8.3.1)
* torchvision (0.12.0.dev20211018)

### Data

The `dataset.py` file automatically downloads data samples from the EMNIST database.



