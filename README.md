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

The `dataset.py` file automatically downloads data samples from the EMNIST database. The downloaded data is in a format that can be used to generate meta-training tasks. Running the code for the first time may take slightly longer as the data needs to be prepared.

## Run

The model is trained using `main.py`. This code accepts the following arguments:

```bash
optional arguments:
  --gpu_mode        Accelerate the script using GPU (default: 1)
  --seed            Random seed (default: 1)
  --database        Meta-training database (default: emnist)
  --dim             Dimension of the training data (default: 28)
  --test_name       Name of the folder at the secondary level in the hierarchy of the results directory tree (default: '')
  --episodes        Number of meta-training episodes (default: 600)
  --K               Number of training data points per class (default: 50)
  --Q               Number of query data points per class (default: 10)
  --M               Number of classes per task (default: 5)
  --lamb            Meta-loss regularization parameter (default: 0.)
  --lr_meta         Meta-optimization learning rate (default: 1e-3)
  --a               Initial learning rate for the pseudo-gradient term at episode 0 (default: 1e-3)
  --res             Result directory (default: 'results')
  --avg_window      The size of moving average window used in the output figures (default: )
  --vec             Index vector specifying the plasticity terms to be used for model training in adaptation (default: '')
  --fbk             Feedback connection type: 1) sym = Symmetric feedback; 2) fix = Fixed random feedback (default: 'fix')
```

You can run the code using the command

```bash
python3 main.py
```

## Citation

You can use this code, as whole or in part, by citing:
```latex
@article{shervani2022meta,
  title={Meta-Learning Biologically Plausible Plasticity Rules with Random Feedback Pathways},
  author={Shervani-Tabar, Navid and Rosenbaum, Robert},
  journal={arXiv preprint arXiv:2210.16414},
  year={2022}
}
```

## Questions

For any questions or comments regarding this work, submit an issue [here](https://github.com/NeuralDynamicsAndComputing/MetaLearning-Plasticity/issues) or contact Navid Shervani-Tabar (nshervan@nd.edu). In the email title, please use "Regarding meta-learning plasticity paper".
