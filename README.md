# scqpth_bench
This repository provides benchmark experiments for the SCQPTH differentiable QP solver. SCQPTH is a differentiable first-order splitting method for convex quadratic programs. The QP solver is implemented as a custom PyTorch module. The forward solver invokes a basic implementation of the ADMM algorithm. Backward differentiation is performed by implicit differentiation of a fixed-point mapping customized to the ADMM algorithm.

For more information please see our publication:

[arXiv (preprint)](https://github.com/ipo-lab/scqpth)


## Core Dependencies:
To use the ADMM solver you will need to install [numpy](https://numpy.org) and [Pytorch](https://pytorch.org). Please see requirements.txt for details and to run demos and replicate experiments.

## Runtime Experiments:
The [demo](demo) directory contains simple demos for forward solving and backward differentiating through the ADMM solver. The following runtime experiments are available in the [experiments](experiments) directory.

All experiments are conducted on an Apple Mac Pro computer (2.6 GHz 6-Core Intel Core i7,32 GB 2667 MHz DDR4) running macOS ‘Monterey’.

### Experiment 1:
Runtime: n = 10            |  Runtime: n = 50    
:-------------------------:|:-------------------------:
![runtime n 10](/images/experiment_2/png/n_x_10_m_10.png)  |  ![runtime n 50](images/experiment_2/png/n_x_50_m_50.png)

Runtime: n = 100            |  Runtime: n = 250    
:-------------------------:|:-------------------------:
![runtime n 100](images/experiment_2/png/n_x_100_m_100.png)  |  ![runtime n 250](images/experiment_2/png/n_x_250_m_250.png)

Runtime: n = 500            |  Runtime: n = 1000    
:-------------------------:|:-------------------------:
![runtime n 500](images/experiment_2/png/n_x_500_m_500.png)  |  ![runtime n 1000](images/experiment_2/png/n_x_1000_m_1000.png)

Computational performance of SCQPTH, QPTH and Cvxpylayers for random QPs of
various problem sizes, n, constraints m = n, and low stopping tolerance (1e−3).

### Experiment 2:

Runtime: n = 100            |  Convergence: n = 100    
:-------------------------:|:-------------------------:
![runtime dz 500](images/experiment_3/png/n_x_100_m_200.png)  |  ![convergence dz 500](images/experiment_3/png/n_x_100_m_200_loss.png)

Training loss and computational performance for learning p. Problem setup: n = 100, m = 200, batch size = 32, epochs = 100, and stopping tolerance = 1e-3.
