# ENVQT
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Exploiting Noise in Variational Quantum Thermalization"* available on arXiv soon.

Exploiting Noise in Variational Quantum Thermalization (ENVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
Using Google Colab, the following two lines at the very top of the notebook should be sufficient for the algorithm to work:
```bash
!pip install -q tensorflow==2.3.1
!pip install -q tensorflow-quantum
```

## Usage 
For N=4 qubits, L=2 layers in the QAOA-flexible ansatz approximating the thermal state of an Hamiltonian with neighbor ZZ and XX interactions at temperature 1/beta:
```python
  # Circuit initialization
  qc            = QC(N=4,
                  L=2,
                  hamiltonian=["ZZ","XX"],
                  beta=1.0,
                  seed=0,
                  ansatz="qaoa-f")
  for e in tqdm(range(50),leave=False):
    qc.train(use_lambda_gradient = p.optim == "gradient")
    
  qc.plot_history()
```

## Example: N=9, L=5, beta = 1.01, hamiltonian=["ZZ","Z"]

![alt text](https://github.com/jfold/envqt/blob/main/training-history-example.png "Training history")
