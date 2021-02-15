# NAVQT
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Noise-assisted Variational Quantum Thermalization"* available on arXiv soon.

Noise-assisted Variational Quantum Thermalization (NAVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian at a specified temperature. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
Using Google Colab, the following two lines at the very top of the notebook should be sufficient for the algorithm to work:
```bash
!pip -q install tensorflow==2.3.1 tensorflow_probability==0.11 tensorflow-quantum cirq 
```

## Usage 
For N=9 qubits, L=5 layers approximating the thermal state of an Ising Chain Hamiltonian temperature 1/beta:
```python
vqt = NAVQT(N=8,
            L=5,
            K=1000,
            epsilon=0.0014,
            gamma_lr=0.25,
            H_type="IC-u",
            p_error_lr=0.06,
            beta=0.001,
            p_error=0.001,
            seed=3,
            pth="drive/MyDrive/PhD/NAVQT/")
vqt.train(n_epochs=50)
vqt.plot_history()
```

## Example: N=9, L=5, beta = 1.01, hamiltonian=["ZZ","Z"]

![alt text](https://github.com/jfold/envqt/blob/main/training-history-example.png "Training history")
