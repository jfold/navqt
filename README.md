# NAVQT
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Noise-assisted Variational Quantum Thermalization"* available on arXiv soon.

Noise-assisted Variational Quantum Thermalization (NAVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian at a specified temperature. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
The requirements.txt file lists all Python libraries we depend on, and they can be installed using:
```bash
pip install -r requirements.txt
```

## Usage 
For N=8 qubits, L=5 layers approximating the thermal state of an Ising Chain Hamiltonian with uniformly distributed coefficients at temperature 1/beta:
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
            seed=0)
vqt.train(n_epochs=50)
vqt.plot_history()
```

