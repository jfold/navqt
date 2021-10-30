# NAVQT
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Noise-assisted Variational Quantum Thermalization"* available on arXiv soon.

Noise-assisted Variational Quantum Thermalization (NAVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian at a specified temperature. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
The requirements.txt file lists all Python libraries we depend on, and they can be installed using:
```bash
pip install -r requirements.txt
```

## Usage 
Example:
```python
from navqt import NAVQT
navqt = NAVQT(N=4,model="IC-u",ansatz="qaoa-f",K=100,beta=10.0,p_err=1e-4,epsilon=1e-5,multilambda=False)
navqt.train()
navqt.plot_history()
```

