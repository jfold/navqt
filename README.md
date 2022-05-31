# NAVQT
This repository contains the code and examples for the corresponding paper [*Foldager, Jonathan, et al. "Noise-assisted Variational Quantum Thermalization"*](https://www.nature.com/articles/s41598-022-07296-z).

Noise-assisted Variational Quantum Thermalization (NAVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian at a specified temperature. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
```bash
pip install -q tensorflow==2.3.1 tensorflow_probability==0.11.0 tensorflow-quantum==0.4.0 cirq==0.9.1
```
## Usage 
Shell:
```bash
python main.py "N=4|model=IC-u|ansatz=qaoa-r|beta=10.0"
```

Notebook:
```python
from navqt import NAVQT
navqt = NAVQT(N=4,model="IC-u",ansatz="qaoa-r",beta=10.0)
navqt.train()
navqt.plot_history()
```

