# NAVQT
This repository contains the code and examples for the corresponding paper entitled *Foldager, Jonathan, et al. "Noise-assisted Variational Quantum Thermalization"*.

Noise-assisted Variational Quantum Thermalization (NAVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian at a specified temperature. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy. 



## Install packages
The requirements.txt file lists all Python libraries we depend on, and they can be installed using:
```bash
pip install -r requirements.txt
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

