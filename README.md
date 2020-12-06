# ENVQT
Exploiting Noise in Variational Quantum Thermalization (ENVQT) is an algorithm used to learn the parameters in a variational quantum circuit which prepares a thermal state of a Hamiltonian. Different from other approaches it considers the noise itself as a variational parameter which can be learned using approximations on the entropy.

## Install packages
```bash
pip install -q tensorflow==2.3.1
pip install -q tensorflow-quantum
```

## Usage 
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

## Example: N=9, L=4, beta = 1.01
![alt text](https://github.com/jfold/envqt/edit/main/icon48.pdf "Training history")