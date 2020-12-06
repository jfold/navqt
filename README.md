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

## Example: N=9, L=5, beta = 1.01
Notation:
    
![formula](https://render.githubusercontent.com/render/math?math=\langle%20F%20\rangle=\langle%20H%20\rangle%20&#43%20\frac{1}{\beta}S)

![alt text](https://github.com/jfold/envqt/blob/main/training-history-example.png "Training history")
