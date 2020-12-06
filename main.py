import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import os
import sympy
import pickle
from scipy.linalg import expm,logm
from scipy.optimize import minimize
import numpy as np
import itertools
from cirq import Simulator, DensityMatrixSimulator
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['font.size'] = 26
matplotlib.rcParams['figure.figsize'] = (15, 6)

# Parameter class
class p:
  # =============================================================================
  # Default setting
  # =============================================================================
  # INTEGERS
  N                       = 3
  seed                    = 0
  save_iter 							= 5
  ints                    = ["N","seed","save_iter"]
  # FLOATS
  beta                    = 0.1
  lambda0                 = 1e-3
  theta_lr                = 0.4
  lambda_lr               = 1e-3
  floats                  = ["beta","lambda0","theta_lr","lamda_lr"]
  # BOOLEANS
  randH                   = False
  check_exist             = True
  lean_save 							= True
  bools                   = ["randH","check_exist","lean_save"]
  # STRINGS
  model                   = 'ZZ+Z+X'
  optim                   = 'gradient'
  strings                 = ["model","optim"]
  
class QC(object):
  def __init__(self, 
               N = 1, 
               N_ancilla = 0,
               L = 1, 
               expectation_method = "analytical", 
               hamiltonian=["ZZ","Z"],
               coefficients = {},
               theta_lr = 0.01,
               lambda_lr = 0.01,
               n_expectation_samples = 50,
               seed = 0,
               hard_Hc = False,
               lambda0 = 1e-3,
               sim_mixed = False,
               ansatz="qaoa-r",
               beta=1.0, # Boltzmann defualt
               theta_std = 1e-1,
               epsilon=1e-4,
               learning_rate_decay = 0.05,
               savepth="results/",
               hadamard_at_beginning = True,
               verbøgse=False):
    self.seed                   = seed
    self.verbøgse               = verbøgse
    self.N                      = N
    self.N_ancilla              = N_ancilla
    self.d                      = 2**N
    self.L                      = L
    self.epsilon                = epsilon
    self.learning_rate_decay    = 1-learning_rate_decay
    self.ansatz                 = ansatz
    self.expectation_method     = expectation_method
    self.n_expectation_samples  = n_expectation_samples
    self.qubits                 = cirq.GridQubit.rect(self.N, 1)
    self.theta_lr               = theta_lr
    self.lambda_lr              = lambda_lr
    self.lambda_                = lambda0
    self.savepth                = savepth
    self.beta                   = beta
    self.hamiltonian            = hamiltonian
    self.coefficients           = coefficients
    self.hadamard_at_beginning  = hadamard_at_beginning
    np.random.seed(self.seed)

    ## HAMILTONIAN
    if self.hamiltonian == "boltzmann":
      self.W              = coefficients["W"]
      self.b              = coefficients["b"]
      self.n_v,self.n_h   = np.shape(self.W)
    else:
      # Z coefficients
      self.Z_J            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "ZZ" in hamiltonian else np.zeros((self.N,1))
      self.Z_b            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "Z" in hamiltonian else np.zeros((self.N,1))

      # X coefficients
      self.X_J            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "XX" in hamiltonian else np.zeros((self.N,1))
      self.X_b            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "X" in hamiltonian else np.zeros((self.N,1))

      # Y coefficients
      self.Y_J            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "YY" in hamiltonian else np.zeros((self.N,1))
      self.Y_b            = (np.random.randn(self.N,1) if hard_Hc else np.ones((self.N,1))) if "Y" in hamiltonian else np.zeros((self.N,1))

    # Make backends
    self.simulator      = DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_),ignore_measurement_results=True)
    self.mix_simulator  = DensityMatrixSimulator(noise=cirq.depolarize(1.0))

    # Make circuit, hamiltonian and expectations
    self.update_state()

    # Calculate number of parameters and gates
    self.n_params                   = len(self.symbols)
    if "entangl" in ansatz:
      self.n_gates_pr_qubit         = 1 + (3+2)*self.L  # 1 hadamard + 3 rotations + 2 entangling gates
      self.n_gates                  = self.n_gates_pr_qubit*self.N
    if "qaoa" in ansatz:
      self.n_gates_pr_qubit         = 1 + (2+2)*self.L  # 1 hadamard + 2 rotations + 2 entangling gates
      self.n_gates                  = self.n_gates_pr_qubit*self.N
    if "rbm" in ansatz:
      self.n_gates                  = (self.W.size + 1 + 1)*self.L
    if "fire" in ansatz:
      self.n_gates                  = (2*self.W.size + 1 + 1)*self.L

    self.thetas                     = np.random.randn((self.n_params))*theta_std
    self.thetas_tf                  = tf.Variable(tf.convert_to_tensor([self.thetas],dtype=tf.float32))
    self.lambda_tf                  = tf.Variable(tf.convert_to_tensor([self.lambda_],dtype=tf.float32))
    self.current_rho                = self.forward()

    # Metrics
    self.loss                       = [self.get_loss()]
    self.free_energy                = [self.get_free_energy(run_sim=False)]
    self.entropy                    = [self.get_entropy()]
    self.entropy_true               = [self.get_entropy(use_true=True,run_sim=False)]
    self.energy                     = [self.get_H_expectation_by_matrix()]
    self.fidelity                   = [self.thermal_state_fidelity(run_sim=False)]
    self.tr_dist                    = [self.thermal_state_trace_distance(run_sim=False)]

    self.lambdas                    = [self.lambda_]
    self.best_loss                  = self.loss[-1]
    self.best_thetas                = self.thetas.copy()
    self.best_lambda                = self.lambda_
    self.rho_circuit_eigenvalues    = np.real(np.linalg.eigvals(self.current_rho))
    self.rho_thermal_eigenvalues    = np.real(np.linalg.eigvals(self.thermal_matrix))
    self.theta_optimizer            = tf.optimizers.Adam(self.theta_lr)
    self.lambda_optimizer           = tf.optimizers.Adam(self.lambda_lr)
    self.circuit_init_eigenvalues   = np.real(np.linalg.eigvals(self.current_rho))

    beta_str                        = "--beta-"+ ("%.2E" % +self.beta).replace("+","p").replace("-","m").replace(".","-")
    lambda0_str                     = "--lamda0-"+ ("%.2E" % +self.lambda_).replace("+","p").replace("-","m").replace(".","-")
    hamilton_hard_str               = "h-" if hard_Hc else "s-"
    hamilton_str                    = "--H"+hamilton_hard_str+ '+'.join(sorted(set(self.hamiltonian)))
    self.settings                   = "optim-"+p.optim+"--N-"+str(self.N)+"--L-"+str(self.L)+hamilton_str+beta_str+lambda0_str+"--seed-"+str(self.seed)

  def calc_targets(self):
    self.H_eigs     = np.sort(np.real(np.linalg.eigvalsh(self.Hc_matrix)))
    self.H_gap      = np.abs(self.H_eigs[0] - self.H_eigs[1])

    self.rho_eigs   = np.sort(np.real(np.linalg.eigvalsh(self.thermal_matrix)))
    self.rho_gap    = np.abs(self.rho_eigs[0] - self.rho_eigs[1])

    self.H_target = np.real(np.trace(np.dot(self.Hc_matrix,self.thermal_matrix)))
    eig           = np.real(np.linalg.eigvalsh(self.thermal_matrix)) 
    eig           = np.array([np.maximum(self.epsilon,x) for x in eig])
    self.S_target = -eig.dot(np.log(eig))
    self.F_target = self.H_target - self.S_target*(1/self.beta)

  def plot_density_matrices(self,parameters=[],save=False):
    f       = plt.figure(figsize=(20,10))
    ax1     = plt.subplot(121)
    
    if parameters == []:
      rho     = np.absolute(self.forward())
    else:
      rho     = np.absolute(self.forward(parameters))

    # CIRCUIT STATE
    if self.N < 5:
      pos1    = ax1.imshow(rho)
      for y_index, y in enumerate(range(self.d)):
        for x_index, x in enumerate(range(self.d)):
          label   = np.round(rho[y_index, x_index],2)
          text_x  = x
          text_y  = y
          ax1.text(text_x, text_y, label, color='black', ha='center', va='center')
    else:
      pos1    = ax1.imshow(rho)

    eigvals = np.real(np.linalg.eigvals(rho))
    plt.title("Output")
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    f.colorbar(pos1,label=r"$|\rho_{ij}|$");
    ax1.grid(False)

    # THERMAL STATE
    ax2     = plt.subplot(122)
    plt.title("Thermal state")
    rho     = np.absolute(self.thermal_matrix)
    if self.N < 5:
      pos2    = plt.imshow(rho)
      # Text in pixels
      for y_index, y in enumerate(range(self.d)):
        for x_index, x in enumerate(range(self.d)):
          label   = np.round(rho[y_index, x_index],2)
          text_x  = x
          text_y  = y
          ax2.text(text_x, text_y, label, color='black', ha='center', va='center')
    else:
      pos2    = plt.imshow(rho)


    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    f.colorbar(pos2,label=r"$|\rho_{ij}|$");
    ax2.grid(False)

    if save:
      f.savefig(self.savepth+"density-matrices--"+self.settings+".pdf")

    plt.show()

    ax3 = plt.subplot(111)
    w   = 0.25
    x   = np.arange(self.d)
    ax3.bar(x-w, np.sort(eigvals), width=w, color='b',label="Spectrum after training")
    ax3.bar(x, np.sort(self.rho_thermal_eigenvalues), width=w, color='g',label="Thermal spectrum")
    frame2 = plt.gca()
    frame2.axes.xaxis.set_ticklabels([])
    plt.legend()
    plt.ylabel("Eigenvalue")

    if save:
      f.savefig(self.savepth+"spectra--"+self.settings+".pdf")

    plt.show()

    plt.subplot(111)
    plt.hist(eigvals, 20,alpha=0.5, label='Output state')
    plt.hist(self.rho_thermal_eigenvalues,20, alpha=0.5, label='Thermal state')
    plt.legend(loc='upper right')
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")

    if save:
      f.savefig(self.savepth+"eigenvalues--"+self.settings+".pdf")

    plt.show()

  def plot_history(self,save=False,figsize=(20,10)):
    f = plt.figure(figsize=figsize)
    plt.subplot(2,3,1)
    plt.plot(self.loss,'--.',label="Approx")
    plt.plot(self.free_energy,'--s',label="True")
    plt.plot([self.F_target]*len(self.loss),'-',label="Target")
    plt.xlabel("Epochs");
    plt.legend()
    plt.ylabel(r"$\langle F \rangle$");

    plt.subplot(2,3,2)
    plt.plot(self.entropy,'--.',label="Approx.")
    plt.plot(self.entropy_true,'--s',label="True")
    plt.plot([self.S_target]*len(self.entropy),'-',label="Target")
    plt.xlabel("Epochs");
    plt.legend()
    plt.ylabel(r"$S$");

    plt.subplot(2,3,3)
    plt.plot(self.energy,'--.',label="System")
    plt.plot([self.H_target]*len(self.energy),'-',label="Target")
    plt.xlabel("Epochs");
    plt.legend()
    plt.ylabel(r"$\langle H \rangle$");

    plt.subplot(2,3,4)
    plt.plot(self.lambdas,'--.')
    plt.xlabel("Epochs");
    plt.ylabel(r"$\lambda$");

    plt.subplot(2,3,5)
    plt.plot(self.fidelity,'--.',label="Fidelity ("+str(np.round(np.max(self.fidelity),2))+")")
    plt.plot(self.tr_dist,'--.',label="Trace distance("+str(np.round(np.min(self.tr_dist),2))+")")
    plt.ylim([0.0,1.0])
    plt.legend()
    plt.xlabel("Epochs");
    # plt.ylabel("Fidelity");

    ax  = plt.subplot(2,3,6)
    w   = 0.15
    x   = np.arange(self.d)
    ax.bar(x-w, np.sort(np.real(np.linalg.eigvals(self.forward()))), width=w, color='b',label="Spectrum after training")
    ax.bar(x, np.sort(self.rho_thermal_eigenvalues), width=w, color='g',label="Thermal spectrum")
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    plt.legend()
    plt.ylabel("Eigenvalue")

    if save:
      f.savefig(self.savepth+"history--"+self.settings+".pdf")

    plt.show()

  def update_state(self):
    self.lambda_            = np.minimum(np.maximum(self.epsilon,self.lambda_),1-self.epsilon)
    self.qubits             = cirq.GridQubit.rect(self.N, 1)
    self.construct_hamiltonian()
    self.construct_circuit()
    self.simulator          = DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_))
    self.build_simulator()
    self.calc_targets()
    if hasattr(self, 'thetas'):
      self.thetas_tf          = tf.Variable(tf.convert_to_tensor([self.thetas],dtype=tf.float32))
    if hasattr(self, 'lambda_'):
      self.lambda_tf          = tf.Variable(tf.convert_to_tensor([self.lambda_],dtype=tf.float32))

  def build_simulator(self):
    if self.expectation_method == "analytical":
      self.calc_pure_expectation = tfq.layers.Expectation(differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01),backend=self.simulator) 
      self.calc_mixed_expectation = tfq.layers.Expectation(differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01),backend=self.mix_simulator) 
    if self.expectation_method == "sampled":
      self.calc_pure_expectation = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01),backend=self.simulator)
      self.calc_mixed_expectation = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01),backend=self.mix_simulator)
    if self.expectation_method == "safe_sampled":
      self.calc_pure_expectation = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ParameterShift(),backend=self.simulator)
      self.calc_mixed_expectation = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ParameterShift(),backend=self.mix_simulator)

  def construct_hamiltonian(self):
    self.Hc_coefficients  = []
    self.Hc_symbols       = []
    self.Hc_operators     = []
    self.Hc_matrices      = []
    self.Hc               = cirq.PauliSum()
    self.Hc_matrix        = np.zeros((2**self.N,2**self.N)) + 0j

    if self.hamiltonian == "boltzmann":
      H   = cirq.PauliSum()
      # INTERACTION
      for i_v in range(self.n_v):
        for i_h in range(self.n_h):
          if i_v >= i_h:
            self.Hc -= float(self.W[i_v,i_h])*cirq.Z(self.qubits[i_v])*cirq.Z(self.qubits[self.n_v+i_h]) 
            self.Hc_coefficients.append(self.W[i_v,i_h])
            self.Hc_symbols.append("ZZ_"+str(i_v)+str(self.n_v+i_h))
            self.Hc_operators.append(float(self.W[i_v,i_h])*cirq.Z(self.qubits[i_v])*cirq.Z(self.qubits[self.n_v+i_h]) )

            dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
            dummy_circuit += cirq.Circuit(cirq.ZZ(self.qubits[i_v],self.qubits[self.n_v+i_h]))
            self.Hc_matrices.append(cirq.unitary(dummy_circuit))
            self.Hc_matrix -= self.W[i_v,i_h]*cirq.unitary(dummy_circuit)
      # BIAS
      for i in range(self.N):
        self.Hc -= float(self.b[i])*cirq.Z(self.qubits[i]) 
        self.Hc_coefficients.append(self.b[i])
        self.Hc_symbols.append("Z_"+str(i))
        self.Hc_operators.append(float(self.b[i])*cirq.Z(self.qubits[i]))

        dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
        dummy_circuit += cirq.Circuit(cirq.Z(self.qubits[i]))
        self.Hc_matrices.append(cirq.unitary(dummy_circuit))
        self.Hc_matrix -= self.b[i]*cirq.unitary(dummy_circuit)
    else:
      for i,qubit in enumerate(self.qubits):
        ########### INTERACTION
        if i < self.N-1:
        # ZZ
          if self.Z_J[i] != 0:
            self.Hc -= float(self.Z_J[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[i+1])
            self.Hc_coefficients.append(self.Z_J[i])
            self.Hc_symbols.append("ZZ_"+str(i)+str(i+1))
            self.Hc_operators.append(-float(self.Z_J[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[i+1]))

            dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
            dummy_circuit += cirq.Circuit(cirq.ZZ(self.qubits[i],self.qubits[i+1]))
            self.Hc_matrices.append(cirq.unitary(dummy_circuit))
            self.Hc_matrix -= self.Z_J[i]*cirq.unitary(dummy_circuit)
        # XX
          if self.X_J[i] != 0:
            self.Hc -= float(self.X_J[i])*cirq.X(self.qubits[i])*cirq.X(self.qubits[i+1])
            self.Hc_coefficients.append(self.X_J[i])
            self.Hc_symbols.append("XX_"+str(i)+str(i+1))
            self.Hc_operators.append(-float(self.X_J[i])*cirq.X(self.qubits[i])*cirq.X(self.qubits[i+1]))

            dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
            dummy_circuit += cirq.Circuit(cirq.XX(self.qubits[i],self.qubits[i+1]))
            self.Hc_matrices.append(cirq.unitary(dummy_circuit))
            self.Hc_matrix -= self.X_J[i]*cirq.unitary(dummy_circuit)
        # YY
          if self.Y_J[i] != 0:
            self.Hc -= float(self.Y_J[i])*cirq.Y(self.qubits[i])*cirq.Y(self.qubits[i+1])
            self.Hc_coefficients.append(self.Y_J[i])
            self.Hc_symbols.append("YY_"+str(i)+str(i+1))
            self.Hc_operators.append(-float(self.Y_J[i])*cirq.Y(self.qubits[i])*cirq.Y(self.qubits[i+1]))

            dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
            dummy_circuit += cirq.Circuit(cirq.YY(self.qubits[i],self.qubits[i+1]))
            self.Hc_matrices.append(cirq.unitary(dummy_circuit))
            self.Hc_matrix -= self.Y_J[i]*cirq.unitary(dummy_circuit)
          
        ########### BIAS
        # Z
        if self.Z_b[i] != 0:
          self.Hc -= float(self.Z_b[i])*cirq.Z(self.qubits[i])
          self.Hc_coefficients.append(self.Z_b[i])
          self.Hc_symbols.append("Z_"+str(i))
          self.Hc_operators.append(-float(self.Z_b[i])*cirq.Z(self.qubits[i]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.Z(self.qubits[i]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.Z_b[i]*cirq.unitary(dummy_circuit)
        # X
        if self.X_b[i] != 0:
          self.Hc -= float(self.X_b[i])*cirq.X(self.qubits[i])
          self.Hc_coefficients.append(self.X_b[i])
          self.Hc_symbols.append("X_"+str(i))
          self.Hc_operators.append(-float(self.X_b[i])*cirq.X(self.qubits[i]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.X(self.qubits[i]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.X_b[i]*cirq.unitary(dummy_circuit)
        # Y
        if self.Y_b[i] != 0:
          self.Hc -= float(self.Y_b[i])*cirq.Y(self.qubits[i])
          self.Hc_coefficients.append(self.Y_b[i])
          self.Hc_symbols.append("Y_"+str(i))
          self.Hc_operators.append(-float(self.Y_b[i])*cirq.Y(self.qubits[i]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.Y(self.qubits[i]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.Y_b[i]*cirq.unitary(dummy_circuit)
      
      ############ Last interaction
      if self.N > 2:
        # ZZ
        if self.Z_J[i] != 0:
          self.Hc   -= float(self.Z_J[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[0]) 
          self.Hc_coefficients.append(self.Z_J[i])
          self.Hc_symbols.append("ZZ_"+str(i)+str(0))
          self.Hc_operators.append(-float(self.Z_J[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[0]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.ZZ(self.qubits[i],self.qubits[0]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.Z_J[i]*cirq.unitary(dummy_circuit)

        # XX
        if self.X_J[i] != 0:
          self.Hc   -= float(self.X_J[i])*cirq.X(self.qubits[i])*cirq.X(self.qubits[0]) 
          self.Hc_coefficients.append(self.X_J[i])
          self.Hc_symbols.append("XX_"+str(i)+str(0))
          self.Hc_operators.append(-float(self.X_J[i])*cirq.X(self.qubits[i])*cirq.X(self.qubits[0]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.XX(self.qubits[i],self.qubits[0]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.X_J[i]*cirq.unitary(dummy_circuit)

        # YY
        if self.Y_J[i] != 0:
          self.Hc   -= float(self.Y_J[i])*cirq.Y(self.qubits[i])*cirq.Y(self.qubits[0]) 
          self.Hc_coefficients.append(self.Y_J[i])
          self.Hc_symbols.append("YY_"+str(i)+str(0))
          self.Hc_operators.append(-float(self.Y_J[i])*cirq.Y(self.qubits[i])*cirq.Y(self.qubits[0]))

          dummy_circuit = cirq.Circuit(cirq.I(self.qubits[n]) for n in range(self.N))
          dummy_circuit += cirq.Circuit(cirq.YY(self.qubits[i],self.qubits[0]))
          self.Hc_matrices.append(cirq.unitary(dummy_circuit))
          self.Hc_matrix -= self.Y_J[i]*cirq.unitary(dummy_circuit)

    exp_H = expm(-self.beta*self.Hc_matrix)
    self.thermal_matrix = (1/np.trace(exp_H))*exp_H
    # self.thermal_matrix[np.isnan(self.thermal_matrix)] = 1.0  # might happen for large beta
    # self.thermal_matrix /= np.trace(self.thermal_matrix)      # trace one

  def construct_circuit(self):
    self.symbols            = []
    self.circuit   = cirq.Circuit()
    if self.ansatz == "entangl":
      for l in range(self.L):
        block = []
        for n in range(self.N):
          theta_        = 'theta_' + str(n) + "_0_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rz(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        block = []
        for n in range(self.N):
          theta_        = 'theta_' + str(n) + "_1_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rx(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        block = []
        for n in range(self.N):
          theta_        = 'theta_' + str(n) + "_2_"+ str(l)
          theta         = sympy.Symbol(theta_)
          block.append([cirq.rz(theta)(self.qubits[n])])
          self.symbols.append(theta_)
        self.circuit.append(cirq.Moment(block)) 

        self.circuit  += cirq.Circuit(cirq.CNOT(self.qubits[n],self.qubits[n+1]) for n in range(self.N-1))
        self.circuit  += cirq.Circuit(cirq.CNOT(self.qubits[-1],self.qubits[0]))

    elif self.ansatz == "qaoa-r":

      if self.hadamard_at_beginning:
        self.circuit  += cirq.H.on_each(self.qubits)

      for l in range(self.L):
        gamma         = sympy.Symbol('gamma' + str(l+1))
        beta          = sympy.Symbol('beta' + str(l+1))

        self.circuit  += cirq.Circuit((cirq.ZZPowGate(exponent=gamma).on(self.qubits[i], self.qubits[i+1]) for i in range(self.N - 1)))
        if self.N > 2:
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[-1], self.qubits[0])) 

        self.circuit  += cirq.Circuit((cirq.rz(gamma)(self.qubits[i]) for i in range(self.N)))
        self.circuit  += cirq.Circuit((cirq.rx(beta)(self.qubits[i]) for i in range(self.N)))

        self.symbols.append('gamma' + str(l+1))
        self.symbols.append('beta' + str(l+1))
        
    elif self.ansatz == "qaoa-f": 
      if self.hadamard_at_beginning:
        self.circuit  += cirq.H.on_each(self.qubits)

      for l in range(self.L):
        for n in range(self.N-1):
          sym           = 'gamma' + str(l+1) + '_' + str(n+1)
          gamma         = sympy.Symbol(sym)
          self.symbols.append(sym)
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[n],self.qubits[n+1]))

        if self.N > 2:
          sym           = 'gamma' + str(l+1) + '_' + str(self.N)
          gamma         = sympy.Symbol(sym)
          self.symbols.append(sym)
          self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[-1], self.qubits[0])) 

        gates = []
        for n in range(self.N):
          sym           = 'eta' + str(l+1) + '_' + str(n+1)
          eta           = sympy.Symbol(sym)
          gates.append(cirq.Circuit(cirq.rz(eta)(self.qubits[n])))
          self.symbols.append(sym)
        self.circuit.append(cirq.Moment(gates))

        gates = []
        for n in range(self.N):
          sym           = 'beta' + str(l+1) + '_' + str(n+1)
          beta          = sympy.Symbol(sym)
          gates.append(cirq.Circuit(cirq.rx(beta)(self.qubits[n])))
          self.symbols.append(sym)
        self.circuit.append(cirq.Moment(gates))

    elif self.ansatz == "rbm-r":

      if self.hadamard_at_beginning:
        self.circuit  += cirq.H.on_each(self.qubits)
      for l in range(self.L):
        gamma         = sympy.Symbol('gamma' + str(l+1))
        beta          = sympy.Symbol('beta' + str(l+1))

          # INTERACTION
        for i_v in range(self.n_v):
          for i_h in range(self.n_h):
            self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[i_v], self.qubits[self.n_v+i_h]))

        self.circuit  += cirq.Circuit((cirq.rz(gamma)(self.qubits[i]) for i in range(self.N)))
        self.circuit  += cirq.Circuit((cirq.rx(beta)(self.qubits[i]) for i in range(self.N)))

        self.symbols.append('gamma' + str(l+1))
        self.symbols.append('beta' + str(l+1))

    elif self.ansatz == "rbm-f":

      if self.hadamard_at_beginning:
        self.circuit  += cirq.H.on_each(self.qubits)
      for l in range(self.L):
        # INTERACTION
        for i_v in range(self.n_v):
          for i_h in range(self.n_h):
            gamma         = sympy.Symbol('gamma' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))
            self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[i_v], self.qubits[self.n_v+i_h]))
            self.symbols.append('gamma' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))

        gates = []
        for i in range(self.N):
          gamma         = sympy.Symbol('gamma' + str(l+1)+"_"+str(i+1))
          gates.append(cirq.Circuit(cirq.rz(gamma)(self.qubits[i])))
          self.symbols.append('gamma' + str(l+1)+"_"+str(i+1))
        self.circuit.append(cirq.Moment(gates))
        
        gates = []
        for i in range(self.N):
          beta         = sympy.Symbol('beta' + str(l+1)+"_"+str(i+1))
          gates.append(cirq.Circuit(cirq.rx(beta)(self.qubits[i])))
          self.symbols.append('beta' + str(l+1)+"_"+str(i+1))
        self.circuit.append(cirq.Moment(gates))  

    elif self.ansatz == "fire":

      if self.hadamard_at_beginning:
        self.circuit  += cirq.H.on_each(self.qubits)
      for l in range(self.L):
        # INTERACTION
        for i_v in range(self.n_v):
          for i_h in range(self.n_h):
            gamma         = sympy.Symbol('gamma' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))
            self.circuit  += cirq.Circuit(cirq.ZZPowGate(exponent=gamma).on(self.qubits[i_v], self.qubits[self.n_v+i_h]))
            self.symbols.append('gamma' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))

        for i_v in range(self.n_v):
          for i_h in range(self.n_h):
            beta         = sympy.Symbol('beta' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))
            self.circuit  += cirq.Circuit(cirq.XXPowGate(exponent=beta).on(self.qubits[i_v], self.qubits[self.n_v+i_h]))
            self.symbols.append('beta' + str(l+1)+"_"+str(i_v+1)+"_"+str(i_h+1))

        gates = []
        for i in range(self.N):
          gamma         = sympy.Symbol('gamma' + str(l+1)+"_"+str(i+1))
          gates.append(cirq.Circuit(cirq.rz(gamma)(self.qubits[i])))
          self.symbols.append('gamma' + str(l+1)+"_"+str(i+1))
        self.circuit.append(cirq.Moment(gates))
        
        gates = []
        for i in range(self.N):
          beta         = sympy.Symbol('beta' + str(l+1)+"_"+str(i+1))
          gates.append(cirq.Circuit(cirq.rx(beta)(self.qubits[i])))
          self.symbols.append('beta' + str(l+1)+"_"+str(i+1))
        self.circuit.append(cirq.Moment(gates))  



    self.symbols        = list(self.symbols)

  def forward(self, parameters = np.array([]),choose_best=False):
    if parameters.size != 0:
      l         = np.minimum(np.maximum(3*self.epsilon,parameters[-1]),1-3*self.epsilon)
      simulator = DensityMatrixSimulator(noise=cirq.depolarize(l)) 
    else:
      if choose_best:
        parameters  = self.best_thetas.copy()
        simulator   = DensityMatrixSimulator(noise=cirq.depolarize(self.best_lambda),ignore_measurement_results=True)
      else:
        parameters  = self.thetas.copy()
        simulator   = self.simulator

    return simulator.simulate(self.circuit,cirq.ParamResolver({symbol: parameters[i] for i,symbol in enumerate(self.symbols)})).final_density_matrix

  def get_loss(self):
    H     = self.get_H_expectation_by_matrix()
    S     = self.get_entropy()
    loss  = H - (1/self.beta)*S
    return loss

  def get_free_energy(self,run_sim=True):
    H     = self.get_H_expectation_by_matrix()
    S     = self.get_entropy(use_true=True,run_sim=False)
    F     = H - (1/self.beta)*S

    return F

  def get_H_expectation_by_matrix(self):
    H_expectation = np.real(np.trace(np.dot(self.Hc_matrix,self.current_rho)))
    return H_expectation

  def get_H_expectation(self,thermal=False,expectation_method=None,hamiltonian=None):

    if hamiltonian is None:
      hamiltonian = self.Hc

    if thermal:
      expectation = self.calc_mixed_expectation(self.circuit,
                                          operators=hamiltonian,
                                          symbol_names=self.symbols,
                                          symbol_values=self.thetas_tf)
      
    if "analytical" in self.expectation_method:
      if expectation_method is None:
        expectation = self.calc_pure_expectation(self.circuit,
                                            operators=hamiltonian,
                                            symbol_names=self.symbols,
                                            symbol_values=self.thetas_tf)
      else:
        expectation = expectation_method(self.circuit,
                                            operators=hamiltonian,
                                            symbol_names=self.symbols,
                                            symbol_values=self.thetas_tf)
    elif "sample" in self.expectation_method:
      if expectation_method is None:
        expectation = self.calc_pure_expectation(self.circuit,
                                            operators=hamiltonian,
                                            symbol_names=self.symbols,
                                            symbol_values=self.thetas_tf,
                                            repetitions=self.n_expectation_samples)
      else:
        expectation = expectation_method(self.circuit,
                                            operators=hamiltonian,
                                            symbol_names=self.symbols,
                                            symbol_values=self.thetas_tf,
                                            repetitions=self.n_expectation_samples)
    return expectation

  def get_entropy(self,epsilon=None, use_true=False,run_sim=True):
    if use_true:
      # Shannon definition
      rho = self.forward() if run_sim else self.current_rho
      eig = np.real(np.linalg.eigvalsh(rho)) 
      eig = np.array([np.maximum(self.epsilon,x) for x in eig])
      S   = -eig.dot(np.log(eig))
    else:
      # Approximation
      if epsilon is not None:
        lambda_ = self.lambda_ + epsilon
      else:
        lambda_ = self.lambda_
      d   = self.d
      m   = self.n_gates

      i1  = (1-lambda_)**m + (1-(1-lambda_)**m)/d

      i2  = (1-(1-lambda_)**m)/d

      S   = i1*np.log(i1 + self.epsilon)
      S   += (d-1)*i2*np.log(i2 + self.epsilon)
      S   = -S

    return S

  def get_fin_diff_sim(self,epsilon):
    if self.expectation_method == "analytical":
      fin_dif_sim = tfq.layers.Expectation(backend=DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_+epsilon))) 
    if self.expectation_method == "sampled":
      fin_dif_sim = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ForwardDifference(grid_spacing=0.01),backend=DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_+epsilon)))
    if self.expectation_method == "safe_sampled":
      fin_dif_sim = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ParameterShift(),backend=DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_+epsilon)))

    return fin_dif_sim
    
  def get_lambda_gradient(self, H_fd_order=2,S_fd_order=2):
    lambda_           = self.lambda_
    d           = self.d
    n_g         = self.n_gates

    # ENERGY
    # Finite difference coefficients taken from: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    fd_coeffs   = [[1/2],[2/3,-1/12],[3/4,-3/20,1/60],[4/5,-1/5,4/105,-1/280]]
    
    f_fd_coeffs = fd_coeffs[H_fd_order]
    nabla_H     = 0
    for k,c in enumerate(f_fd_coeffs):
      simulator   = self.get_fin_diff_sim((k+1)*self.epsilon)
      forward     = self.get_H_expectation(expectation_method=simulator)

      simulator   = self.get_fin_diff_sim(-(k+1)*self.epsilon)
      backward    = self.get_H_expectation(expectation_method=simulator)

      nabla_H     += c*(forward-backward)

    nabla_H       /= self.epsilon

    # ENTROPY
    if S_fd_order > -1:
      f_fd_coeffs = fd_coeffs[S_fd_order]
      nabla_S     = 0
      for k,c in enumerate(f_fd_coeffs):
        forward     = self.get_entropy(epsilon=(k+1)*self.epsilon)
        backward    = self.get_entropy(epsilon=-(k+1)*self.epsilon)
        nabla_S     += c*(forward-backward)

      nabla_S       /= self.epsilon

    else:

      nabla_S     = self.N
      nabla_S     *= (d-1)*n_g*(1-lambda_)**(n_g-1)
      nabla_S     *= (-np.log(-(-1+(1-lambda_)**n_g)/d) + np.log((1-(1-lambda_)**n_g+d*(1-lambda_)**n_g)/d))
      nabla_S     *= 1/d

    nabla_F     = nabla_H - (1/self.beta)*nabla_S

    if self.verbøgse:
      print("Fidelity = %.2E nabla F = %.2E nabla S = %.2E nabla H = %.2E lambda = %.2E" % (self.thermal_state_fidelity(),nabla_F.numpy()[0][0], nabla_S, nabla_H.numpy()[0][0], lambda_))
    
    return tf.convert_to_tensor(nabla_F.numpy()[0],dtype=tf.float32)

  @tf.function
  def get_theta_gradients(self):
    with tf.GradientTape() as g:
      g.watch(self.thetas_tf)
      H_expectation = self.get_H_expectation()
    gradients = g.gradient(H_expectation, self.thetas_tf)
    return gradients

  def thermal_state_fidelity(self,run_sim=True):
    rho = self.forward() if run_sim else self.current_rho
    return cirq.fidelity(rho,self.thermal_matrix)

  def thermal_state_trace_distance(self,run_sim=True):
    rho   = self.forward() if run_sim else self.current_rho
    diff  = rho-self.thermal_matrix
    return np.real((1/2)*np.trace(np.dot(diff,diff)))
      
  def set_optimizers(self,theta_lr=-1,lambda_lr=-1):
    if theta_lr > 0:
      self.theta_optimizer.learning_rate = theta_lr
    if lambda_lr > 0:
      self.lambda_optimizer.learning_rate = lambda_lr

  def lambda_loss(self,x):
    x                 = np.minimum(np.maximum(3*self.epsilon,x),1-3*self.epsilon)
    self.lambda_      = x
    self.simulator    = DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_),ignore_measurement_results=True)
    self.build_simulator()
    self.current_rho  = self.forward()
    return self.get_loss()

  def train(self,n_epochs=50,theta_lr=-1,lambda_lr=-1,stop_fidelity=1.0,use_lambda_gradient=True,try_lambda_jump=False, lambda_jump = 0.05):
    self.set_optimizers(theta_lr,lambda_lr)

    if use_lambda_gradient:
      # Calculate lambda gradient
      d_dlambda_ = self.get_lambda_gradient(H_fd_order=0,S_fd_order=0)
    else:
      l   = self.lambda_
      res = minimize(self.lambda_loss,
              l,
              method=p.optim, # Nelder-Mead BFGS L-BFGS-B, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
              options={'maxiter': 1})
    
    
    # Calculate theta gradients 
    d_dtheta = self.get_theta_gradients()
    # Update ansatz theta
    self.theta_optimizer.apply_gradients(zip([d_dtheta], [self.thetas_tf]))

    if use_lambda_gradient:
      # Update noise parameter lambda
      self.lambda_optimizer.apply_gradients(zip([d_dlambda_], [self.lambda_tf]))
      # Update numpy/tensorflow integration
      self.lambda_    = np.minimum(np.maximum(3*self.epsilon,np.squeeze(self.lambda_tf.numpy())),1-3*self.epsilon)
      # Try to jump out of max
      if try_lambda_jump:
        self.lambda_  = np.minimum(np.maximum(3*self.epsilon,self.lambda_ + lambda_jump),1-3*self.epsilon)
    else:
      self.lambda_ 		= np.minimum(np.maximum(3*self.epsilon,res.x*self.lambda_lr + (1-self.lambda_lr)*self.lambda_),1-3*self.epsilon)
    
    self.thetas     = np.squeeze(self.thetas_tf.numpy())
    self.lambda_tf  = tf.Variable(tf.convert_to_tensor([self.lambda_],dtype=tf.float32))

    # Update simulator backend
    self.simulator  = DensityMatrixSimulator(noise=cirq.depolarize(self.lambda_),ignore_measurement_results=True)
    self.build_simulator()

    # Get current circuit output
    self.current_rho = self.forward()

    # Save metrics
    self.loss.append(self.get_loss())
    self.free_energy.append(self.get_free_energy(run_sim=False))
    self.entropy.append(self.get_entropy())
    self.entropy_true.append(self.get_entropy(use_true=True,run_sim=False))
    self.energy.append(self.get_H_expectation_by_matrix())
    self.lambdas.append(self.lambda_)
    self.fidelity.append(self.thermal_state_fidelity(run_sim=False))
    self.tr_dist.append(self.thermal_state_trace_distance(run_sim=False))

    # Save best model so far
    if self.loss[-1] < self.best_loss:
      self.best_loss          = self.loss[-1]
      self.best_thetas        = self.thetas.copy()
      self.best_lambda        = self.lambda_

    # Store spectra
    self.rho_circuit_eigenvalues    = np.real(np.linalg.eigvals(self.current_rho))
    self.rho_thermal_eigenvalues    = np.real(np.linalg.eigvals(self.thermal_matrix))
    
    
def save_performance_lean(qc,filename="file"):
  qc_ = {}
  for d in qc.__dict__:
    if d in ["loss","free_energy","entropy","entropy_true","energy","fidelity","tr_dist","rho_e0","rho_e1","rho_gap","H_e0","H_e1","H_gap","H_target","S_target","F_target"]:
      qc_.update({d: getattr(qc, d)})
  f = open(filename,"wb")
  pickle.dump(qc_,f)
  f.close()
  
def save_pkl(obj,filename="file"):
  f = open(filename + ".pkl","wb")
  pickle.dump(obj,f)
  f.close()

def save_qc(qc,filename="file"):
  qc_ = {}
  for d in qc.__dict__:
    if d not in ["mix_simulator","calc_pure_expectation","calc_mixed_expectation","circuit","Hc","thermal"] and "cirq" not in str(getattr(qc, d)):
      qc_.update({d: getattr(qc, d)})
  f = open(filename + ".pkl","wb")
  pickle.dump(qc_,f)
  f.close()

def load_qc(filename):
  qc = QC(ansatz="qaoa-r")
  # Open and load file
  with open(filename, "rb") as input_file:
    qc_ = pickle.load(input_file)
  # Translate to empty QC object
  for key, value in qc_.items():
    setattr(qc,key,value)
  qc.update_state()
  
  return qc

# """ Simple Use Case example """
#savepth   = "drive/My Drive/"
# Circuit initialization
#  qc            = QC(N=4,
#                  L=2,
#                  hamiltonian=["ZZ","XX"],
#                  beta=1.0,
#                  savepth=savepth,
#                  seed=0,
#                  ansatz="qaoa-f")
#  for e in tqdm(range(50),leave=False):
#    qc.train(use_lambda_gradient = p.optim == "gradient")
#    
#  qc.plot_history()
