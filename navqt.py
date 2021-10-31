import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
import tensorflow_probability as tfp
import tensorflow_quantum as tfq
import cirq 
from cirq import Simulator, DensityMatrixSimulator
import sympy
import numpy as np
import pandas as pd
from scipy.linalg import expm,logm,sqrtm
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from IPython.display import clear_output
from cirquit import Circuit
print("GPU possibility:",tf.test.is_gpu_available)
print("Intra threads:",tf.config.threading.get_intra_op_parallelism_threads()) 
print("Inter threads:",tf.config.threading.get_inter_op_parallelism_threads())






class NAVQT(Circuit):
	"""Noise-assisted variational quantum thermalizer (NAVQT).
		The implementation is based on Foldager et al. (2021) 
		"Noise-Assisted Variational Quantum Thermalization" and allows for 
		parameterizing both unitary gates and depolerization channels in 
		quantum circuits. 
		"""
	def __init__(self, **kwargs):    
		super().__init__()
		self.set_defaults()
		self.__dict__.update(kwargs)
		assert self.N >= 3
		""" SETTINGS """
		self.L                      = self.L if self.L > 0 else int(np.ceil(self.N/2))
		self.K                      = self.K if self.K > 0 else int(np.maximum(100,np.minimum(self.N*500,100*self.N*(1/self.beta)))) # int(self.N/2)*1000
		self.opt_true_G							= True if self.multilambda else self.opt_true_G 
		self.d                      = 2**self.N
		self.m                      = self.L
		self.unitary_optimizer      = getattr(tf.optimizers, self.gamma_optim)(learning_rate=self.gamma_lr)
		self.noise_optimizer        = getattr(tf.optimizers, self.p_err_optim)(learning_rate=self.p_err_annealing_lr) if self.annealing else getattr(tf.optimizers, self.p_err_optim )(learning_rate=self.p_err_lr)
		self.qubits                 = cirq.GridQubit.rect(self.N, 1)
		np.random.seed(self.H_seed)
		tf.random.set_seed(self.H_seed)
		self.H                      = self.get_H()
		np.random.seed(self.gamma_seed)
		tf.random.set_seed(self.gamma_seed)
		self.p_err              		= np.random.uniform(low=1e-8,high=1e-1,size=(1,)) if self.lambda_random else np.array([self.p_err])
		self.p_errs             		= np.random.uniform(low=1e-8,high=self.p_err,size=(self.N,)) if self.lambda_random else np.array([self.p_err]*self.N).squeeze() 
		self.p_err_tf             	= tf.Variable(self.p_err,dtype=tf.float32)
		self.p_errs_tf             	= tf.Variable(self.p_errs,dtype=tf.float32)
		self.calc_targets()
		self.ngrad_H_gamma 				  = np.array([0.0])
		self.grad_H_p_err 					= np.array([0.0])
		self.grad_TS_p_err 					= np.array([0.0])
		self.is_trained 						= False
		""" Create circuit """
		self.gamma_symbols        = []
		self.error_symbols        = []
		self.create_circuits()
		self.gammas               = self.gamma_init()
		self.errors               = self.sample_errors()
		""" Optimal parameters init """
		self.best_gammas          = tf.identity(self.gammas)
		self.best_p_err         	= self.p_err
		""" Save settings """
		self.settings             = "N-{}--H-{}--L-{}--A-{}--K-{}--beta-{}--p-err-{:.2E}--g-lr-{:.2E}--l-lr-{:.2E}--cheat-{}--multi-{}--g-seed-{}".format(self.N, ## p-{:.2E}--g-a-lr-{:.2E}--l-a-lr-{:.2E}--annealing-{}--
																																self.model,
																																self.L,
																																self.ansatz,
																																self.K,
																																self.beta,
																																np.linalg.norm(self.p_err,ord=2),
																																# self.gamma_annealing_lr,
																																# self.p_err_annealing_lr,
																																self.gamma_lr,
																																self.p_err_lr,
																																# self.annealing,
																																self.opt_true_G,
																																self.multilambda,
																																self.gamma_seed).replace(".","-")
		self.update_history(row=0)

	def __str__(self):
		j 		= 20
		str_ = "Noise-Assisted Variational Quantum Thermalizer \n\r"
		str_ += f"System: ".ljust(j)+ f" N= {self.N}".ljust(j) + f"H= {self.model}".ljust(j) + f"beta= {self.beta}".ljust(j)+ f"seed= {self.H_seed}  \n\r"+ f"Tr[rho]= {self.tr_target_rho}  \n\r"
		str_ += f"Circuit: ".ljust(j)+f"Ansatz= {self.ansatz}".ljust(j) + f"L= {self.L}".ljust(j) + f"seed= {self.gamma_seed}  \n\r"
		str_ += f"Trained: {self.is_trained} \n\r"
		if self.is_trained:
			F 		= np.nanmax(self.history["F"])
			str_ += f"Thermal state Fidelity: {F}"
		return str_

	def set_defaults(self):
		# =============================================================================
		# Default settings
		# =============================================================================
		# BOOLEANS
		self.DMS 								= False
		self.p_err_lr_decay 		= False
		self.gamma_lr_decay 		= False
		self.annealing          = False
		self.opt_true_G         = False
		self.multilambda        = False
		self.lambda_random      = False
		# INTEGERS
		self.N                	= 3
		self.L 									= 0
		self.K 									= 0
		self.H_seed             = 0
		self.gamma_seed         = 0
		self.gamma_fd_order     = 0
		self.p_err_fd_order     = 0
		self.max_iter 					= 1000
		# FLOATS
		self.beta               = 10.0
		self.lb              		= 1e-8
		self.ub              		= 1-self.lb
		self.p_err 							= 1e-1
		self.gamma_lr           = 4e-3
		self.gamma_annealing_lr = 4e-3
		self.p_err_lr           = 1e-3
		self.p_err_annealing_lr = 0.0
		self.epsilon 						= 1e-2
		self.gamma_std 					= 1e-10
		self.fd_coeffs       		= [[1/2],[2/3,-1/12],[3/4,-3/20,1/60],[4/5,-1/5,4/105,-1/280]] # Finite difference coefficients taken from: https://en.wikipedia.org/wiki/Finite_difference_coefficient
		# STRINGS
		self.experiment         = 'test'
		self.model              = 'IC-u'
		self.ansatz 						= "qaoa-r"
		self.p_err_optim 				= "Adam"
		self.gamma_optim 				= "Adam"
		self.savepth     				= "/"

	def get_H_from_rho(self):
		pass

	def get_H(self,loc=0,scale=1):
		H_type, coeff_type = self.model.split("-")
		H   = cirq.PauliSum()

		if H_type == "IC" or H_type == "TFI":
			if coeff_type == "u":
				J_coeffs 		= np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
				h_z_coeffs 	= np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
				h_x_coeffs 	= np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
			elif coeff_type == "r":
				np.random.seed(self.H_seed)
				tf.random.set_seed(self.H_seed)
				J_coeffs 		= tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed)
				h_z_coeffs 	= tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed + 100)
				h_x_coeffs 	= tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed + 200)
			
				J_coeffs 			= tf.linalg.normalize(J_coeffs, ord='euclidean')[0].numpy()
				h_z_coeffs 		= tf.linalg.normalize(h_z_coeffs, ord='euclidean')[0].numpy()
				h_x_coeffs 		= tf.linalg.normalize(h_x_coeffs, ord='euclidean')[0].numpy()
		elif H_type == "Heisenberg":
			if coeff_type == "u":
				J_XX_coeffs = np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
				J_YY_coeffs = np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
				J_ZZ_coeffs = np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
				h_X_coeffs  = np.ones((self.N,))#tf.linalg.normalize(, ord='euclidean')[0].numpy()
			elif coeff_type == "r":
				np.random.seed(self.H_seed)
				tf.random.set_seed(self.H_seed)
				J_XX_coeffs = tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed)
				J_YY_coeffs = tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed  + 100)
				J_ZZ_coeffs = tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed  + 200)
				h_X_coeffs  = tfp.distributions.Normal(loc=loc,scale=scale).sample(sample_shape=(self.N,),seed=self.H_seed  + 1000)
		
				J_XX_coeffs 	= tf.linalg.normalize(J_XX_coeffs, ord='euclidean')[0].numpy()
				J_YY_coeffs 	= tf.linalg.normalize(J_YY_coeffs, ord='euclidean')[0].numpy()
				J_ZZ_coeffs 	= tf.linalg.normalize(J_ZZ_coeffs, ord='euclidean')[0].numpy()
				h_X_coeffs 		= tf.linalg.normalize(h_X_coeffs, ord='euclidean')[0].numpy()
		elif H_type == "RBM":
			pass

		if H_type == "IC":
			for i in range(self.N):
				if i < self.N-1:
					H  -= float(J_coeffs[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[i+1])
				H     -= float(h_z_coeffs[i])*cirq.Z(self.qubits[i])
			H  -= float(J_coeffs[-1])*cirq.Z(self.qubits[-1])*cirq.Z(self.qubits[0])

		elif H_type == "TFI":
			for i in range(self.N):
				if i < self.N-1:
					H  -= float(J_coeffs[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[i+1])
				H     -= float(h_x_coeffs[i])*cirq.X(self.qubits[i])
				H     -= float(h_z_coeffs[i])*cirq.Z(self.qubits[i])
			H  -= float(J_coeffs[-1])*cirq.Z(self.qubits[-1])*cirq.Z(self.qubits[0])

		elif H_type == "Heisenberg":
			for i in range(self.N-1):
				if i < self.N-1:
					H  -= float(J_XX_coeffs[i])*cirq.X(self.qubits[i])*cirq.X(self.qubits[i+1])
					H  -= float(J_YY_coeffs[i])*cirq.Y(self.qubits[i])*cirq.Y(self.qubits[i+1])
					H  -= float(J_ZZ_coeffs[i])*cirq.Z(self.qubits[i])*cirq.Z(self.qubits[i+1])
				H     -= float(h_X_coeffs[i])*cirq.X(self.qubits[i])
			H  -= float(J_XX_coeffs[-1])*cirq.X(self.qubits[-1])*cirq.X(self.qubits[0])
			H  -= float(J_YY_coeffs[-1])*cirq.Y(self.qubits[-1])*cirq.Y(self.qubits[0])
			H  -= float(J_ZZ_coeffs[-1])*cirq.Z(self.qubits[-1])*cirq.Z(self.qubits[0])

		self.H_str            = str(H).replace(", 0))","").replace("((","_")
	
		H_matrix  						= H.matrix()
		A         						= -self.beta*H_matrix
		eigs   								= np.linalg.eigvals(A).real
		c         						= np.max(eigs)
		expos     						= eigs - c
		log_rho_t 						= A - np.identity(self.d)*(c+np.log(np.sum(np.exp(expos))))
		rho_target 						= expm(log_rho_t)
		self.target_spectrum  = np.linalg.eigvalsh(rho_target).real
		w, v 									= np.linalg.eigh(rho_target)
		w 										= np.sqrt(np.maximum(w, 0))
		sqrt_rho_target 			= (v * w).dot(v.conj().T)
		self.target_rho   		= tf.cast(rho_target,dtype=tf.complex64)
		self.target_rho_sqrtm = tf.cast(sqrt_rho_target,dtype=tf.complex64)
		self.tr_target_rho 		= np.trace(rho_target).real
	
		if not np.isclose(self.tr_target_rho, 1.0,rtol=1e-2):
			raise ValueError("TRACE OF THERMAL STATE NOT 1: Tr = " + str(self.tr_target_rho))
		return H

	def H_expectation(self,p_err=None,gammas=None):
		if self.DMS:
			p_err 				= self.p_err if p_err is None else p_err
			simulator 		= DensityMatrixSimulator(noise=cirq.depolarize(p_err))
			expectation 	= tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint(),
																				 backend=simulator)(
																					self.unitary,
																					operators=self.H, 
																					symbol_names=self.gamma_symbols,
																					symbol_values=self.gammas) 
		else:
			values      = tf.concat([tf.tile(self.gammas, tf.constant([self.K,1])), self.errors], 1)
			expectation = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint())(
																self.circuits,
																operators=self.H, 
																symbol_names=self.symbols,
																symbol_values=values)
		return tf.reduce_mean(expectation)

	def calc_targets(self):
		eig             = tf.linalg.eigvalsh(self.target_rho).numpy().real
		eig             = np.sort(np.array([x for x in eig if x > self.lb]))
		self.S_target   = -eig.dot(np.log(eig))
		self.H_target 	= tf.linalg.trace(tf.linalg.matmul(self.target_rho,self.H.matrix())).numpy().real
		self.G_target   = self.H_target - (1/self.beta)*self.S_target
		self.S_max      = np.log(self.d)
		
		eig             = np.sort(tf.linalg.eigvalsh(self.H.matrix()).numpy().real)
		self.H_0        = eig[0]
		self.H_1        = eig[1]
		self.E_gap      = self.H_1-self.H_0
		self.H_hardness = (self.H_1-self.H_0)/np.abs(self.H_0)

	def fidelity(self):
		mat         = tf.linalg.matmul(tf.linalg.matmul(self.target_rho_sqrtm,self.rho),self.target_rho_sqrtm).numpy()
		F           = tf.reduce_sum(np.sqrt(np.clip(np.linalg.eigvals(mat).real,0,1)))**2
		F 					= F.numpy().real
		if np.isclose(F,1,rtol=1e-3):
			F 				= 1.0
		if np.isclose(F,0,rtol=1e-3):
			F 				= 0.0
		return F if 0 <= F <= 1  else np.nan

	def trace_distance(self):
		T = (1/2)*tf.linalg.trace(tf.linalg.matmul(self.rho-self.target_rho,self.rho-self.target_rho))
		return T.numpy().real if 0 <= T.numpy().real <= 1 else np.nan

	def update_history(self,row,save=False):
		self.rho      = self.run_circuit()
		F             = self.fidelity()
		T             = self.trace_distance()
		S             = self.entropy(approximation=False,rho=self.rho)
		H             = self.H_expectation().numpy().real
		G             = H - (1/self.beta)*S
		p_err         = np.linalg.norm(self.p_errs,ord=2) if self.multilambda else self.p_err
		RMSE          = np.abs(G-self.G_target)
		S_approx      = np.nan if self.multilambda else self.entropy(approximation=True)
		G_approx      = np.nan if self.multilambda else H - (1/self.beta)*S_approx 
		data          = [F,T,RMSE,G,G_approx,H,S,S_approx,p_err,self.ngrad_H_gamma,tf.norm(self.grad_H_p_err,ord='euclidean'),tf.norm(self.grad_TS_p_err,ord='euclidean')]
		if row == 0:
			self.history  = pd.DataFrame(columns=["F","T","RMSE","G","G_approx","H","S","S_approx","p_err","ngrad_H_gamma","grad_H_p_err","grad_TS_p_err"])
		self.history.loc[row] = data
		if save:
			self.history.to_csv(self.savepth+"history---"+self.settings+".csv")

	def G_min_fun(self,x):
		self.gammas     = tf.convert_to_tensor(np.expand_dims(x[:-1],axis=1).T,dtype=tf.float32)
		self.p_err      = x[-1]
		self.p_err_tf   = tf.Variable(self.p_err,dtype=tf.float32)
		self.errors     = self.sample_errors()
		self.update_history(self.history.shape[0])
		return self.history["G"].values[-1]

	@tf.function
	def gamma_gradients(self):
		with tf.GradientTape() as g:
			g.watch(self.gammas)
			E       = self.H_expectation()
		gradients = g.gradient(E, self.gammas)
		if self.DMS:
			return gradients
		return tf.transpose(tf.expand_dims(tf.reduce_mean(gradients,axis=0),axis=1))

	def gamma_fd_nabla_S(self):
		coeffs        = self.fd_coeffs[self.gamma_fd_order]
		gammas        = self.gammas.numpy()
		nabla_S       = np.zeros(len(gammas))
		for i_g,gamma in enumerate(gammas):
			for k,c in enumerate(coeffs):
				gammas[i_g] = gamma + (k+1)*self.epsilon
				rho    			= self.run_circuit(gammas_=gammas)
				forward     = self.entropy(approximation=False,rho=rho)
				
				gammas[i_g] = gamma - (k+1)*self.epsilon
				rho    			= self.run_circuit(gammas_=gammas)
				backward    = self.entropy(approximation=False,rho=rho)

				nabla_S[i_g]+= c*(forward-backward)

			nabla_S[i_g]  /= self.epsilon
			gammas        = self.gammas.numpy()
		return nabla_S

	def entropy(self,approximation=False, rho=None):
		if approximation:
			i1  = (1-self.p_err)**self.m + (1-(1-self.p_err)**self.m)/self.d
			i2  = (1-(1-self.p_err)**self.m)/self.d
			S   = i1*np.log(i1 + self.epsilon)
			S   += (self.d-1)*i2*np.log(i2 + self.epsilon)
			S   = -S
		else:
			if rho is None:
				rho = self.run_circuit()
			eig = tf.linalg.eigvalsh(rho).numpy().real
			eig = np.array([x for x in eig if x > 1e-20])
			S   = -eig.dot(np.log(eig))
		return S

	def error_gradient(self): # Consider moving entropy into same forward equation
		coeffs        = self.fd_coeffs[self.p_err_fd_order]
		""" ENERGY: Finite-difference"""
		nabla_H       = 0
		for k,c in enumerate(coeffs):
			p_1         = np.maximum(np.minimum(self.ub,self.p_err + (k+1)*self.epsilon),self.lb)
			self.errors = self.sample_errors(p_err=p_1)
			forward     = self.H_expectation(p_err=p_1)
			
			p_2         = np.maximum(np.minimum(self.ub,self.p_err - (k+1)*self.epsilon),self.lb)
			self.errors = self.sample_errors(p_err=p_2)
			backward    = self.H_expectation(p_err=p_2)

			nabla_H     += c*(forward-backward)

		nabla_H       /= self.epsilon
		
		""" ENTROPY: gradient  """
		if self.opt_true_G:                     # CHEATING
			nabla_S       = 0
			for k,c in enumerate(coeffs):
				p_1         = np.maximum(np.minimum(self.ub,self.p_err + (k+1)*self.epsilon),self.lb)
				self.errors = self.sample_errors(p_err=p_1)
				forward     = self.entropy(approximation=False)
				
				p_2         = np.maximum(np.minimum(self.ub,self.p_err - (k+1)*self.epsilon),self.lb)
				self.errors = self.sample_errors(p_err=p_2)
				backward    = self.entropy(approximation=False)

				nabla_S     += c*(forward-backward)

			nabla_S       /= self.epsilon
			nabla_S				= np.array([nabla_S])
		else:                                 # APPROXIMATION
			t1     				= (self.d-1)*self.m*(1-self.p_err)**(self.m-1)
			inside 				= np.maximum(-(-1+(1-self.p_err)**self.m)/self.d,1e-20)
			t2     				= -np.log(inside) 
			t3						= np.log((1-(1-self.p_err)**self.m+self.d*(1-self.p_err)**self.m)/self.d)
			nabla_S 			= self.N*t1*(t2+t3)*1/self.d

		self.grad_H_p_err 	= nabla_H
		self.grad_TS_p_err 	= (1/self.beta)*nabla_S
		
		gradient     = nabla_H - (1/self.beta)*nabla_S

		return gradient

	def errors_gradient(self):
		coeffs        	= self.fd_coeffs[self.p_err_fd_order]
		""" ENERGY: Finite-difference"""
		gradients     	= np.zeros(len(self.p_errs))
		for i_p,p_err in enumerate(self.p_errs):
			for k,c in enumerate(coeffs):
				p_1         = np.maximum(np.minimum(self.ub,p_err + (k+1)*self.epsilon),self.lb)
				self.errors = self.sample_errors_i(i=i_p,p_i=p_1)
				nablas_H 		= self.H_expectation(p_err=p_1) 
				nablas_S 		= self.entropy(approximation=False)
				forward     = nablas_H - (1/self.beta)*nablas_S
				
				p_2         = np.maximum(np.minimum(self.ub,p_err - (k+1)*self.epsilon),self.lb)
				self.errors = self.sample_errors_i(i=i_p,p_i=p_2)
				nablas_H 		= self.H_expectation(p_err=p_2) 
				nablas_S 		= self.entropy(approximation=False)
				backward    = nablas_H - (1/self.beta)*nablas_S

				gradients[i_p]	+= c*(forward-backward)

			gradients[i_p] /= self.epsilon
			
		return tf.convert_to_tensor(gradients,dtype=tf.float32)

	def grid_search(self,n_evals=100):
		self.gs_f							= []
		pbar    							= tqdm(range(n_evals),leave=False)
		for e in pbar:
			p_gammas            = tfp.distributions.Uniform(low=-np.pi, high=np.pi)
			gammas 							= p_gammas.sample(sample_shape=(1,len(self.gamma_symbols)))
			p_errors            = tfp.distributions.Uniform(low=self.lb, high=1e-2).sample(sample_shape=(self.N))
			p_errs 							= tf.tile(p_errors,tf.constant([self.L*3])) if self.multilambda else np.array([(p_errors[0]/3) for _ in range( self.L*self.N*3)])
			bern_dist           = tfp.distributions.Bernoulli(probs=p_errs, dtype=tf.float32)
			errors             	= bern_dist.sample(self.K,seed=self.gamma_seed)
			values        			= tf.concat([tf.tile(gammas, tf.constant([self.K,1])), errors], 1)
			noisy_states  			= tfq.layers.State()(self.circuits, symbol_names=self.symbols,symbol_values=values).to_tensor()
			probs         			= tf.ones([self.K], dtype=tf.float32) / float(self.K)
			self.rho    				= self.pure_state_tensor_to_density_matrix(noisy_states, probs)
			self.gs_f.append(self.fidelity())
			pbar.set_postfix({"F_opt":np.nanmax(self.gs_f)})

	def train(self,n_epochs=300,early_stop=False,grad_norm=False,plot_it=False): 
		best_F  				= 0
		best_Fs 				= []
		best_G_approx  	= np.inf
		best_G_approxs 	= []
		pbar    				= tqdm(range(self.history.shape[0],self.history.shape[0]+n_epochs),leave=False)
		print("Training...")
		for e in pbar:
			# Gradients
			gamma_gradients   	= self.gamma_gradients()
			if self.opt_true_G:
				gamma_fd_nabla_S 	= self.gamma_fd_nabla_S()
				gamma_gradients 	= gamma_gradients - (1/self.beta)*gamma_fd_nabla_S
			
			self.ngrad_H_gamma 	= np.linalg.norm(gamma_gradients.numpy(),ord=2)
			p_err_gradient    	= self.errors_gradient() if self.multilambda else self.error_gradient()
			if grad_norm:
				gamma_gradients   /= tf.norm(gamma_gradients,ord='euclidean')
				p_err_gradient   	/= tf.norm(p_err_gradient,ord='euclidean')
			# Unitary update
			self.unitary_optimizer.apply_gradients(zip([gamma_gradients],[self.gammas]))
			# Noise update
			if self.multilambda:
				self.noise_optimizer.apply_gradients(zip([p_err_gradient],[self.p_errs_tf]))
				self.p_errs      	= np.minimum(self.ub,np.maximum(self.p_errs_tf.numpy(),self.lb)) 
				self.p_errs_tf   	= tf.Variable(self.p_errs,dtype=tf.float32)
			else:
				self.noise_optimizer.apply_gradients(zip([p_err_gradient],[self.p_err_tf]))
				self.p_err      	= np.minimum(self.ub,np.maximum(self.p_err_tf.numpy(),self.lb)) 
				self.p_err_tf   	= tf.Variable(self.p_err,dtype=tf.float32)

			# Update history
			self.errors     	= self.sample_errors()
			self.update_history(row=e,save=e%5==0)
	 		
			if plot_it:
				clear_output()
				self.plot_history()
	 
			# Optimal parameters
			cur_F         = self.history["F"].values[-1]
			cur_G_approx  = self.history["G_approx"].values[-1]
			if not np.isnan(cur_F) and cur_F > best_F:
				best_F 										= cur_F
				best_Fs                   = [best_F]
				best_G_approx 						= cur_G_approx
				best_G_approxs          	= [best_G_approx]
				self.best_gammas          = tf.identity(self.gammas)
				self.best_p_err           = self.p_err
			elif not np.isnan(cur_G_approx) and cur_G_approx < best_G_approx:
				best_G_approx 						= cur_G_approx
				best_G_approxs          	= [best_G_approx]
			else:	 # Counting towards early stopping; cheating but saving computational time
				best_Fs.append(best_F)
				best_G_approxs.append(best_G_approx)
				if early_stop and not self.annealing and len(best_Fs) > 100 and len(best_G_approxs) > 100:
					print("Terminated with convergence on F and G_approx")
					break
				elif self.annealing and len(best_Fs) > 20: # annealing switch
					self.unitary_optimizer  = getattr(tf.optimizers, self.gamma_optim)(learning_rate=self.gamma_annealing_lr)
					self.noise_optimizer    = getattr(tf.optimizers, self.p_err_optim)(learning_rate=self.p_err_lr)
					best_Fs                 = [best_F]
					print("Changed learning rates!")
					self.annealing          = False
			if np.isnan(cur_F):
				self.worst_gammas         = tf.identity(self.gammas)
				self.worst_p_err          = self.p_err
				print("Stopped due to F = nan")
				break
				
			# Early stopping
			if early_stop and cur_F > 0.999: # cheating but saving computational time
				print("Terminated with F > 0.999")
				break

			# Learning rate
			if self.p_err_lr_decay and e % 200 == 0:
				self.noise_optimizer.learning_rate = self.noise_optimizer.learning_rate*0.1
			if self.gamma_lr_decay and e % 200 == 0:
				self.unitary_optimizer.learning_rate *= 0.1

			# Progress bar
			if self.multilambda:
				pbar.set_postfix({"F_opt":np.round(best_F,3),"lambdas":self.p_errs})
			else:	
				pbar.set_postfix({"F_opt":np.round(best_F,3),"lambda":self.p_err})
		# Use optimal parameters
		self.gammas     = tf.Variable(self.best_gammas)
		self.p_err      = self.best_p_err
		self.p_err_tf   = tf.Variable(self.p_err,dtype=tf.float32)
		# self.errors     = self.sample_errors()
		# self.update_history(row=e+1,save=e%5==0)
		self.is_trained = True
		print("Done!")
		print("Maximum Fidelity:",np.round(best_F,4))
