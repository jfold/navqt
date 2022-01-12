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
from scipy.linalg import expm, logm, sqrtm
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from IPython.display import clear_output
from src.cirquit import Circuit
from typing import Any, List, Optional, Tuple, Union, Dict
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 14
# print("GPU possibility:", tf.test.is_gpu_available)
# print("Intra threads:", tf.config.threading.get_intra_op_parallelism_threads())
# print("Inter threads:", tf.config.threading.get_inter_op_parallelism_threads())
