from navqt import NAVQT
import os
import sys
import numpy as np
import time

def run():
	start = time.time()
	"""### Import string arguments"""
	try:
		args        = sys.argv[-1].split("|")
	except:
		args        = []
	print("ARGUMENTS")
	print(args)
	print("---------")
	qc  		= NAVQT()
	kwargs 	= {"savepth":"./results/"}
	for _,arg in enumerate(args):
		try: 
			var = arg.split('=')[0]
			
			if type(getattr(qc,var)) is int:
				val = int(arg.split('=')[1])
			elif type(getattr(qc,var)) is bool:
				val = arg.split('=')[1].lower() == "true"
			elif type(getattr(qc,var)) is float or type(getattr(qc,var)) is np.ndarray:
				val = float(arg.split('=')[1])
			elif type(getattr(qc,var)) is str:
				val = arg.split('=')[1]
			else:
				print("COULD NOT FIND VARIABLE:",var)
			kwargs.update({var: val})
			print(var,":",val)
		except:
			print("Trouble with " + arg) 

	"""Train w/ gradients"""
	qc  	= NAVQT(**kwargs)
	cwd 	= os.getcwd() + "/results/"
	if os.path.isdir(cwd) and not os.path.isdir(cwd + qc.experiment):
		os.mkdir(cwd + qc.experiment)
		print("Created new folder named ", cwd + qc.experiment)

	qc.savepth = cwd + qc.experiment + "/" 
	if not os.path.isfile(qc.savepth + "history---"+qc.settings+".pdf"):
		print(qc)
		qc.train(n_epochs=qc.max_iter,early_stop=True,grad_norm=True) 
		qc.plot_history(save=True)
		print("Succesfully saved file(s) to:", qc.savepth + "history---"+qc.settings+".*")
	else:
		print("File exists!")


if __name__ == '__main__':
	run()