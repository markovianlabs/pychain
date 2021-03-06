"""
Example demonstrating the usage for the linear-fit scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../src')
from mcmc import MCMC
import matplotlib
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 
#----------------------------------------------------------

class Anim(MCMC):
	"""
	Extends the original MCMC class to sample the parameters of a linear model.

	Parameters
	-----------
	MCMC (Class): Parent MCMC class.
	m (Float): Feducial value of the slope of the linear data.
	c (Float): Feducial value of the intercept of the linear data.
	RedStd (Float): Feducial value of the standard deviation of the linear data.
	"""
	def __init__(self, m=5.0, c=25.0, RedStd=15.0, alpha=1.5, delay=0.001):
		"""
		Instantiates the class by synthetically generating data.
		"""
		MCMC.__init__(self, TargetAcceptedPoints=5000, \
				NumberOfParams=2, Mins=[0.0,20.0], Maxs=[10.0,30.0], SDs=[1.0,2.0], alpha=alpha,\
				write2file=True, outputfilename='chain.mcmc', randomseed=250192)		

		self.m = m
		self.c = c
		self.X=np.linspace(-10, 10, 25)
		self.delta = np.random.uniform(low=-1*RedStd, high=RedStd, size=len(self.X))
		self.Y = (m*self.X + c) + self.delta

		self.delay = delay

#----------------------------------------------------------

	def FittingFunction(self, Params):
		"""
		Parametric form of the model.

		Parameters
		----------
		Params (1d array): Numpy array containing values of the parameters. 

		Returns
		-------
		model values (y = mx + c)
		"""
		return Params[0]*self.X + Params[1]

#----------------------------------------------------------

	def chisquare(self, Params):
		"""
		Computes Chi-square.

		Parameters
		----------
		Params (1d array): Numpy array containing values of the parameters. 

		Returns
		-------
		chi square.
		"""
		kisquare = ((self.Y-self.FittingFunction(Params))/self.delta)**2
		return np.sum(kisquare)

#----------------------------------------------------------

	def MainChain(self):
		"""
		Runs the chain.

		Returns
		-------
		Acceptance rate.
		"""

		f, axarr = plt.subplots(1, 2, figsize=(16,7))


		axarr[1].set_xlim(4, 6)
		axarr[1].set_ylim(22, 28)

		axarr[1].set_xlabel('$\mathtt{m}$', fontsize=22)
		axarr[1].set_ylabel('$\mathtt{c}$', fontsize=22)


		# Initialising the chain
		OldStep = self.FirstStep()
		Oldchi2 = self.chisquare(OldStep)
		Bestchi2 = Oldchi2

		# Preparing output file
		# if self.write2file:
		# 	outfile = open(self.outputfilename,'w')
		# 	writestring = '%1.6f \t'*self.NumberOfParams

		# Initialising multiplicity and accepted number of points.
		multiplicity = 0
		acceptedpoints = 0

		xlist=[]
		ylist=[]
		i=0
		# Chain starts here...
		# for i in range(self.NumberOfSteps):
		while True:
			i += 1
			if acceptedpoints == self.TargetAcceptedPoints:
				break

			multiplicity += 1

			# Generating next step and its chi-square
			NewStep = self.NextStep(OldStep)
			Newchi2 = self.chisquare(NewStep)

		# 	# Checking if it is to be accepted.
			GoodPoint = self.MetropolisHastings(Oldchi2,Newchi2)

			# Updating step scale using a threshold chi-square.
			if Newchi2<2*len(self.Y):
						self.CovMat = self.alpha*np.diag(self.SD**2)

			if GoodPoint:
				# Updating best chi-square so far in the chain.
				if Newchi2<Bestchi2:
					BestStep = NewStep
					Bestchi2 = Newchi2
					print "Best chi-square and step so far: ", Bestchi2, NewStep

				acceptedpoints += 1
				multiplicity = 0

				# Updating number of accepted points.
				axarr[0].clear()
				axarr[0].set_xlim(-10, 10)
				axarr[0].set_ylim(-50, 100)
				axarr[0].errorbar(self.X, self.Y, self.delta, color='k', ms=8, ls='', marker='s')
				axarr[0].plot(self.X, self.FittingFunction(OldStep), 'k', ls='-', lw=2)
				axarr[0].set_title('$\mathtt{Step:\ %i,\ AccPoints: %i}$'%(i+1, acceptedpoints), fontsize=22)
				xlist.append(OldStep[0])
				ylist.append(OldStep[1])
				axarr[1].plot(xlist[-3:-1], ylist[-3:-1], 'k', lw=0.2)

				axarr[1].set_title('$\mathtt{m=%1.3f,\ c=%1.3f,\ \chi^2=%1.3f}$'%(OldStep[0], OldStep[1], Oldchi2), fontsize=22)
				axarr[0].set_xlabel('$\mathtt{X}$', fontsize=22)
				axarr[0].set_ylabel('$\mathtt{Y}$', fontsize=22)
				plt.pause(self.delay)
				plt.draw()

				# Updating the old step. 
				OldStep = NewStep
				Oldchi2 = Newchi2
			else:
				continue
		# Writing Best chi-square of the full chain and the acceptance ratio.
		print "Best chi square: %1.5f"%Bestchi2
		print "Acceptance Ratio: %1.5f"%(float(acceptedpoints)/i) 



		axarr[0].clear()
		axarr[0].set_xlim(-10, 10)
		axarr[0].set_ylim(-50, 100)
		axarr[0].errorbar(self.X, self.Y, self.delta, color='k', ms=8, ls='', marker='s')
		axarr[0].plot(self.X, self.FittingFunction(BestStep), 'k', ls='-', lw=2)
		axarr[0].set_title('$\mathtt{Step:\ %i,\ AccPoints: %i}$'%(i+1, acceptedpoints), fontsize=22)
		axarr[1].plot(BestStep[0], BestStep[1], 'or', ms=12, label='$\mathtt{BestFit}$')
		axarr[1].plot(self.m, self.c, 'og', ms=12, label='$\mathtt{Fiducial}$')
		axarr[1].legend(loc=1, fontsize=18, numpoints=1)
		axarr[1].set_title('$\mathtt{m=%1.3f,\ c=%1.3f,\ \chi^2=%1.3f}$'%(BestStep[0], BestStep[1], Bestchi2), fontsize=22)
		axarr[0].set_xlabel('$\mathtt{X}$', fontsize=22)
		axarr[0].set_ylabel('$\mathtt{Y}$', fontsize=22)
		plt.show()

		return float(acceptedpoints)/i


#----------------------------------------------------------



#==============================================================================

if __name__=="__main__":
	if len(sys.argv) == 1:
		alpha = 1.0
		delay = 0.001
	elif len(sys.argv) == 3:
		alpha = float(sys.argv[1])
		delay = float(sys.argv[2])
	else:
		print "Syntax: python anim.py <alpha> <delay>"
		sys.exit()
		
	co = Anim(alpha=alpha, delay=delay)
	co.MainChain()
