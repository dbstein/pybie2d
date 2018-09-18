import numpy as np
import scipy as sp
import warnings
import scipy.sparse.linalg

from ..kernels.high_level.laplace import Laplace_Layer_Apply,				\
						Laplace_Layer_Singular_Apply, Laplace_Layer_Form,	\
						Laplace_Layer_Singular_Form
from ..misc.gmres_counter import Gmres_Counter
from ..pairing import Pairing

class LaplaceDirichletSolver(object):
	"""
	Iterative solver for Dirichlet-Laplace Problems
	Solve Types:
		'full_iterative' : no preforming, no preconditioner, everything on the fly
		'iterative' : preformed corrections, local preconditioner
		'formed' : fully form the operator
	"""
	def __init__(self, collection, solve_type='iterative', check_close=True, tolerance=1e-12):
		self.collection = collection
		self.type = solve_type
		self.check_close = check_close
		self.tolerance = 1e-12
		self.outer = [] # used for lgmres algorithm
		self.initialize()
	def initialize(self):
		self.shape = (self.collection.N, self.collection.N)
		if self.type == 'full_iterative':
			self.preconditioner = sp.sparse.linalg.LinearOperator(
				      				shape=self.shape, matvec=self.identity)
		if self.type == 'iterative':
			self.SLP_adjustments = []
			self.DLP_adjustments = []
			self.local_inverses = []
			for i in range(self.collection.n_boundaries):
				bdy = self.collection.boundaries[i]
				side = self.collection.sides[i]
				if side == 'e':
					SU = Laplace_Layer_Singular_Form(bdy, ifcharge=True)
					SN = Laplace_Layer_Form(bdy, ifcharge=True)
					self.SLP_adjustments.append(SU-SN)
				else:
					self.SLP_adjustments.append(None)
				self.DLP_adjustments.append(-0.25*bdy.curvature*bdy.weights/np.pi)
				AU = Laplace_Layer_Singular_Form(bdy, ifdipole=True)
				if side == 'e':
					AU += SU
					np.fill_diagonal(AU, AU.diagonal()+0.5)
				else:
					np.fill_diagonal(AU, AU.diagonal()-0.5)
				self.local_inverses.append(np.linalg.inv(AU))
			self.preconditioner = sp.sparse.linalg.LinearOperator(
				      		shape=self.shape, matvec=self.local_preconditioner)
		if self.check_close:
			if self.type == 'full_iterative':
				backend = 'fly'
			elif self.type == 'iterative':
				backend = 'preformed'
			else:
				backend = 'full preformed'
			self.pairings = np.empty([self.collection.n_boundaries, self.collection.n_boundaries], dtype=object)
			self.codes = np.empty([self.collection.n_boundaries, self.collection.n_boundaries], dtype=object)
			for i in range(self.collection.n_boundaries):
				bdyi = self.collection.boundaries[i]
				for j in range(self.collection.n_boundaries):
					if i != j:
						bdyj = self.collection.boundaries[j]
						pair = Pairing(bdyi, bdyj, self.collection.sides[i], self.tolerance)
						code = pair.Setup_Close_Corrector(do_DLP=True, do_SLP=self.collection.sides[i]=='e', backend=backend)
						self.pairings[i,j] = pair
						self.codes[i,j] = code
					else:
						self.pairings[i,j] = None
						self.codes[i,j] = None
		if self.type == 'formed':
			self.OP = np.empty([self.collection.N, self.collection.N], dtype=float)
			# naive evals
			for i in range(self.collection.n_boundaries):
				i1, i2 = self.collection.get_inds(i)
				iside = self.collection.sides[i]
				ibdy = self.collection.boundaries[i]
				for j in range(self.collection.n_boundaries):
					if i == j:
						self.OP[i1:i2, i1:i2] = Laplace_Layer_Singular_Form(ibdy, ifdipole=True, ifcharge=iside=='e')
					else:
						j1, j2 = self.collection.get_inds(j)
						jbdy = self.collection.boundaries[j]
						self.OP[j1:j2, i1:i2] = Laplace_Layer_Form(ibdy, jbdy, ifdipole=True, ifcharge=iside=='e')
			# close corrections
			if self.check_close:
				for i in range(self.collection.n_boundaries):
					i1, i2 = self.collection.get_inds(i)
					for j in range(self.collection.n_boundaries):
						if i != j:
							j1, j2 = self.collection.get_inds(j)
							pair = self.pairings[i,j]
							code = self.codes[i,j]
							try:
								self.OP[j1:j2, i1:i2][pair.close_points, :] += pair.close_correctors[code].preparations['correction_mat']
							except:
								pass
			# add in 0.5I terms
			for i in range(self.collection.n_boundaries):
				i1, i2 = self.collection.get_inds(i)
				sign = 1.0 if self.collection.sides[i] == 'e' else -1.0
				self.OP[i1:i2, i1:i2] += sign*0.5*np.eye(self.collection.boundaries[i].N)
			self.inverse_OP = np.linalg.inv(self.OP)
	def solve(self, bc, **kwargs):
		if self.type == 'full_iterative' or self.type == 'iterative':
			return self.solve_iterative(bc, **kwargs)
		else:
			return self.inverse_OP.dot(bc)
	def solve_iterative(self, bc, disp=False, **kwargs):
		counter = Gmres_Counter(disp)
		operator = sp.sparse.linalg.LinearOperator(
			      				shape=self.shape, matvec=self.Apply_Operator)
		out = sp.sparse.linalg.gmres(operator, bc, M=self.preconditioner,
													 callback=counter, **kwargs)
		return out[0]
	def local_preconditioner(self, tau):
		out = tau.copy()
		for i in range(self.collection.n_boundaries):
			ind1, ind2 = self.collection.get_inds(i)
			if self.local_inverses[i] is not None:
				out[ind1:ind2] = self.local_inverses[i].dot(tau[ind1:ind2])
		return out
	def identity(self, tau):
		return tau
	def Apply_Operator(self, tau):
		# first apply naive quad
		ch = tau*self.collection.SLP_vector
		u = Laplace_Layer_Apply(self.collection, charge=ch, dipstr=tau)
		# sweep through the boundaries and make corrections
		for i in range(self.collection.n_boundaries):
			bdy = self.collection.boundaries[i]
			ind1, ind2 = self.collection.get_inds(i)
			# for the DLPs (this happens on everyone)
			if self.type == 'full_iterative':
				adj = -0.25*bdy.curvature*bdy.weights/np.pi
			else:
				adj = self.DLP_adjustments[i]
			u[ind1:ind2] += adj*tau[ind1:ind2]
			# for the SLPs (only on 'e' sides)
			if self.collection.sides[i] == 'e':
				if self.type == 'full_iterative':
					su1 = Laplace_Layer_Singular_Apply(bdy, charge=tau[ind1:ind2])
					nu1 = Laplace_Layer_Apply(bdy, charge=tau[ind1:ind2])
					u[ind1:ind2] += (su1 - nu1)
				else:
					u[ind1:ind2] += self.SLP_adjustments[i].dot(tau[ind1:ind2])
				# the 0.5I part for 'e' sides
				u[ind1:ind2] += 0.5*tau[ind1:ind2]
			else:
				# the 0.5I part for 'i' sides
				u[ind1:ind2] -= 0.5*tau[ind1:ind2]
		# now do the close corrections
		if self.check_close:
			for i in range(self.collection.n_boundaries):
				i_ind1, i_ind2 = self.collection.get_inds(i)
				for j in range(self.collection.n_boundaries):
					j_ind1, j_ind2 = self.collection.get_inds(j)
					if i != j:
						self.pairings[i,j].Close_Correction(u[j_ind1:j_ind2], tau[i_ind1:i_ind2], self.codes[i,j])
		return u

def Evaluate_Tau(collection, target, tau):
	ch = tau*collection.SLP_vector
	return Laplace_Layer_Apply(collection, target, charge=ch, dipstr=tau)
