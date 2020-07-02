# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:13:12 2020

@author: Paolo
"""

import numpy as np
import scipy.optimize


# Anaytic Center Finding

def analytic_center_fun(x, A, b):
	q = np.dot(A,x)-b + 1e-8
	val = np.sum(np.log(q)) 
	grad = np.sum((1./q).reshape((-1,1)) * A, axis = 0)
	hess = np.sum( (-1./(q*q)).reshape((-1,1,1))  *  A.reshape((-1,-1,1))   *    A.reshape((-1,1,-1)), axis = 0)
	return val, grad, hess

def analytic_center_hess(A, b):
	def hess_fun(x):
		q = b - np.dot(A,x) + 1e-8
		hess = np.sum( (1./(q*q)).reshape((-1,1,1))  *  np.expand_dims(A, axis=2)   *   np.expand_dims(A, axis=1) , axis = 0)
		return hess
	return hess_fun

def analytic_center_grad(A, b):
	def grad_fun(x):
		q = b - np.dot(A,x) + 1e-8
		grad = np.sum((1./q).reshape((-1,1)) * A, axis = 0)
		return grad
	return grad_fun

def analytic_center_val(A, b):
	def val_fun(x):
		q = b - np.dot(A,x) + 1e-8
		val = - np.sum(np.log(q))
		return val
	return val_fun

def get_feasible(A,b):
	#returns a feasible point for Ax < b
	# need to make the inequalities a little tighter so that the logarithms aren't freaking out
	res = scipy.optimize.linprog(-np.random.random(A.shape[1]),A_ub = A, b_ub = b - 1e-5, method='interior-point')
	if res.success == True:
		return res.x
	else:
		print("failure to find feasible point ", res)
		return "FAIL"

def analytic_center(A,b):
	print("Calulating Center")
	x0 = get_feasible(A,b)
	print(x0)
	#xc = scipy.optimiatzew
	#xc, fopt, fcalls, gcalls, hcalls, warn 
	# Had a problem where the hessian is too big at the boundary. Let it stop there
	# The stopping condition is based on delta x rather than grad?
	# Could maybe do some burn in with a couple gradient steps.
	res = scipy.optimize.fmin_ncg(f = analytic_center_val(A,b), x0 = x0, fprime = analytic_center_grad(A,b), fhess = analytic_center_hess(A,b) )
	xc = res
	print(res)
	print(xc)
	return xc

def getAtilde(A):
    last_column = A[:,-1].reshape(-1,1)
    A_tilde = A[:,0:-1]-np.tile(last_column, (1,A.shape[1]-1))
    return A_tilde
