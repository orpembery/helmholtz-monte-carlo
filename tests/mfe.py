#import firedrake as fd
#from helmholtz_monte_carlo.error_analysis import all_qoi_samples
#from helmholtz_firedrake.problems import StochasticHelmholtzProblem
#from helmholtz_firedrake.coefficients import UniformKLLikeCoeff
from helmholtz_monte_carlo import error_analysis as err
#import numpy as np
#from generation_code import qmc_test

num_spatial_cores = 2


#qmc_test(num_spatial_cores)

# Below is the contents of qmc_test

k = 5.0

h_spec = (1.0,-1.5)

J = 10

nu = 1#20

M = 1

delta = 1.0
    
lambda_mult = 1.0

qois = ['integral','origin']

err.investigate_error(k,h_spec,J,nu,M,'qmc',delta,lambda_mult,qois,num_spatial_cores=num_spatial_cores,dim=2,display_progress=False)


# ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)

# lots = 100

# points = 3

# mesh = fd.UnitSquareMesh(points,points,comm)

# V = fd.FunctionSpace(mesh,"CG",1)

# J = 2

# delta = 1.0

# lambda_mult = 1.0

# n_0 = 1.0

# num_kl = 10

# kl_mc_points = np.random.rand(num_kl,J)

# n_stoch = UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,kl_mc_points)

# prob = StochasticHelmholtzProblem(5.0,V,n_stoch=n_stoch)

# prob.use_mumps()

# prob.solve()

# for ii in range(lots):  
    
#     all_qoi_samples(prob,['integral','origin'],comm,False)

    
