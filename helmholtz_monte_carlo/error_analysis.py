from helmholtz_firedrake import problems as hh
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils as hh_utils
from helmholtz_firedrake.coefficients import SamplingError
from helmholtz_monte_carlo.point_generation import mc_points
import firedrake as fd
import numpy as np

def investigate_error(k_range,h_spec,J_range,nu,M_range,point_generation_method,delta,lambda_mult,qoi,dim=2):
    """Investigates the error in Monte-Carlo methods for Helmholtz.

    Computes an approximation to the root-mean-squared error in
    Monte-Carlo or Quasi-Monte Carlo approximations of expectations of
    quantities of interest associated with the solution of a stochastic
    Helmholtz problem, where the randomness enters through a random
    field refractive index, given by an artificial-KL expansion.

    Parameters:

    k_range - list of positive floats - the range of k for which to do
    computations

    h_spec - 2-tuple - h_spec[0] should be a positive float and
    h_spec[1] should be a float. These specify the values of the mesh
    size h for which we will run experiments.
    h = h_spec[0] * k**h_spec[1].

    J_range - list of positive ints - the range of stochastic dimensions
    in the artificial-KL expansion for which to do experiments.

    nu - positive int - the number of random shifts to use in
    randomly-shifted QMC methods. Combiones with M to give number of
    integration points for Monte Carlo.

    M_range - list of positive ints - Specifies the range of numbers of
    integration points for which to do computations - NOTE: for Monte
    Carlo, the number of integration points will be given by
    nu*(2**M). For Quasi-Monte Carlo, we will sample 2**m integration
    points, and then randomly shift these nu times as part of the
    estimator.

    point_generation_method - either 'qmc' or 'mc'. 'qmc' means a QMC
    lattice rule is used to generate the points, whereas 'mc' means the
    points are randomly generated according to a uniform distribution on
    the unit cube.

    delta - parameter controlling the rate of decay of the magntiude of
    the coefficients in the artifical-KL expansion - see
    helmholtz_firedrake.coefficients.UniformKLLikeCoeff for more
    information.

    lambda_mult - parameter controlling the absolute magntiude of the
    coefficients in the artifical-KL expansion - see
    helmholtz_firedrake.coefficients.UniformKLLikeCoeff for more
    information.

    qoi - NEED TO SPECIFY WHAT QOIS ARE AVAILABLE

    dim - either 2 or 3 - the spatial dimension of the Helmholtz
    Problem.
    """
    
    
    for k in k_range:
        print(k)
        mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],dim)

        mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
        
        for J in J_range:           

            for M in M_range:

                if point_generation_method is 'mc':

                    N = nu*(2**M)
                    kl_mc_points = mc_points(J,N,point_generation_method,seed=1)

                elif point_generation_method is 'qmc':
                    N = 2**M
                    kl_mc_points = mc_points(J,N,point_generation_method,seed=1)

                n_0 = 1.0
                    
                kl_like = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,kl_mc_points)

                # Create the problem
                V = fd.FunctionSpace(mesh,"CG",1)
                prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=kl_like)

                prob.f_g_plane_wave()

                prob.use_mumps()
                
                if point_generation_method is 'mc':

                    samples = all_qoi_samples(prob,qoi)
                                        
                    # Calculate the approximation
                    approx = samples.mean()

                    # Calculate the error - formula taken from [Graham,
                    # Kuo, Nuyens, Scheichl, Sloan, JCP 230,
                    # pp. 3668-3694 (2011), equation (4.4)]
                    error = np.sqrt(((samples - approx)**2.0).sum()/(float(N)*float(N-1)))

                if point_generation_method == 'qmc':
                    approximations = []

                    for shift_no in range(nu):
                        # Randomly shift the points
                        prob.n_stoch.reinitialise()
                        prob.n_stoch.stochastic_points.shift()

                        samples = all_qoi_samples(prob,qoi)
                        print('qmc samples')
                        print(samples)
                        # Compute the approximation to the mean for
                        # these shifted points
                        approximations.append(samples.mean())

                    approximations = np.array(approximations)

                    # Calculate the overall approximation to the mean
                    approx = approximations.mean()

                    # Calculate the error - formula taken from [Graham,
                    # Kuo, Nuyens, Scheichl, Sloan, JCP 230,
                    # pp. 3668-3694 (2011), equation (4.6)]
                    error = np.sqrt(((approximations-approx)**2).sum()/(float(nu)*(float(nu)-1.0)))
                    
                # Save approximation and error in appropriate data frame

                # Save data frame to file with extra metadata (how? - utility function?) (And also output the results to screen?)

        print(k)
        print(approx)
        print(error)
    
    return [k,approx,error]
                    

def all_qoi_samples(prob,qoi):
    """Computes all samples of the qoi for a StochasticHelmholtzProblem.

    This is a helper function for investigate_error.

    Parameters:

    prob - a StochasticHelmholtzProblem

    qoi - one of the QOIs allowed in investigate_error.

    Outputs:

    samples - numpy array containing the values of the QOI for each realisation.
    """

    samples = []

    # For debugging
    dummy = 1.0

    sample_no = 0
    
    while True:
        sample_no += 1
        print(sample_no)
        prob.solve()
        
        #For debugging/testing
        if qoi is 'testing':
            #samples.append(FUNCTION(prob.u_h))
            samples.append(dummy)
            dummy += 1.0
            
        elif qoi is 'integral':
            samples.append(fd.assemble(prob.u_h * fd.dx))

        try:
            prob.sample()
        # Get a SamplingError when there are no more realisations
        except SamplingError:
            break
#        print('sampled!')

    return np.array(samples)
