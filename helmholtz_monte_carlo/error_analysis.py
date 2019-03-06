from helmholtz_firedrake import problems as hh
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils as hh_utils
from helmholtz_firedrake.coefficients import SamplingError
import helmholtz_monte_carlo.point_generation as point_gen
import firedrake as fd
import numpy as np

def investigate_error(k,h_spec,J,nu,M,
                      point_generation_method,
                      delta,lambda_mult,qois,dim=2):
    """Investigates the error in Monte-Carlo methods for Helmholtz.

    Computes an approximation to the root-mean-squared error in
    Monte-Carlo or Quasi-Monte Carlo approximations of expectations of
    quantities of interest associated with the solution of a stochastic
    Helmholtz problem, where the randomness enters through a random
    field refractive index, given by an artificial-KL expansion.

    Parameters:

    k - positive float - the wavenumber for which to do computations.

    h_spec - 2-tuple - h_spec[0] should be a positive float and
    h_spec[1] should be a float. These specify the values of the mesh
    size h for which we will run experiments.
    h = h_spec[0] * k**h_spec[1].

    J - positive int - the stochastic dimension in the artificial-KL
    expansion for which to do experiments.

    nu - positive int - the number of random shifts to use in
    randomly-shifted QMC methods. Combiones with M to give number of
    integration points for Monte Carlo.

    M - positive ints - Specifies the number of integration points for
    which to do computations - NOTE: for Monte Carlo, the number of
    integration points will be given by nu*(2**M). For Quasi-Monte
    Carlo, we will sample 2**m integration points, and then randomly
    shift these nu times as part of the estimator.

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

    qois - list of strings - the Quantities of Interest that are
    computed. Currently the only options for the elements of the string
    are:
        'integral' - the integral of the solution over the domain.
        'origin' the point value at the origin.
    There is also an option 'testing', but this is used solely for
    testing the functions.

    dim - either 2 or 3 - the spatial dimension of the Helmholtz
    Problem.

    Output:

    results - list containing 3 items: [k,list of the approximations to
    the mean of the qoi, list of estimates of the error in the
    approximations].
    """

    num_qois = len(qois)
    
    mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],
                                              dim)

    mesh = fd.UnitSquareMesh(mesh_points,mesh_points)
        
    if point_generation_method is 'mc':

        N = nu*(2**M)
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,seed=1)

    elif point_generation_method is 'qmc':
        N = 2**M
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,seed=1)

    n_0 = 1.0
                
    kl_like = coeff.UniformKLLikeCoeff(
        mesh,J,delta,lambda_mult,n_0,kl_mc_points)

    # Create the problem
    V = fd.FunctionSpace(mesh,"CG",1)
    prob = hh.StochasticHelmholtzProblem(
        k,V,A_stoch=None,n_stoch=kl_like)

    prob.f_g_plane_wave()

    prob.use_mumps()
                
    if point_generation_method is 'mc':

        samples = all_qoi_samples(prob,qois)
        
        approx = []
        
        error = []
        # Calculate the approximation
        for ii in range(num_qois):
            this_approx = samples[ii].mean()
            approx.append(this_approx)
                        
        # Calculate the error - formula taken from
        # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
        # 230, pp. 3668-3694 (2011), equation (4.4)]
        this_error = np.sqrt(((samples[ii] - approx)**2.0).sum()\
                             /(float(N)*float(N-1)))
        error.append(this_error)
                        
    elif point_generation_method == 'qmc':

        approx = []
        
        error = []
        
        for ii in range(num_qois):
            
            approximations = []
            
            for shift_no in range(nu):
                # Randomly shift the points
                prob.n_stoch.change_all_points(
                    point_gen.shift(kl_mc_points,seed=shift_no))
                
                samples = all_qoi_samples(prob,qois)
                # Compute the approximation to the mean for
                # these shifted points
                
                # For testing
                if qois == ['testing_qmc']:
                    samples = np.array([float(shift_no+1)])
                        
                approximations.append(samples.mean())

            approximations = np.array(approximations)

            # Calculate the overall approximation to the
            # mean for this shift
            this_approx = approximations.mean()

            # Calculate the error - formula taken from
            # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
            # 230, pp. 3668-3694 (2011), equation (4.6)]
            this_error = np.sqrt(((approximations-this_approx)**2).sum()\
                                 /(float(nu)*(float(nu)-1.0)))

            approx.append(this_approx)
            
            error.append(this_error)
                    
    # Save approximation and error in appropriate data frame
    # TODO

    # Save data frame to file with extra metadata (how? -
    # utility function?)
    # TODO
    
    return [k,approx,error]
                    

def all_qoi_samples(prob,qois):
    """Computes all samples of the qoi for a StochasticHelmholtzProblem.

    This is a helper function for investigate_error.

    Parameters:

    prob - a StochasticHelmholtzProblem

    qois - list of some of the qois allowed in investigate_error.

    Outputs:

    samples - numpy array containing the values of the qoi for each
    realisation. Each row corresponds to a different qoi.

    """
    samples = []
    
    for qoi in qois:
    
        this_samples = []

        if qoi is 'testing':    
            dummy = 1.0

        sample_no = 0

        while True:
            sample_no += 1
            print(sample_no)
            prob.solve()

            if qoi is 'testing':
                this_samples.append(dummy)
                dummy += 1.0

            elif qoi is 'integral':
                # This is currently a bit of a hack, because there's a bug
                # in complex firedrake.
                V = prob.u_h.function_space()
                func_real = fd.Function(V)
                func_imag = fd.Function(V)
                func_real.dat.data[:] = np.real(prob.u_h.dat.data)
                func_imag.dat.data[:] = np.imag(prob.u_h.dat.data)
                this_samples.append(fd.assemble(func_real * fd.dx) + 1j*fd.assemble(func_imag * fd.dx))

            elif qoi is 'origin':
                # This (experimentally) gives the value of the function at
                # (0,0).
                this_samples.append(prob.u_h.dat.data[0])

            try:
                prob.sample()
            # Get a SamplingError when there are no more realisations
            except SamplingError:
                break

        samples.append(this_samples)

    return np.array(samples)
