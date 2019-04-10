from helmholtz_firedrake import problems as hh
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils as hh_utils
from helmholtz_firedrake.coefficients import SamplingError
import helmholtz_monte_carlo.point_generation as point_gen
import firedrake as fd
import numpy as np

def investigate_error(k,h_spec,J,nu,M,
                      point_generation_method,
                      delta,lambda_mult,qois,num_spatial_cores,dim=2,display_progress=False):
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
    There are also the options 'testing' and 'testing_qmc', but these
    are used solely for testing the functions.

    num_spatial_cores - int - the number of cores we want to use to
    solve our PDE. (You need to specify this as we might use ensemble
    parallelism to speed things up.)

    dim - either 2 or 3 - the spatial dimension of the Helmholtz
    Problem.

    display_progress - boolean - if true, prints the sample number each
    time we sample.

    Output:
    # MC needs updating for multiple QOIs
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    results - list containing 2 items: [k,samples], where samples is a
    list containing all of the samples of the QoI. If
    point_generation_method is 'mc', then samples is a list of length nu
    * (2**M), where each entry is a (complex-valued) float corresponding
    to a sample of the QoI. If point_generation_method is 'qmc', then
    samples is a list of length nu, where each entry of samples is a
    list of length num_qois, each entry of which is a numpy array of
    length 2**M, each entry of which is as above.
    """
    num_qois = len(qois)
    
    mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],
                                              dim)
    
    ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)
    
    mesh = fd.UnitSquareMesh(mesh_points,mesh_points,comm=ensemble.comm)
        
    if point_generation_method is 'mc':
        # This needs updating one I've figured out a way to do seeding in a parallel-appropriate way !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        N = nu*(2**M)
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,seed=1)

    elif point_generation_method is 'qmc':
        N = 2**M
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,section=[ensemble.ensemble_comm.rank,ensemble.ensemble_comm.size],seed=1)

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

        samples = all_qoi_samples(prob,qois,display_progress)
        
        # approx = []
        
        # error = []
        # # Calculate the approximation
        # for ii in range(num_qois):

        #     this_approx = samples[ii].mean()
        #     approx.append(this_approx)
                        
        #     # Calculate the error - formula taken from
        #     # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
        #     # 230, pp. 3668-3694 (2011), equation (4.4)]
        #     this_error = np.sqrt(((samples[ii] - this_approx)**2.0).sum()\
        #     /(float(N)*float(N-1)))
        #     error.append(this_error)
                        
    elif point_generation_method == 'qmc':

        samples = []
        
        # approx = []
        
        # error = []

        # all_approximations = [[] for ii in range(num_qois)]
                   
        for shift_no in range(nu):
            if display_progress:
                print(shift_no+1,flush=True)
            # Randomly shift the points
            prob.n_stoch.change_all_points(
                point_gen.shift(kl_mc_points,seed=shift_no))

            this_samples = all_qoi_samples(prob,qois,display_progress)
            # Compute the approximation to the mean for
            # these shifted points
            
            # # For testing
            # if qois == ['testing_qmc']:
            #     this_samples = [np.array(float(shift_no+1))]
            # elif qois == ['testing_qmc','testing_qmc']:
            #     this_samples = [np.array(float(shift_no+1)),np.array(float(shift_no+1))]
                
            # for ii in range(num_qois):

            #     all_approximations[ii].append(this_samples[ii].mean())

            # For outputting samples
            samples.append(this_samples)

        # all_approximations = [np.array(approximation) for approximation in all_approximations]
                
        # # Calculate the QMC approximations for each qoi
        # approx = [approximation.mean() for approximation in all_approximations]

        # # Calculate the error for each qoi - formula taken from
        # # [Graham, Kuo, Nuyens, Scheichl, Sloan, JCP
        # # 230, pp. 3668-3694 (2011), equation (4.6)]
        # error = [np.sqrt(((approx[ii]-all_approximations[ii])**2).sum()\
        #                      /(float(nu)*(float(nu)-1.0))) for ii in range(num_qois)]

    # Save data frame to file with extra metadata (how? -
    # utility function?)
    # TODO

    comm = ensemble.ensemble_comm 

    # Despite the fact that there will be multiple procs with rank 0, I'm going to assume for now that this all works.
    samples_tmp = comm.gather(samples,root=0)

    #Whip it all into order
    if comm.rank == 0:
        for ii in range(comm.size):
            rec_samples = samples_tmp[ii]
            for shift_no in range(nu):
                for qoi_no in range(num_qois):
                    samples[shift_no][qoi_no] = np.hstack((samples[shift_no][qoi_no],rec_samples[shift_no][qoi_no]))
    # Broadcast
    samples = comm.bcast(samples,root=0)
    
    return [k,samples]
                    

def all_qoi_samples(prob,qois,display_progress):
    """Computes all samples of the qoi for a StochasticHelmholtzProblem.

    This is a helper function for investigate_error.

    Parameters:

    prob - a StochasticHelmholtzProblem

    qois - list of some of the qois allowed in investigate_error.

    display_progress - boolean - if true, prints the sample number each
    time we sample.

    Outputs:

    samples - list of numpy arrays containing the values of each qoi for
    each realisation. samples[ii] corresponds to qois[ii].

    """
    num_qois = len(qois)
    
    samples = [[] for ii in range(num_qois)]

    # For testing purposes
    dummy = 1.0

    sample_no = 0
    while True:
        sample_no += 1
        if display_progress:
            print(sample_no,flush=True)
            
        prob.solve()        

        # Using 'set' below means we only tackle each qoi once.
        for this_qoi in set(qois):

            # This is a little hack that helps with testing
            if this_qoi is 'testing':
                prob_input = dummy
            else:
                prob_input = prob
                
            this_qoi_findings = qoi_finder(qois,this_qoi)
            
            if this_qoi_findings[0]:
                for ii in this_qoi_findings[1]:
                    samples[ii].append(qoi_eval(prob_input,this_qoi))       

        try:
            prob.sample()
            # Next line is only for testing
            dummy += 1.0
        # Get a SamplingError when there are no more realisations
        except SamplingError:
            prob.n_stoch.reinitialise()
            break

    samples = [np.array(this_samples) for this_samples in samples]

    return samples

def qoi_finder(qois,this_qoi):
    """Helper function that finds this_qoi in qois.

    Parameters:

    qois - list of strings

    this_qoi - a string

    Returns:

    list, the entries of which are:

    in_list - Boolean - True if this_qoi is an element of qois, False
    otherwise.

    indices- list of ints - the entries of qois that are this_qoi. Empty
    list if in_list is False.

    """
    in_list = this_qoi in qois

    indices = []
    if in_list:
        current_pos = 0
        for ii in range(qois.count(this_qoi)):
            this_index = qois.index(this_qoi,current_pos)
            indices.append(this_index)
            current_pos = this_index + 1

    return [in_list,indices]

def qoi_eval(prob,this_qoi):
    """Helper function that evaluates qois.

    prob - Helmholtz problem (or, for testing purposes only, a float)

    this_qoi - string, one of ['testing','integral','origin']

    output - the value of the qoi for this realisation of the
    problem. None if this_qoi is not in the list above.

    """
    if this_qoi is 'testing':
        output = prob

    elif this_qoi is 'integral':
        # This is currently a bit of a hack, because there's a bug
        # in complex firedrake.
        V = prob.u_h.function_space()
        func_real = fd.Function(V)
        func_imag = fd.Function(V)
        func_real.dat.data[:] = np.real(prob.u_h.dat.data)
        func_imag.dat.data[:] = np.imag(prob.u_h.dat.data)
        output = fd.assemble(func_real * fd.dx) + 1j * fd.assemble(func_imag * fd.dx)
        
    elif this_qoi is 'origin':
        # This (experimentally) gives the value of the function at
        # (0,0).
        output = prob.u_h.dat.data[0]

    else:
        output = None

    return output
        
