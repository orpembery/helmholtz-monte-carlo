from helmholtz_firedrake import problems as hh
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils as hh_utils
from helmholtz_firedrake.coefficients import SamplingError
import helmholtz_monte_carlo.point_generation as point_gen
import firedrake as fd
import numpy as np
import warnings
from copy import deepcopy

def generate_samples(k,h_spec,J,nu,M,
                     point_generation_method,
                     delta,lambda_mult,j_scaling,
                     qois,
                     num_spatial_cores,dim=2,
                     display_progress=False,physically_realistic=False):
    """Generates samples for Monte-Carlo methods for Helmholtz.

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

    j_scaling - parameter controlling the oscillation in the basis
    functions in the artifical-KL expansion - see
    helmholtz_firedrake.coefficients.UniformKLLikeCoeff for more
    information.

    qois - list of strings - the Quantities of Interest that are
    computed. Currently the only options for the elements of the string
    are:
        'integral' - the integral of the solution over the domain.
        'origin' the point value at the origin.
        'top_right' the point value at (1,1)
        'gradient_top_right' the gradient at (1,1)
    There are also the options 'testing' and 'testing_qmc', but these
    are used solely for testing the functions.

    num_spatial_cores - int - the number of cores we want to use to
    solve our PDE. (You need to specify this as we might use ensemble
    parallelism to speed things up.)

    dim - either 2 or 3 - the spatial dimension of the Helmholtz
    Problem.

    display_progress - boolean - if true, prints the sample number each
    time we sample.

    physically_realistic - boolean - if true, f and g correspond to a
    scattered plane wave, n is cut off away from the truncation
    boundary, and n is >= 0.1. Otherwise, f and g are given by a plane
    wave. The 'false' option is used to verify regression tests.

    Output:
    Warning: MC needs updating for multiple QOIs results - list
    containing 3 items: [k,samples,n_coeffs], where samples is a list
    containing all of the samples of the QoI, and n_coeffs is a list
    containing all of the KL-coefficients for each realisation of n.

    If point_generation_method is 'mc', then samples is a list of length
    num_qois, each entry of which is a numpy array of length nu *
    (2**M), where each entry is either: (i) a (complex-valued) float, or (ii) a numpy column vector, corresponding
    to a sample of the QoI.

    If point_generation_method is 'qmc', then samples is a list of
    length nu, where each entry of samples is a list of length num_qois,
    each entry of which is a numpy array of length 2**M, each entry of
    which is as above. n_coeffs is a list of length nu, each entry of
    which is a 2**M by J numpy array, each row of which contains the
    KL-coefficients needed to generate the particular realisation of n.

    """

    if point_generation_method is 'mc':
        warnings.warn("Monte Carlo sampling currently doesn't work",Warning)
        
    num_qois = len(qois)
    
    mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],
                                              dim)
    
    ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)
    
    mesh = fd.UnitSquareMesh(mesh_points,mesh_points,comm=ensemble.comm)

    comm = ensemble.ensemble_comm

    n_coeffs = []
        
    if point_generation_method is 'mc':
        # This needs updating one I've figured out a way to do seeding in a parallel-appropriate way !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        N = nu*(2**M)
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,seed=1)

    elif point_generation_method is 'qmc':
        N = 2**M
        kl_mc_points = point_gen.mc_points(
            J,N,point_generation_method,section=[comm.rank,comm.size],seed=1)
        
    n_0 = 1.0
                
    kl_like = coeff.UniformKLLikeCoeff(
        mesh,J,delta,lambda_mult,j_scaling,n_0,kl_mc_points)

    # Create the problem
    V = fd.FunctionSpace(mesh,"CG",1)
    prob = hh.StochasticHelmholtzProblem(
        k,V,A_stoch=None,n_stoch=kl_like)

    angle = np.pi/4.0

    if physically_realistic:
    
        prob.f_g_scattered_plane_wave([np.cos(angle),np.sin(angle)])

        prob.sharp_cutoff(np.array((0.5,0.5)),0.75)

        prob.n_min(0.1)
    else:
        prob.f_g_plane_wave([np.cos(angle),np.sin(angle)])

    prob.use_mumps()
                
    if point_generation_method is 'mc':

        samples = all_qoi_samples(prob,qois,ensemble.comm,display_progress)
                        
    elif point_generation_method == 'qmc':

        samples = []
                   
        for shift_no in range(nu):
            if display_progress:
                print(shift_no+1,flush=True)
            # Randomly shift the points
            prob.n_stoch.change_all_points(
                point_gen.shift(kl_mc_points,seed=shift_no))

            n_coeffs.append(deepcopy(prob.n_stoch.current_and_unsampled_points()))
            
            this_samples = all_qoi_samples(prob,qois,ensemble.comm,display_progress)

            # For outputting samples
            samples.append(this_samples)            

    comm = ensemble.ensemble_comm 

    samples = fancy_allgather(comm,samples,'samples')

    n_coeffs = fancy_allgather(comm,n_coeffs,'coeffs')
    
    return [k,samples,n_coeffs]

def fancy_allgather(comm,to_gather,gather_type):
    """Effectively does an allgather, but for the kind of list we're
    using here as a datastructure.

    The gather should be over an ensemble communicator, not a spatial one.

    Inputs:

    comm - the MPI communicator over which to do the gather.

    to_gather - the list (of lists, possibly) to gather onto all
    processes.

    gather_type - a string, either 'samples' or 'coeffs'. If 'samples',
    assumes each entry of to_gather is itself a list, and each entry of
    this list is a 1-d numpy array, and we gather by concatenating all
    these arrays together. If 'coeffs', assumes each entry of to_gather
    is a 2-d numpy array, and we gather by vertically concatenating
    these.

    Outputs:

    gathered - the gathered list, should have the same format as
    to_gather, but holds the data from all of them.
    """

    # Despite the fact that there will be multiple procs with rank 0, I'm going to assume for now that this all works.
    gathered_tmp = comm.gather(to_gather,root=0)

    gathered = []
    
    #Whip it all into order
    if comm.rank == 0:
        gathered = to_gather
        for ii_to_gather in range(len(to_gather)):
            for ii in range(1,comm.size):
                rec_gathered = gathered_tmp[ii]

                if gather_type is 'samples':
                    for ii_qoi in range(len(to_gather[0])):
                        gathered[ii_to_gather][ii_qoi] = np.hstack((gathered[ii_to_gather][ii_qoi],rec_gathered[ii_to_gather][ii_qoi]))
                        
                elif gather_type is 'coeffs':
                    gathered[ii_to_gather] = np.vstack((gathered[ii_to_gather],rec_gathered[ii_to_gather]))
                    
                else:
                    raise NotImplementedError
            
    # Broadcast
    gathered = comm.bcast(gathered,root=0)

    return gathered
                    

def all_qoi_samples(prob,qois,comm,display_progress):
    """Computes all samples of the qoi for a StochasticHelmholtzProblem.

    This is a helper function for investigate_error.

    Parameters:

    prob - a StochasticHelmholtzProblem

    qois - list of some of the qois allowed in investigate_error.

    comm - the communicator for spatial parallelism.

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
        for this_qoi in sorted(set(qois)):

            # This is a little hack that helps with testing
            if this_qoi is 'testing':
                prob_input = dummy
            else:
                prob_input = prob
                
            this_qoi_findings = qoi_finder(qois,this_qoi)
            
            if this_qoi_findings[0]:
                for ii in this_qoi_findings[1]:
                    samples[ii].append(qoi_eval(prob_input,this_qoi,comm))


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

def qoi_eval(prob,this_qoi,comm):
    """Helper function that evaluates qois.

    prob - Helmholtz problem (or, for testing purposes only, a float)

    this_qoi - string, one of ['testing','integral','origin']

    comm - the communicator for spatial parallelism.

    output - the value of the qoi for this realisation of the
    problem. None if this_qoi is not in the list above.

    """
    if this_qoi == 'testing':
        output = prob

    elif this_qoi == 'integral':
        # This is currently a bit of a hack, because there's a bug
        # in complex firedrake.
        # It's also non-obvious why this works in parallel....
        V = prob.u_h.function_space()
        func_real = fd.Function(V)
        func_imag = fd.Function(V)
        func_real.dat.data[:] = np.real(prob.u_h.dat.data)
        func_imag.dat.data[:] = np.imag(prob.u_h.dat.data)
        output = fd.assemble(func_real * fd.dx) + 1j * fd.assemble(func_imag * fd.dx)
        
    elif this_qoi == 'origin':
        # This gives the value of the function at (0,0).
        output = eval_at_mesh_point(prob.u_h,np.array([0.0,0.0]),comm)

    elif this_qoi == 'top_right':
        # This gives the value of the function at (1,1).
        output = eval_at_mesh_point(prob.u_h,np.array([1.0,1.0]),comm)

    elif this_qoi == 'gradient_top_right':
        # This gives the gradient of the solution at the
        # top-right-hand corner of the domain.
        gradient = fd.grad(prob.u_h)

        DG_spaces = [fd.FunctionSpace(prob.V.mesh(),"DG",1) for ii in range(len(gradient))]

        DG_functions = [fd.Function(DG_space) for DG_space in DG_spaces]

        for ii in range(len(DG_functions)):
            DG_functions[ii].interpolate(gradient[ii])

        point = tuple([1.0 for ii in range(len(gradient))])
            
        output = np.array([[eval_at_mesh_point(DG_fun,point,comm)] for DG_fun in DG_functions],ndmin=2)

    else:
        output = None

    return output
        
def eval_at_mesh_point(v,point,comm):
    """Evaluates a Function at a point on the mesh. Only tested for
    1st-order CG and 0th- and 1st-order DG elements.

    Parameters:

    v - a Firedrake function

    point - a tuple of the correct length, giving the coordinates of the
    mesh point at which we evaluate.

    comm - the communicator for spatial parallelism.
    """
    mesh = v.function_space().mesh()
        
    location_in_list = []
    
    # In each dimension, find out which mesh points match the ith
    # coordinate of our desired point.

    data_tmp = deepcopy(v.dat.data_ro)
    
    for ii in range(len(point)):

        # Need to interpolate the coordinates into the function space,
        # to allow for DG spaces, where the node coordinates aren't in
        # a one-to-one correspondence with the DoFs.        
        v.interpolate(mesh.coordinates[ii])

        coords = v.vector().gather()

        dists = np.abs(np.real(coords - point[ii]))
    
        if ii == 0:
            distances_to_point = dists
        else:
            distances_to_point = np.vstack((distances_to_point,dists))
        
    norm_distances = np.linalg.norm(distances_to_point,axis=0,ord=2)

    min_dist = np.min(norm_distances)

    loc = np.isclose(norm_distances,min_dist)
    
    # Put the old data back
    v.dat.data[:] = data_tmp
    
    return v.vector().gather()[loc][0]
