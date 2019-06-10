from helmholtz_firedrake import problems as hh
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils as hh_utils
from helmholtz_firedrake.coefficients import SamplingError
import helmholtz_monte_carlo.point_generation as point_gen
import firedrake as fd
import numpy as np
import warnings
from copy import deepcopy
from scipy import optimize

def generate_samples(k,h_spec,J,nu,M,
                     point_generation_method,
                     delta,lambda_mult,j_scaling,
                     qois,
                     num_spatial_cores,dim=2,
                     display_progress=False,physically_realistic=False,
                     nearby_preconditioning=False,
                     nearby_preconditioning_proportion=1):
    
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
    randomly-shifted QMC methods. Combines with M to give number of
    integration points for Monte Carlo.

    M - positive int - Specifies the number of integration points for
    which to do computations - NOTE: for Monte Carlo, the number of
    integration points will be given by nu*(2**M). For Quasi-Monte
    Carlo, we will sample 2**m integration points, and then randomly
    shift these nu times as part of the estimator.

    point_generation_method string - either 'mc' or 'qmc', specifying
    Monte-Carlo point generation or Quasi-Monte-Carlo (based on an
    off-the-shelf lattice rule). Monte-Carlo generation currently
    doesn't work, and so throws an error.

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

    nearby_preconditioning - boolean - if true, nearby preconditioning
    is used in the solves. A proportion (given by nearby_preconditioning
    proportion) of the realisations have their exact LU decompositions
    computed, and then these are used as preconditioners for all the
    other problems (where the preconditioner used is determined by the
    nearest problem, in some metric, that has had a preconditioner
    computed). Note that if ensembles are used to speed up the solution
    time, some LU decompositions may be calculated more than once. But
    for the purposes of assessing the effectiveness of the algorithm (in
    terms of total # GMRES iterations), this isn't a problem.

    nearby_preconditioning_proportion - float in [0,1]. See the text for
    nearby_preconditioning above.

    Output:
    If point_generation_method is 'qmc', then samples is a list of
    length nu, where each entry of samples is a list of length num_qois,
    each entry of which is a numpy array of length 2**M, each entry of
    which is either: (i) a (complex-valued)
    float, or (ii) a numpy column vector, corresponding to a sample of
    the QoI.. n_coeffs is a list of length nu, each entry of
    which is a 2**M by J numpy array, each row of which contains the
    KL-coefficients needed to generate the particular realisation of n.

    """

    if point_generation_method is 'mc':
        raise NotImplementedError("Monte Carlo sampling currently doesn't work")
        
    num_qois = len(qois)
    
    mesh_points = hh_utils.h_to_num_cells(h_spec[0]*k**h_spec[1],
                                              dim)
    
    ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)
    
    mesh = fd.UnitSquareMesh(mesh_points,mesh_points,comm=ensemble.comm)

    comm = ensemble.ensemble_comm

    n_coeffs = []
        
    if point_generation_method is 'mc':
        # This needs updating one I've figured out a way to do seeding
        # in a parallel-appropriate way
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

        GMRES_its = []
                   
        for shift_no in range(nu):
            if display_progress:
                print('Shift number:',shift_no+1,flush=True)
            # Randomly shift the points
            prob.n_stoch.change_all_points(
                point_gen.shift(kl_mc_points,seed=shift_no))

            n_coeffs.append(deepcopy(prob.n_stoch.current_and_unsampled_points()))

            if nearby_preconditioning:
                [centres,nearest_centre] = find_nbpc_points(M,nearby_preconditioning_proportion,prob.n_stoch,J,point_generation_method,prob.n_stoch.current_and_unsampled_points())
            else:
                centres = None
                nearest_centre = None
            
            [this_samples,this_GMRES_its] = all_qoi_samples(prob,qois,ensemble.comm,display_progress,centres,nearest_centre,J,delta,lambda_mult,j_scaling,n_0)

            # For outputting samples and GMRES iterations
            samples.append(this_samples)
            GMRES_its.append(this_GMRES_its)
            

    comm = ensemble.ensemble_comm

    samples = fancy_allgather(comm,samples,'samples')

    n_coeffs = fancy_allgather(comm,n_coeffs,'coeffs')

    GMRES_its = fancy_allgather(comm,GMRES_its,'coeffs')
    
    return [k,samples,n_coeffs,GMRES_its]

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

    # Despite the fact that there will be multiple procs with rank 0,
    # I'm going to assume for now that this all works.
    gathered_tmp = comm.gather(to_gather,root=0)

    gathered = []
    
    # Whip it all into order
    if comm.rank == 0:
        gathered = to_gather
        for ii_nu in range(len(to_gather)):
            for ii_proc in range(1,comm.size):
                rec_gathered = gathered_tmp[ii_proc]

                if gather_type is 'samples':
                    for ii_qoi in range(len(to_gather[0])):
                        gathered[ii_nu][ii_qoi] = np.hstack((gathered[ii_nu][ii_qoi],rec_gathered[ii_nu][ii_qoi]))
                        
                elif gather_type is 'coeffs':
                    gathered[ii_nu] = np.vstack((gathered[ii_nu],rec_gathered[ii_nu]))
                    
                else:
                    raise NotImplementedError
            
    # Broadcast
    gathered = comm.bcast(gathered,root=0)

    return gathered
                    

def all_qoi_samples(prob,qois,comm,display_progress,centres=None,nearest_centre=None,J=None,delta=None,lambda_mult=None,j_scaling=None,n_0=None):
    """Computes all samples of the qoi for a StochasticHelmholtzProblem.

    This is a helper function for investigate_error.

    Parameters:

    prob - a StochasticHelmholtzProblem

    qois - list of some of the qois allowed in investigate_error.

    comm - the communicator for spatial parallelism.

    display_progress - boolean - if true, prints the sample number each
    time we sample.

    centres - a list of numpy arrays of length J. The points at which we
    calculate the preconditioners for nearby preconditioning. If no
    preconditioning is used, None.

    nearest_centre - a list of ints of length 'number of qmc
    calculations to do on this ensemble member' - gives the 'centre'
    nearest to the corresponding sample.

    Outputs:

    samples - list of numpy arrays containing the values of each qoi for
    each realisation. samples[ii] corresponds to qois[ii].

    GMRES_its - list of ints, giving the number of GMRES iterations for
    each sample. If no GMRES was used, None.

   """
    # TODO: Update documentation here
    nearby_preconditioning = centres is not None
    num_qois = len(qois)

    GMRES_its = []
    
    samples = [[] for ii in range(num_qois)]

    # For testing purposes
    dummy = 1.0

    if nearby_preconditioning:
        # Order points with respect to the preconditioner that is used
        new_order = np.argsort(nearest_centre)
        prob.n_stoch.change_all_points(prob.n_stoch.current_and_unsampled_points()[new_order])
        nearest_centre = nearest_centre[new_order]
        current_centre = update_centre(prob,J,delta,lambda_mult,j_scaling,n_0,centres[nearest_centre[0]])
        
    sample_no = 0

    ii_centre = 0
    while True:
        if nearby_preconditioning and (centres[ii_centre] != current_centre).any():
                current_centre = update_centre(prob,J,delta,lambda_mult,j_scaling,n_0,centres[nearest_centre[0]])
                ii_centre += 1
        sample_no += 1

        if display_progress:
            print(sample_no,flush=True)
                  
        prob.solve()

        if nearby_preconditioning:
            GMRES_its.append(prob.GMRES_its)

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

    # Need to tread carefully for vector-valued QoIs, and do things via numpy
    if type(samples[0][0]) is np.ndarray:
        samples = [np.hstack(this_samples) for this_samples in samples]
    else:
        samples = [np.array(this_samples) for this_samples in samples]

    if len(GMRES_its) == 0:
        GMRES_its = None
        
    return [samples,GMRES_its]

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

        # A bit funny because output needs to be a column vector
        #output = np.array([eval_at_mesh_point(DG_fun,point,comm) for DG_fun in DG_functions],ndmin=2).transpose()

        # For now, set the output to be the first component of the gradient
        output = eval_at_mesh_point(DG_functions[0],point,comm)

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

def weighted_L1_norm(point,array,sqrt_lambda):
    """Calculates the weighted L^1 norms between point and array, weighted by
    sqrt_lambda.

    Point is a single vector

    Array is an array of vectors stacked on top of each other

    Output - column vector containing the norms.
    """
    return np.linalg.norm((point-array)*sqrt_lambda,ord=1,axis=1)

def find_nbpc_points(M,nearby_preconditioning_proportion,kl_like,J,point_generation_method,this_ensemble_points):
    """Finds the points to use as 'centres' for nearby preconditioning, and
    calculates which 'centre' corresponds to each qmc point.
    """
    # Points are generated here, so this is presumably the place to do
    # all the 'figuring out the centres' business.  We distribute the
    # points at the centres of the 'preconditioning balls' using a
    # tensor product grid.  Suppose we knew the radius of these balls
    # (in a weighted L^1-metric) should be r. Then the spacing of the
    # points in dimension j should be
    # \[d_j = r / (J * \sqrt{lambda_j})\]
    # (where the sqrt(lambda_j) are as in
    # helmholtz_firedrake.coefficients.UniformKLLikeCoeff). In order to
    # achieve this spacing, we would need \ceil(1/d_j) points in
    # dimension j. However, we know the number of points, and we reverse
    # engineer the above argument to get the radius of the balls, and
    # the spacing in each dimension. Suppose for simplicity (and because
    # there will be various other fudges and approximations in what
    # follows) that we have 1/d_j points in each dimension. Then the
    # total number of points is $r^{-J} \prod_{j=1}^J J
    # \sqrt{\lambda_j}.$ If we specify that the total number of
    # 'centres' is N_C, then we have
    # \[r = J(\prod_{j=1}^J\sqrt{\lambda_j})^{-J}\],
    # and thence we can determine d_j, and lay down equispaced points in
    # dimension j with this spacing. We then assemble the points in all
    # of stochastic space via tensor products. We then find the actual
    # 'centres' by selecting the QMC points that are nearest to these
    # 'ideal' centres. We then associated each and every QMC point with
    # a 'centre' by selecting the closest 'centre'.

    N = 2**M
    
    num_centres = round(N*nearby_preconditioning_proportion)   

    sqrt_lambda = kl_like._sqrt_lambda
    # Check this is a row vector

    #TODO - tidy this documentation
    # We do a bit of a hack to generate the distribution of the centres
    # in the different dimensions We write a function that assumes we
    # know the radius $r$ of the balls (in the funny metric), that we
    # want, and then gives us the number of points in each dimension
    # (well, not quite, because at this point the 'numbers of points'
    # are not necessarily integers). We then optimise this function
    # (it's nonlinear and nonsmooth) to find the (a?) value of $r$ that
    # gives the correct number of centres. We then round all the decimal
    # numbers to get a number of points that (we hope) isn't too far
    # off.
    # The reason why this is the right function to optimise, I'll write in later

    def continuous_centre_nums(r):
        return np.array([np.max((1.0,ii)) for ii in (float(J)*sqrt_lambda)/(2.0*r)])

    def optim_fn(r):
        return continuous_centre_nums(r).prod()-float(num_centres)

    out = optimize.bisect(optim_fn,0.1,float(J))

    centre_nums = np.round(continuous_centre_nums(out))
    # A better way to do this would be to find the closest point on the integer lattice, but I've no idea how easy/hard that is....
    
    one_d_points = [-0.5+np.linspace(1.0/(jj+1.0),jj/(jj+1.0),int(jj)) for jj in centre_nums]

    centres_meshgrid = np.meshgrid(*one_d_points)

    proposed_centres = np.vstack([coord.flatten() for coord in centres_meshgrid]).transpose()

    # Now to actually locate the centres at QMC points
    all_qmc_points = point_gen.mc_points(
        J,N,point_generation_method,section=[0,1],seed=1)

    centres = []

    for proposed in proposed_centres:
        nearest_point = np.argmin(weighted_L1_norm(proposed,all_qmc_points,sqrt_lambda))

        centres.append(all_qmc_points[nearest_point,:])

    centres = np.vstack(centres)

    actual_num_centres = centres.shape[0]

    # Now find out, for each QMC point in this ensemble member, which
    # centre is nearest to it

    num_points_this_ensemble = this_ensemble_points.shape[0]

    nearest_centre = -np.ones(num_points_this_ensemble,dtype='int')

    for ii_point in range(num_points_this_ensemble):
        point = this_ensemble_points[ii_point,:]

        nearest_centre[ii_point] = np.argmin(weighted_L1_norm(point,centres,sqrt_lambda))

    # Check we've actually assigned a centre to each point
    assert (nearest_centre >= 0).all()

    return [centres,nearest_centre]

def update_centre(prob,J,delta,lambda_mult,j_scaling,n_0,new_centre):
    """Update the preconditioner."""
    # Modified from the function update_pc in
    # helmholtz_nearby_preconditioning.
    n_pre_instance = coeff.UniformKLLikeCoeff(prob.V.mesh(),J,delta,lambda_mult,j_scaling,
                                              n_0,np.array(new_centre,ndmin=2)) # min 2 dimensions?
    return new_centre
