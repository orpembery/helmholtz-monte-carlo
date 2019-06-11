import numpy as np
import latticeseq_b2
from warnings import warn
from copy import deepcopy

def mc_points(J,N,point_generation_method,section,seed=None,
              order_lexicographically=False):
        """Generates either Monte-Carlo or Quasi-Monte-Carlo integration
        points on the multi-dimensional [-1/2,1/2] cube.

        Parameters:

        J - positive int - the length of the KL-like expansion in the
        definition of n.

        N - positive int - N (should be of the form 2**M for qmc
        methods) is the number of integration points to use.

        point_generation_method - either 'qmc' or 'mc'. 'qmc' means a
        QMC lattice rule is used to generate the points, whereas 'mc'
        means the points are randomly generated according to a uniform
        distribution on the unit cube.

        section - list of length 2. The first argument says what
        'section' of the QMC points to take (using zero-based indexing),
        the second gives the total number of 'sections'. For example, if
        section = [0,5], then the function will only output the first
        1/5th of the QMC points. Used in ensemble parallelism. CURRENTLY
        ONLY IMPLEMENTED FOR QMC POINTS.

        seed - seed with which to start the randomness for Monte-Carlo
        points. If seed is None, then no seed is set, and the underlying
        random number generator is used.

        order_lexicographically - qmc points are ordered
        lexicographically, rather than as they are outputted from the
        underlying generation code.

        Outputs:

        points - N x J numpy array, where each row is the coordinates
        of an integration point.

        """
        if point_generation_method is 'qmc':

            M = int(np.log2(N))

            if 2**M != N:
                warn("N is not a power of 2, instead using N = " + str(2**M))
                
            # Generate QMC points on [-1/2,1/2]^J using Dirk Nuyens'
            # code
            qmc_generator = latticeseq_b2.latticeseq_b2(s=J)

            qmc_points = []

            # The following range will have M as its last term
            for m in range((M+1)):
                qmc_points.append(qmc_generator.calc_block(m))

            points = qmc_points[0]

            for ii in range(1,len(qmc_points)):
                points = np.vstack((points,qmc_points[ii]))

        elif point_generation_method is 'mc':
            set_numpy_seed(seed)
            points = np.random.rand(N,J)

        if order_lexicographically:
                points = points[np.lexsort(points.transpose())]
            
        points -= 0.5

        # pps = points per section
        pps = len(points)//section[1]
        
        # Last section may have 'leftover' points
        if section[0] == (section[1]-1):
            points = points[section[0]*pps:,:]
        
        else:
            points = points[(section[0]*pps):((section[0]+1)*pps),:]
            
        return points

def shift(points,seed=None):
    """Applies a random shift to points, 'wrapping them around' the
    unit cube if needed."""

    points = deepcopy(points)
    
    J = points.shape[1]

    set_numpy_seed(seed)
    
    shift = np.random.rand(1,J)        

    # Apply shift on the [0,1] cube
    points += 0.5 + shift

    # Do wrapping
    points = np.divmod(points,1.0)[1]

    # Shift back to [-1,2/1,2] cube
    points -= 0.5

    return points

def set_numpy_seed(seed):
    """Sets numpy random seed, if seed is not None."""
    if seed is not None:
        np.random.seed(seed)
