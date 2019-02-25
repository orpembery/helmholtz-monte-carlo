import numpy as np
import latticeseq_b2

class mc_points(object):
    """Holds a collection of integration points in many dimensions.

    Attributes:

    points - a numpy array containing the coordinates of the integration
    points (see documentation for init).

    Methods:

    shift - shifts the points, 'wrapping them round' the unit cube if
    necessary.
    """

    

    def __init__(self,J,M,point_generation_method,seed=1):
        """Generates either Monte-Carlo or Quasi-Monte-Carlo integration
        points on the multi-dimensional [-1/2,1/2] cube.

        Parameters:

        J - positive int - the length of the KL-like expansion in the
        definition of n.

        M - positive int - 2**M is the number of integration points to use.

        point_generation_method - either 'qmc' or 'mc'. 'qmc' means a QMC
        lattice rule is used to generate the points, whereas 'mc' means the
        points are randomly generated according to a uniform distribution on
        the unit cube.

        seed - seed with which to start the randomness.

        Outputs:

        points - 2**M x J numpy array, where each row is the coordinates
        of an integration point.

        """

        self._J = J

        self._M = M

        self._point_generation_method = point_generation_method

        if point_generation_method is 'qmc':
            # Generate QMC points on [-1/2,1/2]^J using Dirk Nuyens' code
            qmc_generator = latticeseq_b2.latticeseq_b2(s=J)

            qmc_points = []

            # The following range will have M as its last term
            for m in range((M+1)):
                qmc_points.append(qmc_generator.calc_block(m))

            self.points = qmc_points[0]

            for ii in range(1,len(qmc_points)):
                self.points = np.vstack((self.points,qmc_points[ii]))

        elif point_generation_method is 'mc':
            self.points = np.random.rand(2**M,J)

        self.points -= 0.5

    def shift(self):
        """Applies a random shift to points, 'wrapping them around' the
        unit cube if needed."""

        shift = np.random.rand(1,self._J)        
        
        # Apply shift on the [0,1] cube
        self.points += 0.5 + shift

        # Do wrapping
        self.points = np.divmod(self.points,1.0)[1]

        # Shift back to [-1,2/1,2] cube
        self.points -= 0.5
