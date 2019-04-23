import firedrake as fd
import numpy as np
import sys
from helmholtz_monte_carlo.generate_samples import fancy_allgather

if __name__ == '__main__':

    try:
        on_balena = bool(int(sys.argv[1]))
    except IndexError:
        on_balena = False

    # Just need to figure out how to actually run the thing in parallel
    # in pytest. Maybe do Firedrake recursive MPI hackery? (In conftest)

    if on_balena:
        print('loading module')
        from firedrake_complex_hacks import balena_hacks
        balena_hacks.fix_mesh_generation_time()

    overall_size = fd.COMM_WORLD.size

    for num_spatial_cores in range(1,overall_size+1):

        if overall_size % num_spatial_cores == 0:

            # This is the start of a new test #########
            #def test_n_coeff_gather():
            """Tests my gathering function works for coefficients."""

            ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)

            N = 20

            J = 10

            comm = ensemble.ensemble_comm

            to_gather = [float(comm.rank) * np.ones((N,J),dtype='float')]
            
            gathered = fancy_allgather(comm,to_gather,'coeffs')

            truth = np.zeros((N,J),dtype='float')

            for ii in range(1,comm.size):
                truth = np.vstack((truth,float(ii)*np.ones((N,J),dtype='float')))

            assert np.all(gathered == truth)



