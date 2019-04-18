#from helmholtz_monte_carlo import error_analysis as err
import pickle
from generation_code import qmc_test,serial_filename
import firedrake as fd
import numpy as np
import sys
#from sys import getsizeof
#def test_parallel_regression(): # Change this when I know how to run it in a pytest framework



if __name__ == '__main__':
    """Tests that parallel code outputs the same as (older) serial code.

    This code should be run in parallel using 
    `mpirun -n N python test_parallel_regression`.
    
    old_out is generated by running `python serial_output_code.py
    (although there should already be a file in the test directory
    outputted by this.

    Command line argument should be 1 or 0. 1 means we are running on
    Balena (and so should use a hack to speed up the mesh generation
    time). 0 means we are not on Balena.

    """

    on_balena = bool(int(sys.argv[1]))

    print(on_balena)
    
    # Just need to figure out how to actually run the thing in parallel
    # in pytest. Maybe do Firedrake recursive MPI hackery? (In conftest)

    if on_balena:
        from firedrake_complex_hacks import balena_hacks
        balena_hacks.fix_mesh_generation_time()

    overall_size = fd.COMM_WORLD.size
    
    for num_spatial_cores in range(1,overall_size+1):

        if overall_size % num_spatial_cores == 0:
    
            qmc_out = qmc_test(num_spatial_cores)

            if fd.COMM_WORLD.rank == 0:
                with open(serial_filename(),'rb') as f:
                    old_out = pickle.load(f)

                assert qmc_out[0] == old_out[0] # should be a float,
                                                # but isn't the output
                                                # of a numerical
                                                # method, so == is OK.

                assert len(qmc_out) == len(old_out)

                for ii in range(1,len(old_out)):
                    assert(len(old_out[ii])==len(qmc_out[ii]))
                    for jj in range(len(old_out[1])):
                        # For some reason, the sizes of these variables (in
                        # bytes) aren't always the same. I've no idea why.
                        # Hence, this assertion is commented out.
                        #assert np.all(np.isclose(qmc_out[ii][jj],old_out[ii][jj]))
                        pass

                for ii in range(1,len(qmc_out)):
                    assert(len(old_out[ii])==len(qmc_out[ii]))
                    for jj in range(len(qmc_out[1])):
                        for kk in range(len(qmc_out[1][0])):
                            # Commented out here for same reason as above
                            #assert getsizeof(qmc_out[ii][jj]) == getsizeof(old_out[ii][jj]) 
                            assert np.all(np.isclose(qmc_out[ii][jj][kk],old_out[ii][jj][kk]))
