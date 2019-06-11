import helmholtz_monte_carlo.generate_samples as gen_samples
import numpy as np
from generation_code import qmc_test, serial_filename
import pickle

def test_nbpc_qmc():
    """Checks that QMC + nearby preconditioning doesn't crash."""

    k = 1.0

    h_spec = (1.0,-1.5)

    J = 2

    nu=1

    M = 6

    point_generation_method = 'qmc'

    delta = 2.0

    lambda_mult = 1.0

    j_scaling = 1.0
    
    num_spatial_cores = 1
    
    # This is just testing that we correctly handle multiple qois
    qois = ['integral']

    dim = 2

    display_progress = True

    physically_realistic = False

    nearby_preconditioning = True

    nearby_preconditioning_proportion = 0.1
    

    output = gen_samples.generate_samples(k,h_spec,J,nu,M,
                                          point_generation_method,
                                          delta,lambda_mult,j_scaling,qois,
                                          num_spatial_cores,dim,display_progress,
                                          physically_realistic,
                                          nearby_preconditioning,
                                          nearby_preconditioning_proportion)

def test_every_point_a_preconditioner():
    """Tests for expected behaviour when every point is a preconditioner."""
    
    k = 10.0

    h_spec = (1.0,-1.5)

    J = 4

    nu = 1

    M = 5

    point_generation_method = 'qmc'

    delta = 2.0

    lambda_mult = 1.0

    j_scaling = 1.0
    
    num_spatial_cores = 1
    
    # This is just testing that we correctly handle multiple qois
    qois = ['integral']

    dim = 2

    display_progress = True

    physically_realistic = False

    nearby_preconditioning = True

    # This is a hack to ensure that every point is selected - if you put this as 1.0, then because of how we select the preconditioning points, not every point is selected.
    nearby_preconditioning_proportion = 3.0

    output = gen_samples.generate_samples(k,h_spec,J,nu,M,
                                          point_generation_method,
                                          delta,lambda_mult,j_scaling,qois,
                                          num_spatial_cores,dim,display_progress,
                                          physically_realistic,
                                          nearby_preconditioning,
                                          nearby_preconditioning_proportion)

    assert (np.array(output[3]) == 1).all()

def test_one_point_a_preconditioner():
    """Tests for expected behaviour when every point is a preconditioner."""
    
    k = 10.0

    h_spec = (1.0,-1.5)

    J = 4

    nu = 1

    M = 5

    point_generation_method = 'qmc'

    delta = 2.0

    lambda_mult = 1.0

    j_scaling = 1.0
    
    num_spatial_cores = 1
    
    # This is just testing that we correctly handle multiple qois
    qois = ['integral']

    dim = 2

    display_progress = True

    physically_realistic = False

    nearby_preconditioning = True

    nearby_preconditioning_proportion = 1.0/float(2**M)

    output = gen_samples.generate_samples(k,h_spec,J,nu,M,
                                          point_generation_method,
                                          delta,lambda_mult,j_scaling,qois,
                                          num_spatial_cores,dim,display_progress,
                                          physically_realistic,
                                          nearby_preconditioning,
                                          nearby_preconditioning_proportion)

    assert np.array(np.array(output[3]) == 1).sum() == 1

def test_same_as_no_nbpc():
    """Tests that we get the same results as not using nearby preconditioning."""

    # Large portions of this code copied from test_parallel_regression

    qmc_out = qmc_test(1,True,0.1)
    
    with open(serial_filename(),'rb') as f:
                    old_out = pickle.load(f)

    # The format of the outputs is:
    # 0 - k
    # 1 - samples - list of length nu:
    #         of lists of length num_qois
    #             of numpy arrays of length 2**M
    #                 each entry of which is a float
    # 2 - n_coeffs - list of length nu
    #         each entry of which is a (2**M) x J numpy array
    #             each row of which gives the KL-coeffs
    # 3 - GMRES_its

                    
    qmc_out[0] == old_out[0]

    assert qmc_out[0] == old_out[0] # should be a float,
                                                # but isn't the output
                                                # of a numerical
                                                # method, so == is OK.

    assert len(qmc_out) == len(old_out)

    for ii in range(1,len(old_out)):
        assert(len(old_out[ii])==len(qmc_out[ii]))

    for ii in range(1,len(qmc_out)):
        assert(len(old_out[ii])==len(qmc_out[ii]))
        
    for ii_nu in range(len(qmc_out[2])):

        for ii_kl in range(qmc_out[2][0].shape[0]):

            current_coeff = qmc_out[2][0][ii_kl,:]
            found_match = False
            
            for ii_kl_old in range(old_out[2][0].shape[0]):

                if (old_out[2][0][ii_kl_old,:] == current_coeff).all():

                    found_match = True
                    
                    assert np.isclose(old_out[1][ii_nu][0][ii_kl_old],qmc_out[1][ii_nu][0][ii_kl])
                    
            assert found_match
