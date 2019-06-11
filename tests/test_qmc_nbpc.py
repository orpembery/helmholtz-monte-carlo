import helmholtz_monte_carlo.generate_samples as gen_samples
import numpy as np

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

    # This is a hack to ensure that every point is selected - if you put this as 1.0, then because of how we select the preconditioning points, not every point is selected.
    nearby_preconditioning_proportion = 1.0/float(2**M)

    output = gen_samples.generate_samples(k,h_spec,J,nu,M,
                                          point_generation_method,
                                          delta,lambda_mult,j_scaling,qois,
                                          num_spatial_cores,dim,display_progress,
                                          physically_realistic,
                                          nearby_preconditioning,
                                          nearby_preconditioning_proportion)

    assert np.array(np.array(output[3]) == 1).sum() == 1
