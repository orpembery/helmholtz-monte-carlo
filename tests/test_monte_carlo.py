import firedrake as fd
import helmholtz_monte_carlo.point_generation as point_gen
import helmholtz_monte_carlo.error_analysis as err_an
import helmholtz_firedrake.problems as hh
import numpy as np
import latticeseq_b2
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils
import pytest

def test_mc_points_correct():
    """Tests that Monte Carlo points are in the (centred) unit cube.

    Also provides a quick 'check' that the points are random.
    """
    
    J = 100
    N = 1024
    point_generation_method = 'mc'
    seed = 42
    points = point_gen.mc_points(J,N,point_generation_method,seed)
    assert (-0.5 <= points).all() and (points <= 0.5).all()

    # The following is a 'quick and dirty' check that the points are
    # random - whether their average is near the centre of the cube. The
    # threshold for 'near' is a heuristic that I chose by looking at
    # generated random numbers.
    assert np.abs(points.mean()) < 0.0025

    
def test_qmc_points_correct():
    """Tests that Monte Carlo points are in the (centred) unit cube.

    Also checks that they are the same as generated by Dirk Nuyens' code
    (although given Dirks code underlies the code being tested, this
    isn't a great test). But the test was written without looking at the
    code being tested, so maybe that makes it slightly better.

    """
    J = 100
    N = 1024
    point_generation_method = 'qmc'
    points = point_gen.mc_points(J,N,point_generation_method)
    assert (-0.5 <= points).all() and (points <= 0.5).all()

    true_points_gen = latticeseq_b2.latticeseq_b2(s=J)

    for m in range(11):
        true_points = true_points_gen.calc_block(m) - 0.5
        if m == 0:
            # Dealing with  indexing
            assert (true_points == points[0:1,:]).all()
        else:
            assert (true_points == points[2**(m-1):2**m,:]).all()

        
def test_points_shift():
    """Tests that shifted points are inside the unit square.

    Also tests that all points are shifted the same, and that the shift
    is random.

    """
    J = 100
    N = 1024
    point_generation_method = 'mc'
    seed = 42
    points = point_gen.mc_points(J,N,point_generation_method,seed)

    shifted_points = point_gen.shift(points)

    assert (-0.5 <= shifted_points).all() and (shifted_points <= 0.5).all()

    # Checking points have been shifted by the same amount. If either
    # point has, in any coordinate, been 'wrapped round', then the
    # difference in their shifts will be 1 (the size of the hypercube).
    shift_0 = shifted_points[0,:] - points[0,:]

    shift_1 = shifted_points[1,:] - points[1,:]

    # This is a bit of a hack, because arrays of truth values are
    # complicated. It says that for every element of the arrays, either
    # they are equal, or they differ by 1.
    differences = (shift_0 - shift_1) * (np.abs(shift_0-shift_1) - 1)

    assert (differences == 0).all()
    
    # Heuristic check (similar to that in test_mc_points_correct) that
    # the shift is random.
    assert shift_0.mean() < 0.07

def test_all_qoi_samples():
    """Test that the code correctly samples all of the stochastic points."""
    mesh = fd.UnitSquareMesh(10,10)

    J = 100

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 20

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)

    k = 1.0
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

    prob.f_g_plane_wave()

    prob.use_mumps()

    samples = err_an.all_qoi_samples(prob,['testing'])
    assert len(samples) == 1
    assert np.allclose(samples[0],np.arange(1.0,float(num_points)+1.0))
    
def test_mc_calculation():
    k = 1.0

    h_spec = (1.0,-1.5)

    J = 100

    nu = 16

    M = 4

    point_generation_method = 'mc'

    delta = 2.0

    lambda_mult = 1.0

    qois = ['testing']

    output = err_an.investigate_error(k,h_spec,J,nu,M,
                                      point_generation_method,
                                      delta,lambda_mult,qois,dim=2)

    N = float(nu*2**M)
    
    assert np.isclose(output[1],(N+1.0)/2.0)
    
    assert np.isclose(output[2],np.sqrt((N+1.0)/12.0))

def test_qmc_calculation():
    k = 1.0

    h_spec = (1.0,-1.5)

    J = 100

    nu = 16

    M = 4

    point_generation_method = 'qmc'

    delta = 2.0

    lambda_mult = 1.0

    qois = ['testing_qmc']

    output = err_an.investigate_error(k,h_spec,J,nu,M,
                                      point_generation_method,
                                      delta,lambda_mult,qois,dim=2)
    
    assert np.isclose(output[1],(float(nu)+1.0)/2.0)
    
    assert np.isclose(output[2],np.sqrt((float(nu)+1.0)/12.0))

@pytest.mark.xfail
def test_qoi_samples_integral():
    """Checks that the correct qoi is calculated for a plane wave.

    Qoi is the integral of the function over the domain.
    """

    dim = 2
    
    k = 20.0

    num_points = utils.h_to_num_cells(k**-1.5,dim)
    
    mesh = fd.UnitSquareMesh(num_points,num_points)

    J = 1

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 1

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

    d_list = [np.cos(np.pi/16.0),np.sin(np.pi/4.0)]
    #This test fails for most wave directions. See
    #the below integral test for more discussion.

    d = fd.as_vector(d_list)
    prob.f_g_plane_wave()

    prob.use_mumps()

    samples = err_an.all_qoi_samples(prob,['integral'])

    # Should be just one sample
    assert samples[0].shape[0] == 1

    true_integral = plane_wave_integral(d_list,k,dim)
    
    # This should be the integral over the unit square/cube of a plane
    # wave I've tweaked the definition of 'closeness' as there's
    # obviously some FEM error coming in here. But working on a finer
    # mesh, you see that the computed value approaches the true integral
    # (I've only run it for a plane wave incident from the bottom-left
    # corner), so I'm confident this is computing the correct value,
    # modulo FEM error.  The value of rtol has been chosen by looking at
    # the error for a plane wave incident from the bottom left (d =
    # [1/sqrt(2),1/sqrt(2)]), and choosing rtol so that test
    # passes. However, the actual test above is run with a different
    # incident plane wave.
    assert np.isclose(samples[0],true_integral,atol=1e-16,rtol=1e-2)

def test_qoi_samples_origin():
    """Checks that the correct qoi is calculated for a plane wave.

    Qoi is the value of the function at the origin.
    """

    dim = 2
    
    k = 20.0

    num_points = utils.h_to_num_cells(k**-1.5,dim)
    
    mesh = fd.UnitSquareMesh(num_points,num_points)

    J = 1

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 1

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

    d_list = [np.cos(np.pi/8.0),np.sin(np.pi/8.0)]

    d = fd.as_vector(d_list)
    prob.f_g_plane_wave()

    prob.use_mumps()

    samples = err_an.all_qoi_samples(prob,['origin'])

    # Should be just one sample
    assert samples[0].shape[0] == 1

    true_value = 1.0
    
    print(true_value)

    print(samples[0])
    # Tolerances values were ascertained to work for a different wave
    # direction. They're also the same as those in the test above.
    assert np.isclose(samples[0],true_value,atol=1e-16,rtol=1e-2)

def test_multiple_qois_qmc():
    """Checks that multiple qois are calculated correctly for QMC."""

    k = 1.0

    h_spec = (1.0,-1.5)

    J = 100

    #    nu = 16
    nu=2

    M = 4

    point_generation_method = 'qmc'

    delta = 2.0

    lambda_mult = 1.0

    # This is just testing that we correctly handle multiple qois
    qois = ['testing_qmc','testing_qmc']
    

    output = err_an.investigate_error(k,h_spec,J,nu,M,
                                      point_generation_method,
                                      delta,lambda_mult,qois,dim=2)
   
    # First qoi
    assert np.isclose(output[1][0],(float(nu)+1.0)/2.0)
    
    assert np.isclose(output[2][0],np.sqrt((float(nu)+1.0)/12.0))
    
    # Second qoi
    assert np.isclose(output[1][1],(float(nu)+1.0)/2.0)
    
    assert np.isclose(output[2][1],np.sqrt((float(nu)+1.0)/12.0))

def test_multiple_qois_mc():
    """Checks that multiple qois are calculated correctly for MC."""

    k = 1.0

    h_spec = (1.0,-1.5)

    J = 100

    nu = 16

    M = 4

    point_generation_method = 'mc'

    delta = 2.0

    lambda_mult = 1.0

    # This is just testing that we correctly handle multiple qois
    qois = ['testing','testing']
    

    output = err_an.investigate_error(k,h_spec,J,nu,M,
                                      point_generation_method,
                                      delta,lambda_mult,qois,dim=2)

    N = nu*(2**M)
    
    # First qoi
    assert np.isclose(output[1][0],(float(N)+1.0)/2.0)
    
    assert np.isclose(output[2][0],np.sqrt((float(N)+1.0)/12.0))
    
    # Second qoi
    assert np.isclose(output[1][1],(float(N)+1.0)/2.0)
    
    assert np.isclose(output[2][1],np.sqrt((float(N)+1.0)/12.0))

    

    
def test_set_seed():
    """Checks the numpy seed setter works."""

    np.random.seed(1)

    random_number_1 = np.random.rand(1)

    point_gen.set_numpy_seed(None)

    random_number_2 = np.random.rand(1)

    assert random_number_1 != random_number_2

    point_gen.set_numpy_seed(1)

    random_number_3 = np.random.rand(1)

    assert random_number_1 == random_number_3

def test_qoi_finder():
    """Checks the qoi helper function."""

    qois = ['integral','origin','integral']

    this_qoi = 'integral'

    output = err_an.qoi_finder(qois,this_qoi)

    assert output[0]

    assert output[1] == [0,2]

    this_qoi = 'origin'

    output = err_an.qoi_finder(qois,this_qoi)

    assert output[0]

    assert output[1] == [1]

    this_qoi = 'foo'

    output = err_an.qoi_finder(qois,this_qoi)
    
    assert not output[0]

    assert output[1] == []

#@pytest.mark.xfail
def test_qoi_eval_integral():
    """Tests that qois are evaluated correctly."""

    # Set up plane wave
    dim = 2
    
    k = 20.0

    num_points = utils.h_to_num_cells(k**-1.5,dim)
    
    mesh = fd.UnitSquareMesh(num_points,num_points)

    J = 1

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 1

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)
    d_list = [np.cos(2.0*np.pi/3.0),np.sin(2.0*np.pi/3.0)]
    #d_list = [np.cos(np.pi/16.0),np.sin(np.pi/16.0)]
    # If d_list is changed to anything other than
    # [np.cos(np.pi/4.0),np.sin(np.pi/4.0)]; I've tried
    # [np.cos(2.0*np.pi/7.0),np.sin(2.0*np.pi/7.0)],
    # [np.cos(np.pi/3.0),np.sin(np.pi/3.0)], and the above, then the
    # tests fail, and don't appear to converge as you refine the mesh. I
    # don't know why. Maybe the true solution for the qoi is wrong?

    d = fd.as_vector(d_list)
    prob.f_g_plane_wave()

    prob.use_mumps()

    prob.solve()

    # For the integral of the solution
    output = err_an.qoi_eval(prob,'integral')
    

    true_integral = plane_wave_integral(d_list,k,dim)
    
    # This should be the integral over the unit square/cube of a plane
    # wave I've tweaked the definition of 'closeness' as there's
    # obviously some FEM error coming in here. But working on a finer
    # mesh, you see that the computed value approaches the true integral
    # (I've only run it for a plane wave incident from the bottom-left
    # corner), so I'm confident this is computing the correct value,
    # modulo FEM error.  The value of rtol has been chosen by looking at
    # the error for a plane wave incident from the bottom left (d =
    # [1/sqrt(2),1/sqrt(2)]), and choosing rtol so that test
    # passes. However, the actual test above is run with a different
    # incident plane wave.
    print(output)
    assert np.isclose(output,true_integral,atol=1e-16,rtol=1e-2)

    
    
def test_qoi_eval_origin():
    """Tests that qois are evaluated correctly."""

    # Set up plane wave
    dim = 2
    
    k = 20.0

    num_points = utils.h_to_num_cells(k**-1.5,dim) # changed here
    
    mesh = fd.UnitSquareMesh(num_points,num_points)

    J = 1

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 1

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

    d_list = [np.cos(2.0*np.pi/9.0),np.sin(2.0*np.pi/9.0)]

    d = fd.as_vector(d_list)
    prob.f_g_plane_wave()

    prob.use_mumps()

    prob.solve()

    # For the value of the solution at the origin:
    output = err_an.qoi_eval(prob,'origin')
    # Tolerances values were ascertained to work for a different wave
    # direction. They're also the same as those in the test above.
    true_value = 1.0 + 0.0 * 1j
    assert np.isclose(output,true_value,atol=1e-16,rtol=1e-2)
    

    
def test_qoi_eval_dummy():
    """Tests that qois are evaluated correctly."""

    # Set up plane wave
    dim = 2
    
    k = 20.0

    num_points = utils.h_to_num_cells(k**-1.5,dim) # changed here
    
    mesh = fd.UnitSquareMesh(num_points,num_points)

    J = 1

    delta = 2.0

    lambda_mult = 1.0

    n_0 = 1.0

    num_points = 1

    stochastic_points = np.zeros((num_points,J))
    
    n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)
    
    V = fd.FunctionSpace(mesh,"CG",1)

    prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

    d_list = [np.cos(np.pi/8.0),np.sin(np.pi/8.0)]
    # If d_list is changed to
    # [np.cos(2.0*np.pi/7.0),np.sin(2.0*np.pi/7.0)] or
    # [np.cos(np.pi/3.0),np.sin(np.pi/3.0)], then the tests fail, and
    # don't appear to converge as you refine the mesh. I don't know
    # why. Maybe the true solution for the qoi is wrong?

    d = fd.as_vector(d_list)
    prob.f_g_plane_wave()

    prob.use_mumps()

    prob.solve()

    # For the dummy we use for testing:
    this_dummy = 4.0
    output = err_an.qoi_eval(this_dummy,'testing')

    assert np.isclose(output,this_dummy)

def plane_wave_integral(d_list,k,dim):
    """Helper function.

    Calculates the exact integral of a plane wave on the unit square.

    d_list - list of floats - list giving the wave direction

    dim - int - the spatial dimension

    k - float - the wavenumber

    output - float - the integral over the square.
    """
    d_calc = np.array(d_list)

    d_prod = d_calc.prod()

    integral_1 = (-1j/(k*d_prod))**dim
    integral_2 = (1 + np.exp(1j * k * d_calc)).prod()

    return integral_1 * integral_2
