import firedrake as fd
import helmholtz_monte_carlo.generate_samples as gen_samples
import helmholtz_firedrake.problems as hh
import numpy as np
from helmholtz_firedrake import coefficients as coeff
from helmholtz_firedrake import utils
from cos_sin_integrals import cos_integral, sin_integral
import pytest

# I have no idea why this test doesn't work
def test_qoi_eval_integral():
    """Tests that the qoi being the integral of the solution over the domain
    is evaluated correctly."""

    np.random.seed(5)

    angle_vals = 2.0*np.pi * np.random.random_sample(10)

    errors = [[] for ii in range(len(angle_vals))]

    num_points_multiplier = [1,2]
    
    for ii_num_points in range(2):
        
        for ii_angle in range(len(angle_vals)):

            # Set up plane wave
            dim = 2

            k = 20.0

            num_points = num_points_multiplier[ii_num_points]*utils.h_to_num_cells(k**-1.5,dim)

            comm = fd.COMM_WORLD

            mesh = fd.UnitSquareMesh(num_points,num_points,comm)

            J = 1

            delta = 2.0

            lambda_mult = 1.0

            n_0 = 1.0

            num_points = 1

            stochastic_points = np.zeros((num_points,J))

            n_stoch = coeff.UniformKLLikeCoeff(mesh,J,delta,lambda_mult,n_0,stochastic_points)

            V = fd.FunctionSpace(mesh,"CG",1)

            prob = hh.StochasticHelmholtzProblem(k,V,A_stoch=None,n_stoch=n_stoch)

            prob.use_mumps()

            angle = angle_vals[ii_angle]
            
            d = [np.cos(angle),np.sin(angle)]

            prob.f_g_plane_wave(d)

            prob.solve()

            output = gen_samples.qoi_eval(prob,'integral',comm)

            true_integral = cos_integral(k,d) + 1j * sin_integral(k,d)

            error = np.abs(output-true_integral)

            errors[ii_angle].append(error)

    rate_approx = [np.log2(errors[ii][0]/errors[ii][1]) for ii in range(len(errors))]

    # Relative tolerance obtained by selecting a passing value for a
    # different random seed (seed=4)
    assert np.allclose(rate_approx,2.0,atol=1e-16,rtol=1e-2)

    
    
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

    d = [np.cos(2.0*np.pi/9.0),np.sin(2.0*np.pi/9.0)]
    
    prob.f_g_plane_wave(d)

    prob.use_mumps()

    prob.solve()

    # For the value of the solution at the origin:
    output = gen_samples.qoi_eval(prob,'origin',comm=fd.COMM_WORLD)
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
    
    mesh = fd.UnitSquareMesh(num_points,num_points,comm=fd.COMM_WORLD)

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
    prob.f_g_plane_wave(d)

    prob.use_mumps()

    prob.solve()

    # For the dummy we use for testing:
    this_dummy = 4.0
    output = gen_samples.qoi_eval(this_dummy,'testing',comm=fd.COMM_WORLD)

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

