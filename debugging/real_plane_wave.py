import sys
sys.path.append('/home/owen/code/helmholtz-firedrake')

from firedrake import *
import numpy as np
from helmholtz_firedrake import utils
from helmholtz_firedrake import problems as hh
from matplotlib import pyplot as plt

k = 20.0

dim = 2

num_points = utils.h_to_num_cells(k**-2.0,dim)

mesh = UnitSquareMesh(num_points,num_points)

V = FunctionSpace(mesh,"CG",1)

u = Function(V)

x = SpatialCoordinate(mesh)

d_list = [np.cos(2.0*np.pi/3.0),np.sin(2.0*np.pi/3.0)]

d = as_vector(d_list)

u.interpolate(cos(k * inner(x,d))) 

computed_integral = assemble(u*dx)

print(computed_integral)


#d_calc = np.array(d_list)

#d_prod = d_calc.prod()

#integral_1 = (-1j/(k*d_prod))**dim
#integral_2 = (1 + np.exp(1j * k * d_calc)).prod()

#from_test_integral = integral_1 * integral_2

#print(from_test_integral)

d1 = d_list[0]

d2 = d_list[1]

hand_calculated_integral = 1.0/(k**2 * d1 * d2) * ((1-np.cos(k*d1))*(1-np.cos(k*d2)) - np.sin(k*d1)*np.sin(k*d2))

print(hand_calculated_integral)
