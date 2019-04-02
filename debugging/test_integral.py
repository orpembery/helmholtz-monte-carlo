from helmholtz_monte_carlo.error_analysis import qoi_eval
from helmholtz_firedrake.problems import HelmholtzProblem
import firedrake as fd
from helmholtz_firedrake import utils
import numpy as np
from matplotlib import pyplot as plt
from complex_integral import complex_integral


h = 0.5

num_refine = 7
h_list = [0.5**float(hi) for hi in list(range(num_refine+1))]

dim = 2

k = 200.0

d_angle = np.pi/3.0

d = [np.cos(d_angle),np.sin(d_angle)]

d_vec = fd.as_vector(d)

# For just a real sine:
# k = 1,5,10 gets O(h^2) convergence
# k = 20,30,40,200(?) gets it eventually

# For a real Cosine plus imaginary sine:
# k = 1,5,10 gets O(h^2) convergence
# k = 20,30,40,100 gets it eventually

# For an actual plane wave, get basically the same behaviour as with sine/cosine

#truth = 0.5 + 0.5*1j # x + yi

#truth = (1.0 - np.cos(k))/k # real sine

#truth = (1.0 - np.cos(k))/k + (np.sin(k)/k)*1j # real sine + imaginary cosine

truth  =  (1.0/(k**2 * d[0] * d[1])) * ( (np.sin(k*d[0])*np.sin(k*d[1]) - (1.0-np.cos(k*d[0]))*(1.0-np.cos(k*d[1]))) + 1j * ((1.0-np.cos(k*d[0]))*np.sin(k*d[1]) + np.sin(k*d[0])*(1-np.cos(k*d[1]))))# Plane wave

print(truth)

integrals = []

for h in h_list:
    print(h,flush=True)
    num_cells = utils.h_to_num_cells(h,dim)

    mesh = fd.UnitSquareMesh(num_cells,num_cells)

    x = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh,"CG",1)

    func = fd.Function(V)

    #    func.interpolate(x[0]+x[1]*1j) # works fine

    #     func.interpolate(fd.sin(k * x[0])) # gets O(h^2) convergence

    #    func.interpolate(fd.sin(k * x[0]) + fd.cos(k*x[1])*1j) # also fine

    func.interpolate(fd.exp(1j * k * fd.dot(d_vec,x))) # Seems to be fine

    integrals.append(complex_integral(func))


print(integrals)

plt.loglog(h_list,np.abs(np.array(integrals)-truth))

plt.show()

