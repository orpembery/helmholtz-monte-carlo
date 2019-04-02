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

print(h_list[-1],flush=True)

dim = 2

k = 10.0

d_angle = np.pi/3.0

d = [np.cos(d_angle),np.sin(d_angle)]

d_vec = fd.as_vector(d)

true_integral  =  (1.0/(k**2 * d[0] * d[1])) * ( (np.sin(k*d[0])*np.sin(k*d[1]) - (1.0-np.cos(k*d[0]))*(1.0-np.cos(k*d[1]))) + 1j * ((1.0-np.cos(k*d[0]))*np.sin(k*d[1]) + np.sin(k*d[0])*(1-np.cos(k*d[1]))))# Plane wave

# Is this right? - think so, interpolants converge

integrals = []

err_L2 = []

err_H1 = []

interp_integrals = []

for h in h_list:
    print(h,flush=True)
    num_cells = utils.h_to_num_cells(h,dim)

    mesh = fd.UnitSquareMesh(num_cells,num_cells)

    x = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh,"CG",1)

    prob = HelmholtzProblem(k,V)

    prob.f_g_plane_wave(d)

    prob.use_mumps() # Note to self, check if get different results with lu

    prob.solve()  

    prob.plot()
    
    integrals.append(complex_integral(prob.u_h))

    # Check FEM solution is converging

    exact_soln = fd.exp(1j * k * fd.dot(x,d_vec))

    err_L2.append(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="L2"))
    err_H1.append(fd.norms.errornorm(exact_soln,prob.u_h,norm_type="H1"))

    # Check interpolation errors converge correctly

    func_interp = fd.Function(V)

    func_interp.interpolate(exact_soln)

    interp_integrals.append(complex_integral(func_interp))

# Check how integrals of FEM solutions are converging
    
print(integrals)

plt.loglog(h_list,np.abs(np.array(integrals)-true_integral))

plt.title('Errors for k = ' + str(k))

plt.show()

# Errors against a computed value - CURRENTLY INCORRECT

#alternative_truth_func = fd.Function(V) # On finest space

#alternative_truth_func.interpolate(fd.exp(1j * k * fd.dot(d_vec,x)))

#alternative_truth = fd.assemble(alternative_truth_func*fd.dx) # This is only calculating the real part!

#plt.loglog(h_list,np.abs(np.array(integrals)-alternative_truth))

#plt.title('Errors for k = ' + str(k) + ' against computed integral')

#plt.show()



# Checking that FEM is converging correctly

plt.loglog(h_list,err_L2)

plt.title(str(k) + ' L2')

print('L2 relative errors',flush=True)

print(err_L2/fd.norms.norm(func_interp,norm_type="L2")) # As func_interp holds the interpolant of the true function on the finest mesh

plt.show()

plt.loglog(h_list,err_H1)

plt.title(str(k) + ' H1')

plt.show()

print('H1 relative errors',flush=True)

print(err_H1/fd.norms.norm(func_interp,norm_type="H1")) # as for L2

fit_L2 = np.polyfit(np.log(np.array(h_list)),np.log(np.array(err_L2)),deg=1)[0]

fit_H1 = np.polyfit(np.log(np.array(h_list)),np.log(np.array(err_H1)),deg=1)[0]

print(fit_L2,flush=True)

print(fit_H1,flush=True)


# Check interpolation is converging correctly


print(interp_integrals)

plt.loglog(h_list,np.abs(np.array(interp_integrals)-true_integral))

plt.title('Error in integrals of interpolants')

plt.show()
