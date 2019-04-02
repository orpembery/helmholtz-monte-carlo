import firedrake as fd
import numpy as np

def complex_integral(f):
    """f is Firedrake function in complex."""

    V = f.function_space()
    
    func_real = fd.Function(V)
    func_imag = fd.Function(V)
    func_real.dat.data[:] = np.real(f.dat.data)
    func_imag.dat.data[:] = np.imag(f.dat.data)

    return fd.assemble(func_real * fd.dx) + 1j * fd.assemble(func_imag * fd.dx)
