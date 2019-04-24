import firedrake as fd
import numpy as np
from helmholtz_monte_carlo.generate_samples import eval_at_mesh_point
import sys

# Tests that the point evaluation code works correctly in parallel

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

            ensemble = fd.Ensemble(fd.COMM_WORLD,num_spatial_cores)

            spat_comm = ensemble.comm

            num_cells = 100
            
            meshes = [fd.UnitSquareMesh(num_cells,num_cells,comm=spat_comm)]#,fd.UnitCubeMesh(num_cells,num_cells,num_cells,comm=spat_comm)]
            
            for mesh in meshes:
                DG_0_space = fd.FunctionSpace(mesh,"DG",0)
                for V in [fd.FunctionSpace(mesh,"DG",1),fd.FunctionSpace(mesh,"CG",1),DG_0_space]:
                    v = fd.Function(V)

                    x = np.array(fd.SpatialCoordinate(mesh))
                    
                    v.interpolate(x.prod())

                    origin = tuple([0.0 for ii in range(mesh.geometric_dimension())])

                    top_right = tuple([1.0 for ii in range(mesh.geometric_dimension())])

                    top_left = (0.0,1.0)
                    
                    computed_origin = eval_at_mesh_point(v,origin,ensemble.ensemble_comm)

                    computed_top_right = eval_at_mesh_point(v,top_right,ensemble.ensemble_comm)

                    computed_top_left = eval_at_mesh_point(v,top_left,ensemble.ensemble_comm)

                    actual_origin = 0.0

                    actual_top_right = 1.0

                    actual_top_left = 0.0

                    # Need to hack things slightly, because the DG_0
                    # space doesn't interpolate continuous functions
                    # so well
                    if V is DG_0_space:
                        # This is a rough estimate of the error when
                        # interpolating in the DG_0 space
                        atol = 2.0*1.0/(2.0*float(num_cells))
                    else:
                        atol = 1e-16
                    
                    assert np.isclose(computed_origin,actual_origin,atol=atol)

                    assert np.isclose(computed_top_right,actual_top_right,atol=atol)

                    assert np.isclose(computed_top_left,actual_top_left,atol=atol)
            
