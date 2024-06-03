import numpy as np
import sys
import time
import copy
import os

# ------------------------------------------------------- #

from feFlow.physics import IncompressibleNavierStokes
from feFlow.io import h5_mod
from feFlow.mesh import Mesh
from feFlow.solver import PhysicsSolver
import fenics as fe


def calculate_vorticity(vel, function_space, file):
    omega = fe.curl(vel)  # gives mesh function
    omegaField = fe.project(omega, function_space)
    omegaField.rename('omega', 'vorticity')
    file << omegaField
    return

def calculate_shear(n, ds, mu, vel, function_space, file):
    sigma = mu * (fe.grad(vel) + fe.grad(u).T)
    T = sigma * n
    Tt = T - fe.inner(T, n) * n
    shear = fe.TrialFunction(function_space)
    w = fe.TestFunction(function_space)
    weak_form = fe.dot(shear, w)*ds - fe.dot(Tt, w)*ds
    lhs = fe.assemble(fe.lhs(weak_form), keep_diagonal=True)
    lhs.ident_zeros()
    rhs = fe.assemble(fe.rhs(weak_form))
    # solve
    shear_field = fe.Function(function_space)
    la_solver = fe.KrylovSolver('cg', 'jacobi')
    la_solver.parameters['monitor_convergence'] = True
    la_solver.solve(lhs, shear_field.vector(), rhs)
    functionName = 'wss'
    shear_field.rename(functionName, functionName)
    file << shear_field
    return

'''boundaries
2: out1.stl
3: inlet.stl
4: walls.stl
5: out2.stl
'''
mesh = Mesh(mesh_file='mesh.h5')
nse = IncompressibleNavierStokes(mesh)
nse.set_element('CG', 1, 'CG', 1)
nse.build_function_space()

output_dir = 'output'

dt = 0.01
mu = 1e-3
rho = 1

# Set parameters
nse.set_time_step_size(dt)
nse.set_mid_point_theta(0.5)
nse.set_density(rho)
nse.set_dynamic_viscosity(mu)

# Set weak form
nse.set_weak_form(stab=True)

inlet_vel = fe.Constant((0, 0, 0.1))
zero_vec = fe.Constant((0, 0, 0,))
zero_scalar = fe.Constant(0)

id_wall = 23
id_in = 22
id_left_out = 20
id_right_out = 21

u_bcs = {id_in: {'type': 'dirichlet', 'value' : inlet_vel},
         id_wall: {'type': 'dirichlet', 'value' : zero_vec}}
p_bcs = {id_left_out: {'type': 'dirichlet', 'value' : zero_scalar},
         id_right_out: {'type': 'dirichlet', 'value' : zero_scalar}}
bc_dict = {'u': u_bcs,
           'p': p_bcs}
nse.set_bcs(bc_dict)
nse.set_writer(output_dir, "pvd")
la_solver = fe.LUSolver()
solver = PhysicsSolver(nse, la_solver)

# custom parameters
V = fe.VectorFunctionSpace(mesh.mesh, 'CG', 1)
# vorticity setup
omega_file = fe.File(output_dir + '/omega.pvd')
# shear stress setup
shear_file = fe.File(output_dir + '/tau.pvd')
n_hat = fe.FacetNormal(mesh.mesh)
ds = fe.ds


rank = mesh.comm.rank
t = 0
while t < 0.5:
    t += dt

    solver.solve()
    nse.update_previous_solution()
    nse.write(time_stamp=t)

    (u, p) = fe.split(nse.solution)

    calculate_vorticity(u, V, omega_file)
    calculate_shear(n_hat, ds, mu, u, V, shear_file)

    if rank == 0:
        print('Solved time step t = {}'.format(t))
