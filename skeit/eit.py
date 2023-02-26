from asyncio import tasks
import gmsh
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import meshio
from multiprocessing import Pool
import numpy as np
import os
from pathlib import Path
import scipy.sparse
from skfem.helpers import dot, grad
import skfem.io
import skfem
import sys
import tarfile
import time
from tqdm import tqdm
from types import SimpleNamespace
import yaml

class EIT:
        
    def __getattr__(self, attr):
        return getattr(self.mesh, attr)
    
    def __init__(self, mesh, element=None, electrodes=None):
        if isinstance(mesh, str):
            mesh = Path(mesh)
        if isinstance(mesh, Path):
            mesh = self._read_mesh(mesh)
        self.mesh = mesh
        if element is None:
            element = skfem.ElementTriP1
        if electrodes is None:
            electrodes = self.boundaries
        self.basis = skfem.CellBasis(self.mesh, element())
        self.plot = self.basis.plot
        self.p0 = self.basis.with_element(skfem.ElementTriP0())
        self.e_basis = [skfem.FacetBasis(self.mesh, element(), facets=f) for f in electrodes]
        self.L = len(self.e_basis)
        self.P = np.zeros([self.L, self.L-1])
        self.P[0] = 1
        i = np.arange(self.L-1)
        self.P[i+1, i] = -1


    def jacobian(self, sigma0, zl):
        u, U0 = self.measure(sigma0, zl)
        g = [self.basis.interpolate(uu).grad for uu in u]
        J = np.empty([U0.shape[0], sigma0.shape[0]])
        for i in range(self.L):
            for j in range(self.L):
                J[i * self.L + j] = (-1 * dot(g[i], g[j]) * self.basis.dx).sum(-1)
        return U0, J
    
    
    def measure(self, sigma, zl):
        A = self.solve_forward(sigma, zl)

        U_list = list()
        for drive_pair in [[i, (i+1) % self.L] for i in range(self.L)]:
            current = np.zeros(self.L)
            current[drive_pair] = [1, -1]
            b = np.hstack([
                np.zeros(self.basis.N),
                (self.P.T @ current),
            ])
            result = scipy.sparse.linalg.spsolve(A, b)
            single_ended = self.P @ result[self.basis.N:]
            differential = single_ended - np.roll(single_ended, -1)
            U_list.append((result[:self.basis.N], differential))

        return np.array([U[0] for U in U_list]), np.array([U[1] for U in U_list]).reshape(-1)        
        
        
    def solve_forward(self, sigma, zl):

        @skfem.BilinearForm
        def _S_form(u, v, w):
            return w.sigma * dot(grad(u), grad(v))
        @skfem.BilinearForm
        def _M_form(u, v, _):
            return u * v
        @skfem.LinearForm
        def _C_form(u, _):
            return u
        @skfem.Functional
        def _mag_el_func(_):
            return 1

        S = _S_form.assemble(self.basis, sigma=self.p0.interpolate(sigma))

        M = np.sum([
            (1/zl) * _M_form.assemble(el)
            for el in self.e_basis
        ], axis=0)

        C = 1/zl * np.vstack([
            _C_form.assemble(self.e_basis[j+1])
            - _C_form.assemble(self.e_basis[0])
            for j in range(self.L-1)
        ]).T
        i,j,v = scipy.sparse.find(C)
        C = scipy.sparse.coo_matrix((v, (i,j)), shape=C.shape)

        e_mag = np.array([
            _mag_el_func.assemble(el)
            for el in self.e_basis
        ])
        G = np.ones([self.L-1, self.L-1]) * e_mag[0]/zl + np.diag(e_mag[1:]/zl)
        i,j,v = scipy.sparse.find(G)
        G = scipy.sparse.coo_matrix((v, (i,j)), shape=G.shape)

        A = scipy.sparse.bmat([
            [S+M, C],
            [C.T, G]
        ]).tocsr()
        
        return A
        
        
    @staticmethod
    def _read_mesh(mesh_file):
        """Read a gmsh file into an skfem Mesh object."""
        stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w")
        data = meshio.read(str(mesh_file))
        sys.stdout = stdout # reset stdout
        return skfem.io.meshio.from_meshio(data, out=None)    

#######################################################################################
def get_dof(basis, coord):
    return np.linalg.norm(basis.doflocs.T - coord, axis=1).argmin()


#######################################################################################
def add_electrodes(ax, N, LN, alpha, R=1, ls="-", lw=4, color="k"):
    angles = np.linspace(0,2*np.pi,N,endpoint=False) - np.pi/N - np.pi/2
    corner_pts = R * np.array([[np.cos(a), np.sin(a)] for a in angles])
    pts = list()
    for i in range(N):
        j = (i+1)%N
        dp = corner_pts[j] - corner_pts[i]
        p_hat = dp/np.linalg.norm(dp)
        n_gaps = LN + 1
        gap_size = (1-alpha)*np.linalg.norm(dp)/n_gaps
        electrode_size = alpha*np.linalg.norm(dp)/LN
        pts += [corner_pts[i]]
        for k in range(1,n_gaps):
            p = corner_pts[i] + p_hat*k*(gap_size+electrode_size)
            pts += [p - p_hat*electrode_size, p]
    pts = np.array(pts)
    lines = list()
    electrode_lines = list()
    k = 0
    for i in range(N*(LN*2+1)):
        j = (i+1)%len(pts)
        if k % 2 == 1:
            p0 = pts[i]
            p1 = pts[j]
            ax.plot([p0[0],p1[0]], [p0[1],p1[1]], ls=ls, lw=lw, color=color)
        k = (k + 1) % (LN*2+1)   

def moment(basis, data, n=0, x0=0, y0=0):
    @skfem.Functional
    def _moment(w):
        r = np.sqrt((w.x[0]-x0)**2 + (w.x[1]-y0)**2)
        return r**w.n * w.u
    return _moment.assemble(basis, u=data, n=n, x0=x0, y0=y0)

def x_moment(basis, data, n=0, x0=0):
    @skfem.Functional
    def _moment(w):
        r = w.x[0]-x0
        return r**w.n * w.u
    return _moment.assemble(basis, u=data, n=n, x0=x0)

def y_moment(basis, data, n=0, y0=0):
    @skfem.Functional
    def _moment(w):
        r = w.x[1]-y0
        return r**w.n * w.u
    return _moment.assemble(basis, u=data, n=n, y0=y0)

