import gmsh
import meshio
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi
import skfem.io
from tqdm.notebook import tqdm

# make gmsh more readable
_1D = 1
_2D = 2


def _get_voronoi(N, beta, h1):
    r0 = 1/np.sqrt(N*np.cos(np.pi/N)*np.sin(np.pi/N))
    r1 = r0*np.cos(np.pi/N)
    radii =  np.linspace(beta*r1, 0, int(beta*r1/h1), endpoint=False)
    na = [int(2*np.pi*r/h1) for r in radii]
    da = [2*np.pi/na0 for na0 in na]
    a = [da0*np.arange(na0) for na0, da0 in zip(na, da)]
    x = np.hstack([r*np.cos(a0) for a0,r in zip(a,radii)] + [0])
    y = np.hstack([r*np.sin(a0) for a0,r in zip(a,radii)] + [0])
    return Voronoi(np.vstack([x,y]).T)

def build_polygon_mesh(N, LN, alpha, beta, h1, h2, outdir):
    name = f"N{N:02d}_LN{LN:02d}_A{int(alpha*100):03d}_B{int(beta*100):03d}_H{int(h1*1000):04d}_H{int(h2*1000):04d}"
    mesh_file = Path(outdir)/f"poly_{name}.msh"    

    r0 = 1/np.sqrt(N*np.cos(np.pi/N)*np.sin(np.pi/N))
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) - np.pi/N - np.pi/2
    outer_pts = r0*np.array([[np.cos(a), np.sin(a)] for a in angles])
    electrodes = [f"e{i}" for i in range(LN*N)]
    vor = _get_voronoi(N, beta, h1)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 3)

        boundaryt_pts = list()
        for i in range(N):
            j = (i+1)%N
            dp = outer_pts[j] - outer_pts[i]
            p_hat = dp/np.linalg.norm(dp)
            n_gaps = LN + 1
            gap_size = (1-alpha)*np.linalg.norm(dp)/n_gaps
            electrode_size = alpha*np.linalg.norm(dp)/LN
            boundaryt_pts += [outer_pts[i]]
            for k in range(1,n_gaps):
                p = outer_pts[i] + p_hat*k*(gap_size+electrode_size)
                boundaryt_pts += [p - p_hat*electrode_size, p]
        boundaryt_pts = np.array(boundaryt_pts)    
        boundary_pt_ids = [gmsh.model.geo.add_point(*p, 0) for p in boundaryt_pts]

        # Add boundary lines
        boundary_lines = list()
        electrode_lines = list()
        k = 0
        for i in range(N*(LN*2+1)):
            j = (i+1)%len(boundaryt_pts)
            line = gmsh.model.geo.add_line(boundary_pt_ids[i], boundary_pt_ids[j])
            boundary_lines += [line]
            if k % 2 == 1:
                electrode_lines += [line]
            k = (k + 1) % (LN*2+1)
            
        # Add inner points
        pmap = {
            i:gmsh.model.geo.add_point(*p, 0)
            for i, p in enumerate(vor.vertices)
        }
        
        # inner facets
        lmap = dict()
        for rv in vor.ridge_vertices:
            if rv[0] >=0:
                key = (rv[0], rv[1])
                ikey = (rv[1], rv[0])
                if key not in lmap and ikey not in lmap:
                    lmap[key] = gmsh.model.geo.add_line(pmap[rv[0]], pmap[rv[1]])
                    lmap[ikey] = -lmap[key]

        # inner border
        points = list()    
        lines = list()
        x = np.unique(vor.ridge_points[(np.array(vor.ridge_vertices)[:,0] < 0)])
        for i, rp in enumerate(vor.ridge_points):
            if (vor.ridge_vertices[i][0] >= 0) and (rp[0] in x or rp[1] in x):
                points.append(vor.ridge_vertices[i])
        points = np.unique(points)
        asort = np.argsort(np.arctan2(vor.vertices[points][:,0], vor.vertices[points][:,1]))
        points = points[asort]
        for i in range(len(points)):
            j = (i+1)%len(points)
            key = (points[i], points[j])
            lines.append(lmap[key])
        inner_loop = gmsh.model.geo.add_curve_loop(lines)
        outer_loop = gmsh.model.geo.add_curve_loop(boundary_lines)
        center = gmsh.model.geo.add_plane_surface([inner_loop])
        edge =  gmsh.model.geo.add_plane_surface([inner_loop, outer_loop])
        gmsh.model.geo.synchronize()
        gmsh.model.set_physical_name(
            _2D, gmsh.model.add_physical_group(_2D, [edge]), f"s0")
        
        # inner elements
        rv = np.array(vor.ridge_vertices)
        rp = np.array(vor.ridge_points)
        p = np.unique(rp[rv[:,0]<0])
        q = [pp for pp in np.unique(rp) if pp not in p]

        cells = list()
        for k, q0 in enumerate(q):
            lines = list()
            r = [i for i, r in enumerate(rp) if r[0]==q0 or r[1]==q0]
            verts = np.unique(rv[r])
            vert_pts = vor.vertices[verts]
            asort = np.argsort(np.arctan2(vert_pts[:,0]-vor.points[q0][0], vert_pts[:,1]-vor.points[q0][1]))
            verts = verts[asort]
            for i in range(len(verts)):
                j = (i+1)%len(verts)
                key = (verts[i], verts[j])
                lines.append(lmap[key])
            loop = gmsh.model.geo.add_curve_loop(lines)
            cell = gmsh.model.geo.add_plane_surface([loop])
            cells.append((cell, f"s{k+1}"))
        gmsh.model.geo.synchronize()
        for cell, label in cells:
            gmsh.model.set_physical_name(
                _2D, gmsh.model.add_physical_group(_2D, [cell]), label)
        for i, e in enumerate(electrode_lines): 
            gmsh.model.set_physical_name(
                _1D, gmsh.model.add_physical_group(_1D, [e]), electrodes[i])
        gmsh.model.geo.synchronize()

        # Build the fields
        field_list = list()
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", electrode_lines)
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(field, "SizeMin", h2/8)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", h2)
        gmsh.model.mesh.field.setNumber(field, "DistMin", 0.1*r0)
        gmsh.model.mesh.field.setNumber(field, "DistMax", 4*r0)
        field_list.append(field)

        # Use the minimum in a list of fields as the background mesh field:
        mesh_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mesh_field, "FieldsList", field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(mesh_field)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)        

        gmsh.model.geo.synchronize()    
        gmsh.model.mesh.generate(_2D)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.clear()
        gmsh.finalize() 
    return mesh_file

def build_circle_mesh(N, LN, alpha, beta, h1, h2, outdir):
    name = f"N{N:02d}_LN{LN:02d}_A{int(alpha*100):03d}_B{int(beta*100):03d}_H{int(h1*1000):04d}_H{int(h2*1000):04d}"
    mesh_file = Path(outdir)/f"circ_{name}.msh"    

    r0 = 1/np.sqrt(np.pi)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) - np.pi/N - np.pi/2
    outer_pts = r0*np.array([[np.cos(a), np.sin(a)] for a in angles])
    electrodes = [f"e{i}" for i in range(LN*N)]
    gap, ele = np.linalg.solve(
    [[1+LN, LN],
        [0,    LN],],
        [2*np.pi/N, alpha*2*np.pi/N]
    )
    vor = _get_voronoi(N, beta, h1)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 3)
        a0 = np.arctan2(outer_pts[0,0], outer_pts[0,1])
        
        boundaryt_pts = list()
        for i in range(N):
            x0, y0 = outer_pts[i,0], outer_pts[i,1]
            a0 = np.arctan2(y0, x0)
            boundaryt_pts += [outer_pts[i]]        
            for _ in range(LN):
                boundaryt_pts += [[r0*np.cos(a0+gap), r0*np.sin(a0+gap)]]
                boundaryt_pts += [[r0*np.cos(a0+gap+ele), r0*np.sin(a0+gap+ele)]]
                a0 += gap + ele
        boundaryt_pts = np.array(boundaryt_pts)    
        boundary_pt_ids = [gmsh.model.geo.add_point(*p, 0) for p in boundaryt_pts]
        center_pt = gmsh.model.geo.add_point(0, 0, 0)
        
        # Add boundary lines
        boundary_lines = list()
        electrode_lines = list()
        electrode_pt_ids = list()
        k = 0
        for i in range(N*(LN*2+1)):
            j = (i+1)%len(boundaryt_pts)
            line = gmsh.model.geo.add_circle_arc(boundary_pt_ids[i], center_pt, boundary_pt_ids[j])
            boundary_lines += [line]
            if k % 2 == 1:
                electrode_pt_ids += [boundary_pt_ids[i], boundary_pt_ids[j]]
                electrode_lines += [line]
            k = (k + 1) % (LN*2+1)
        outer_loop = gmsh.model.geo.add_curve_loop(boundary_lines)

        # Add inner points
        pmap = {
            i:gmsh.model.geo.add_point(*p, 0)
            for i, p in enumerate(vor.vertices)
        }
        
        # inner facets
        lmap = dict()
        for rv in vor.ridge_vertices:
            if rv[0] >=0:
                key = (rv[0], rv[1])
                ikey = (rv[1], rv[0])
                if key not in lmap and ikey not in lmap:
                    lmap[key] = gmsh.model.geo.add_line(pmap[rv[0]], pmap[rv[1]])
                    lmap[ikey] = -lmap[key]

        # inner border
        points = list()    
        lines = list()
        x = np.unique(vor.ridge_points[(np.array(vor.ridge_vertices)[:,0] < 0)])
        for i, rp in enumerate(vor.ridge_points):
            if (vor.ridge_vertices[i][0] >= 0) and (rp[0] in x or rp[1] in x):
                points.append(vor.ridge_vertices[i])
        points = np.unique(points)
        asort = np.argsort(np.arctan2(vor.vertices[points][:,0], vor.vertices[points][:,1]))
        points = points[asort]
        for i in range(len(points)):
            j = (i+1)%len(points)
            key = (points[i], points[j])
            lines.append(lmap[key])
        inner_loop = gmsh.model.geo.add_curve_loop(lines)
        # outer_loop = gmsh.model.geo.add_curve_loop(boundary_lines)
        center = gmsh.model.geo.add_plane_surface([inner_loop])
        edge =  gmsh.model.geo.add_plane_surface([inner_loop, outer_loop])
        gmsh.model.set_physical_name(
            _2D, gmsh.model.add_physical_group(_2D, [edge]), f"s0")
        
        # inner elements
        rv = np.array(vor.ridge_vertices)
        rp = np.array(vor.ridge_points)
        p = np.unique(rp[rv[:,0]<0])
        q = [pp for pp in np.unique(rp) if pp not in p]

        for k, q0 in enumerate(q):
            lines = list()
            r = [i for i, r in enumerate(rp) if r[0]==q0 or r[1]==q0]
            verts = np.unique(rv[r])
            vert_pts = vor.vertices[verts]
            asort = np.argsort(np.arctan2(vert_pts[:,0]-vor.points[q0][0], vert_pts[:,1]-vor.points[q0][1]))
            verts = verts[asort]
            for i in range(len(verts)):
                j = (i+1)%len(verts)
                key = (verts[i], verts[j])
                lines.append(lmap[key])
            loop = gmsh.model.geo.add_curve_loop(lines)
            cell = gmsh.model.geo.add_plane_surface([loop])
            gmsh.model.set_physical_name(
                _2D, gmsh.model.add_physical_group(_2D, [cell]), f"s{k+1}")
            
        for i, e in enumerate(electrode_lines):
            gmsh.model.set_physical_name(
                _1D, gmsh.model.add_physical_group(_1D, [e]), electrodes[i])
        gmsh.model.geo.synchronize()

        h2 = .05
        # Build the fields
        field_list = list()
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", electrode_lines)
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(field, "SizeMin", h2/8)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", h2)
        gmsh.model.mesh.field.setNumber(field, "DistMin", 0.1*r0)
        gmsh.model.mesh.field.setNumber(field, "DistMax", 4*r0)
        field_list.append(field)

        # Use the minimum in a list of fields as the background mesh field:
        mesh_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mesh_field, "FieldsList", field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(mesh_field)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)        

        gmsh.model.geo.synchronize()    
        gmsh.model.mesh.generate(_2D)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.clear()
        gmsh.finalize() 
    return mesh_file