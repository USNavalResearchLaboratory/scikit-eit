import gmsh
import meshio
import numpy as np
from pathlib import Path
from scipy.spatial import Voronoi
import skfem.io

# make gmsh more readable
_1D = 1
_2D = 2


#######################################################################################
def build_polygonal_mesh(N, LN, alpha, beta, h1, h2, outdir):
    """Build a regular polygonal mesh in gmsh format."""
    name = f"N{N:02d}_LN{LN:02d}_A{int(alpha*100):03d}_B{int(alpha*100):03d}_H{int(h1*1000):04d}_H{int(h2*1000):04d}"
    mesh_file = Path(outdir)/f"{name}.msh"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)

    r = np.linspace(beta*np.cos(np.pi/N), 0, int(beta*np.cos(np.pi/N)/h2), endpoint=False)
    dr = r[0]-r[1]
    cn = 2*np.pi*r / dr
    a = [np.linspace(0, 2*np.pi, int(n), endpoint=False) for n in cn]
    pixels = np.vstack([rr * np.vstack([np.cos(aa), np.sin(aa)]).T for rr, aa in zip(r, a)] + [[0,0]])
    for _ in range(50):
        verts, regions, b = _partition(N, pixels, a=.01)
        pixels = np.array([verts[r].mean(axis=0) for r in regions[1:]])
    pts = list()
    for r in regions[1:]:
        if not any([p in b for p in r]):
            continue
        pts += [p for p in r if p not in b]
    pts = np.unique(pts)
    pts_verts = verts[pts]
    asort = np.argsort([np.arctan2(x, y) for x, y in verts[pts]])
    inner_pts = pts[asort].tolist()
    inner_pts += [inner_pts[0]]

    L = LN * N
    electrodes = [f"e{i}" for i in range(L)]
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 3)

        # Compute the points on the boundary
        angles = np.linspace(0,2*np.pi,N,endpoint=False) - np.pi/N - np.pi/2
        corner_pts = np.array([[np.cos(a), np.sin(a)] for a in angles])
        boundaryt_pts = list()
        for i in range(N):
            j = (i+1)%N
            dp = corner_pts[j] - corner_pts[i]
            p_hat = dp/np.linalg.norm(dp)
            n_gaps = LN + 1
            gap_size = (1-alpha)*np.linalg.norm(dp)/n_gaps
            electrode_size = alpha*np.linalg.norm(dp)/LN
            boundaryt_pts += [corner_pts[i]]
            for k in range(1,n_gaps):
                p = corner_pts[i] + p_hat*k*(gap_size+electrode_size)
                boundaryt_pts += [p - p_hat*electrode_size, p]
        boundaryt_pts = np.array(boundaryt_pts)    
        boundary_pt_ids = [gmsh.model.geo.add_point(*p, 0) for p in boundaryt_pts]

        # Add boundary lines
        boundary_lines = list()
        electrode_lines = list()
        electrode_pt_ids = list()
        k = 0
        for i in range(N*(LN*2+1)):
            j = (i+1)%len(boundaryt_pts)
            line = gmsh.model.geo.add_line(boundary_pt_ids[i], boundary_pt_ids[j])
            boundary_lines += [line]
            if k % 2 == 1:
                electrode_pt_ids += [boundary_pt_ids[i], boundary_pt_ids[j]]
                electrode_lines += [line]
            k = (k + 1) % (LN*2+1)

        # Add the pixels
        nid = 1
        pmap = dict()
        lmap = dict()
        for i, v in enumerate(verts):
            if i in b:
                continue
            pmap[i] = gmsh.model.geo.add_point(v[0], v[1], 0)
        for j,r in enumerate(regions[1:]):
            if any([p in b for p in r]):
                continue
            for i in range(len(r)):
                if (r[i], r[(i+1)%len(r)]) not in lmap:
                    lmap[(r[i], r[(i+1)%len(r)])] = gmsh.model.geo.add_line(pmap[r[i]], pmap[r[(i+1)%len(r)]])
                    lmap[(r[(i+1)%len(r)], r[i])] = -lmap[(r[i], r[(i+1)%len(r)])]
            curve = [ lmap[(r[i], r[(i+1)%len(r)])] for i in range(len(r)) ]
            px_loop = gmsh.model.geo.add_curve_loop(curve)
            px = gmsh.model.geo.add_plane_surface([px_loop])
            gmsh.model.add_physical_group(_2D, [px])

            gmsh.model.set_physical_name(
                _2D, gmsh.model.add_physical_group(_2D, [px]), f"s{nid}")
            nid += 1

        gmsh.model.geo.synchronize()
        inner = [lmap[(inner_pts[a],inner_pts[a+1])] for a in range(len(inner_pts)-1)]
        domain_boundary = gmsh.model.geo.add_curve_loop(boundary_lines+inner)
        domain = gmsh.model.geo.add_plane_surface([domain_boundary])
        gmsh.model.set_physical_name(
            _2D, gmsh.model.add_physical_group(_2D, [domain]), f"s0")
        gmsh.model.geo.synchronize()
        for i, e in enumerate(electrode_lines):
            gmsh.model.set_physical_name(
                _1D, gmsh.model.add_physical_group(_1D, [e]), electrodes[i])
        gmsh.model.add_physical_group(_2D, [domain])
        gmsh.model.geo.synchronize()
 
        # Build the fields
        field_list = list()
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "PointsList", electrode_pt_ids)
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(field, "SizeMin", h1/5)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", h1)
        gmsh.model.mesh.field.setNumber(field, "DistMin", .1)
        gmsh.model.mesh.field.setNumber(field, "DistMax", 2)
        field_list.append(field)

        # Use the minimum in a list of fields as the background mesh field:
        mesh_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mesh_field, "FieldsList", field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(mesh_field)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)        
        
        gmsh.model.mesh.generate(_2D)
        gmsh.write(str(mesh_file))
    finally:
        gmsh.clear()
        gmsh.finalize() 
    return Path(mesh_file)



def _partition(N, pixels, a):
    angles = np.linspace(0,2*np.pi,N,endpoint=False) - np.pi/N - np.pi/2
    vor = Voronoi(pixels)
    unused_verts = list()
    verts = np.vstack([
        [[np.cos(a), np.sin(a)] for a in angles],  # outside corners
        vor.vertices,  # inside voronoi vertices
    ])
    boundary_verts = list(range(N))
    regions = [list(range(N))]  # outside boundary
    domain_lines = verts[[(i, (i+1)%N) for i in range(N)]]

    # adjust ridge_verts map to account for the outside points
    ridge_verts = np.array([
        [-1 if rv[0]<0 else rv[0]+N, rv[1]+N]  # don't change the -1's
        for rv in vor.ridge_vertices
    ])
    ridge_points = np.array(vor.ridge_points)

    # fix the -1's in ridge_verts
    for rp, (i, rv) in zip(ridge_points, enumerate(ridge_verts)):
        if -1 in rv:
            p, q = pixels[rp]
            ridge_dir = np.array([q[1]-p[1], p[0]-q[0]])
            ridge_dir *= pixels[rp].mean(axis=0) @ ridge_dir
            ridge_dir *= 1/np.linalg.norm(ridge_dir)
            ridge_verts[i,0] = verts.shape[0]
            verts = np.vstack([verts, verts[rv[1]] + ridge_dir])

    # drop ridges are outside domain
    unused_ridges = list()
    for rp, (i, rv) in zip(ridge_points, enumerate(ridge_verts)):
        if not any([_in_polygon(domain_lines, verts[rv[i]]) for i in [0,1]]):
            unused_verts += [rv[0], rv[1]]
            unused_ridges.append(i)
    ridge_points = np.delete(ridge_points, unused_ridges, axis=0)
    ridge_verts = np.delete(ridge_verts, unused_ridges, axis=0)        

    # move outside endpoints to boundary
    # watch out for collisions with corner points!
    for rp, (i, rv) in zip(ridge_points, enumerate(ridge_verts)):
        for line in domain_lines:
            if _do_intersect(verts[rv], line):
                new_vert = _intersection_point(verts[rv], line)
                d = np.linalg.norm(verts-new_vert, axis=1)
                if d.min() < a:  # collision with existing vert
                    vert_id = d.argmin()
                else:
                    vert_id = verts.shape[0]
                    verts = np.vstack([verts, new_vert])
                    boundary_verts.append(vert_id)
                if _in_polygon(domain_lines, verts[rv[0]]):
                    unused_verts.append(rv[1])
                    ridge_verts[i,1] = vert_id
                else:
                    unused_verts.append(rv[0])
                    ridge_verts[i,0] = vert_id
                break


    # assemble interior region lists
    for pixel_id in range(pixels.shape[0]):
        pixel_verts = list()
        for rp, (i, rv) in zip(ridge_points, enumerate(ridge_verts)):
            if pixel_id in rp:
                pixel_verts += [rv[0], rv[1]]
        corners = list()
        for j, p in enumerate(verts[:N]):
            line0 = np.vstack([p, pixels[pixel_id]])        
            for i in range(len(pixel_verts)//2):
                line1 = verts[[pixel_verts[i*2], pixel_verts[i*2+1]]]
                if _do_intersect(line0, line1):
                    break
            else:
                corners.append(j)
        pixel_verts = np.unique(np.array(pixel_verts+corners))
        asort = np.argsort([np.arctan2(x, y) for x, y in verts[pixel_verts] - pixels[pixel_id]])
        regions.append(pixel_verts[asort])

    # discard unused vertices
    remap = dict()
    i = 0
    for r in regions:
        for p in r:
            if p not in remap:
                remap[p] = i
                i += 1
    regions = [[remap[p] for p in r] for r in regions]
    boundary_verts = [remap[p] for p in boundary_verts]
    verts = np.array([verts[i] for i in remap.keys()])
    return verts, regions, boundary_verts


def _is_left_of(line, pt):
    return (line[1][0]-line[0][0])*(pt[1]-line[0][1]) - (line[1][1]-line[0][1])*(pt[0]-line[0][0]) > 0


def _in_polygon(plines, pt):
    for line in plines:
        if not _is_left_of(line, pt):
            return False
    return True


def _orientation(p0, p1, p2):
    """Return orientation of three points
    Returns +1 for clockwise, -1 for counterclockwise, and zero for colinear
    """
    return np.sign(((p1[1] - p0[1]) * (p2[0] - p1[0])) - ((p1[0] - p0[0]) * (p2[1] - p1[1])))


def _do_intersect(line0, line1):
    p1,q1 = line0[0], line0[1]
    p2,q2 = line1[0], line1[1]    
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    return ((o1 != o2) and (o3 != o4))


def _intersection_point(line0, line1):
    return np.linalg.solve(
        np.array([
            [line0[0,1]-line0[1,1], line0[1,0]-line0[0,0]],
            [line1[0,1]-line1[1,1], line1[1,0]-line1[0,0]],
        ]),
        -1 * np.array([
            line0[1,1]*line0[0,0] - line0[1,0]*line0[0,1],
            line1[1,1]*line1[0,0] - line1[1,0]*line1[0,1],
        ]))