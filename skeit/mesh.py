import gmsh
import meshio
import numpy as np
from pathlib import Path
import skfem.io

# make gmsh more readable
_1D = 1
_2D = 2


#######################################################################################
def build_polygonal_mesh(N, LN, alpha, h, outdir, circle=None, polygon=None):
    """Build a regular polygonal mesh in gmsh format."""
    name = f"N{N:02d}_LN{LN:02d}_A{int(alpha*100):03d}_H{int(h*1000):04d}"
    mesh_file = Path(outdir)/f"{name}.msh"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    L = N*LN  # number of electrodes
    electrodes = [f"e{i}" for i in range(L)]

    angles = np.linspace(0,2*np.pi,N,endpoint=False) - np.pi/N - np.pi/2
    corner_pts = np.array([[np.cos(a), np.sin(a)] for a in angles])
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
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 3)
        for p in pts:
            gmsh.model.geo.add_point(*p, 0)
        lines = list()
        electrode_lines = list()
        electrode_pts = list()
        k = 0
        for i in range(N*(LN*2+1)):
            j = (i+1)%len(pts)
            line = gmsh.model.geo.add_line(i+1, j+1)
            lines += [line]
            if k % 2 == 1:
                electrode_pts += [i+1, j+1]
                electrode_lines += [line]
            k = (k + 1) % (LN*2+1)
        domain_tags = [gmsh.model.geo.add_curve_loop(lines), ]
        target_loop_tag = None
        if circle is not None:
            x0 = circle["center"][0]
            y0 = circle["center"][1]
            r0 = circle["radius"]
            circle_pts = r0 * np.array([[np.cos(a), np.sin(a)] for a in angles])
            circle_pts[:,0] += x0
            circle_pts[:,1] += y0
            pt_tags = [gmsh.model.geo.add_point(*p, 0) for p in circle_pts]
            center_tag = gmsh.model.geo.add_point(x0, y0, 0)
            segment_tags = [gmsh.model.geo.add_circle_arc(pt_tags[i],
                                                      center_tag,
                                                      pt_tags[(i+1) % len(pt_tags)])
                        for i in range(len(pt_tags))]
            target_loop_tag = gmsh.model.geo.add_curve_loop(segment_tags)
            domain_tags.append(target_loop_tag)
        elif polygon is not None:
            pt_tags = [gmsh.model.geo.add_point(*p, 0) for p in polygon]
            segment_tags = [gmsh.model.geo.add_line(pt_tags[i],
                                                pt_tags[(i+1) % len(pt_tags)])
                        for i in range(len(pt_tags))]
            target_loop_tag = gmsh.model.geo.add_curve_loop(segment_tags)
            domain_tags.append(target_loop_tag)

        gmsh.model.geo.add_plane_surface(domain_tags)
        if target_loop_tag is not None:
            target = gmsh.model.geo.add_plane_surface([target_loop_tag])
        gmsh.model.geo.synchronize()
        gmsh.model.add_physical_group(_2D, [1])
        for i, e in enumerate(electrode_lines):
            gmsh.model.set_physical_name(_1D,
                                         gmsh.model.add_physical_group(_1D, [e]),
                                         electrodes[i])
        if target_loop_tag is not None:
            gmsh.model.set_physical_name(_2D,
                                        gmsh.model.add_physical_group(_2D, [target]),
                                        "t")



        # Build the fields
        field_list = list()

        # Electrode endpoints
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "PointsList", electrode_pts)
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(field, "SizeMin", h/5)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(field, "DistMin", .05)
        gmsh.model.mesh.field.setNumber(field, "DistMax", 1)
        field_list.append(field)

        # Target boundary
        if target_loop_tag is not None:
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", segment_tags)
            field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(field, "InField", distance_field)
            gmsh.model.mesh.field.setNumber(field, "SizeMin", h/5)
            gmsh.model.mesh.field.setNumber(field, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(field, "DistMin", .05)
            gmsh.model.mesh.field.setNumber(field, "DistMax", 1)
            field_list.append(field)

        # Use the minimum in a list of fields as the background mesh field:
        mesh_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(mesh_field, "FieldsList", field_list)
        gmsh.model.mesh.field.setAsBackgroundMesh(mesh_field)

        gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
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