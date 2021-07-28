import pymesh
import numpy as np
mesh = pymesh.load_mesh("*.obj")
mesh.add_attribute("vertex_gaussian_curvature")

data = mesh.get_attribute("vertex_gaussian_curvature")
np.savez('./cur.npz', cur = np.array(data))