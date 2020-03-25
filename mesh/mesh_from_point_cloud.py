import pymesh
from mayavi import mlab
import numpy as np

vertices = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0]
])

tri = pymesh.triangle()
tri.points = vertices
tri.max_area = 0.05
tri.split_boundary = False
tri.verbosity = 0
tri.run()
mesh = tri.mesh
print(mesh)
pymesh.save_mesh("test.obj", mesh)

vertices = mesh.vertices
faces = mesh.faces

print(vertices)
print(faces)

x, y, z = zip(*vertices)
x = np.array(x)
y = np.array(y)
z = np.array(z)

print(x)
print(y)
print(z)

mayavi_mesh = mlab.triangular_mesh(x, y, z, faces)

mlab.show()
