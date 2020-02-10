import numpy as np
import pptk
x = np.random.rand(100, 3)
print(x)
v = pptk.viewer(np.array([[0, 0.5, 0], [0, 0, 0.5]]))
v.set(point_size=0.01)

# blue is z
# red is x
# gree is y
