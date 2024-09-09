#%%
# Import Dependencies
from numpy import zeros
from pygeom.geom3d.vector import Vector, zero_vector

#%%
# Create Array Vector
A = zero_vector((2, 2))
A[0, 0] = Vector(1, 2, 3)
A[0, 1] = Vector(2, 3, 4)
A[1, 0] = Vector(3, 4, 5)
A[1, 1] = Vector(4, 5, 6)

B = zeros(2)
B[0] = 7
B[1] = 8

C = A@B

print(f'A = \n{A}\n')
print(f'B = \n{B}\n')
print(f'C = \n{C}\n')
