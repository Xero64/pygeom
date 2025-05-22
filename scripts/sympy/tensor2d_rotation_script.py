#%%
# Import Dependencies
from sympy import Matrix, Symbol, simplify, Expr, expand

#%%
# Create Symbolic Variables
txx = Symbol('txx', real=True)
txy = Symbol('txy', real=True)
tyx = Symbol('tyx', real=True)
tyy = Symbol('tyy', real=True)

T = Matrix([[txx, txy], [tyx, tyy]])

c = Symbol('c', real=True)
s = Symbol('s', real=True)

R = Matrix([[c, -s], [s, c]])

#%%
# Create Rotation Matrix
Tr1 = R @ T @ R.transpose()
print(f'Tr1 = \n{Tr1}\n')

Tr2 = R @ T @ R.inv()
print(f'Tr2 = \n{Tr2}\n')

Tr1 = simplify(Tr1)
Tr2: Expr = simplify(Tr2)
Tr2 = Tr2.subs(c**2 + s**2, 1)
print(f'Tr1 = \n{Tr1}\n')
print(f'Tr2 = \n{Tr2}\n')

#%%
sxx = expand(Tr1[0, 0])
sxy = expand(Tr1[0, 1])
syx = expand(Tr1[1, 0])
syy = expand(Tr1[1, 1])

print(f'sxx = {sxx}\n')
print(f'sxy = {sxy}\n')
print(f'syx = {syx}\n')
print(f'syy = {syy}\n')
