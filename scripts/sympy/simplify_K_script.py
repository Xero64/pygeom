#%%
# Import Dependencies
from sympy import cos, Symbol, Rational, Expr, sqrt

#%%
# Create Symbols
ang = Symbol('ang', real=True)
add = Symbol('add', real=True)
adm = Symbol('adm', real=True)

#%%
# Define K
K: Expr = Rational(4, 3)/(1/cos(ang/2) + 1)
print(f'K = {K}\n')

K = K.subs(cos(ang/2), sqrt((adm + add)/2)/sqrt(adm))
print(f'K = {K}\n')

K: Expr = K.expand()
K = K.simplify()
print(f'K = {K}\n')

K_check_1 = K.subs(add, -adm)
print(f'K_check_1 = {K_check_1}\n')

K_check_2 = K.subs(add, 0).simplify()
print(f'K_check_2 = {K_check_2}\n')

K_check_3 = K.subs(add, adm).simplify()
print(f'K_check_3 = {K_check_3}\n')
