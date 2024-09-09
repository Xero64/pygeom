#%%
# Import Dependencies
from typing import TYPE_CHECKING

from matplotlib.pyplot import figure
from numpy import asarray, concatenate, full, gradient, linspace
from pygeom.tools.basis import (basis_first_derivatives, basis_functions,
                                basis_second_derivatives)

if TYPE_CHECKING:
    from numpy.typing import NDArray

#%%
# Define Nurbs Basis Functions
k = asarray([0.0, 2.0, 4.0, 6.0, 8.0])
# u = linspace(-2.0, 10.0, 121)
u = linspace(0.0, 8.0, 81)

#%%
# Plot the Nurbs Basis Functions Degree 0
p = 0
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))

Nu = basis_functions(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 1
p = 1
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))

Nu = basis_functions(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 2
p = 2
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))

Nu = basis_functions(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

d2Nu = basis_second_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu.shape[0]):
    ax.plot(u, d2Nu[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()

d2Nu_check: 'NDArray' = gradient(dNu, u, axis=1)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu_check.shape[0]):
    ax.plot(u, d2Nu_check[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 3
p = 3
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))

Nu = basis_functions(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

d2Nu = basis_second_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu.shape[0]):
    ax.plot(u, d2Nu[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()

d2Nu_check: 'NDArray' = gradient(dNu, u, axis=1)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu_check.shape[0]):
    ax.plot(u, d2Nu_check[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 4
p = 4
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))

Nu = basis_functions(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

d2Nu = basis_second_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu.shape[0]):
    ax.plot(u, d2Nu[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()

d2Nu_check: 'NDArray' = gradient(dNu, u, axis=1)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(d2Nu_check.shape[0]):
    ax.plot(u, d2Nu_check[i, :], label=f'd2N_{i}^{p}')
_ = ax.legend()
