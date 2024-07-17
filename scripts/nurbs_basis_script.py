#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import asarray, concatenate, float64, full, linspace, gradient
from pygeom.tools.basis import basis_functions, basis_derivatives

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

#%%
# Define Nurbs Basis Functions
k = asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float64)
u = linspace(-1.0, 5.0, 121, dtype=float64)

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

#%%
# Plot the Nurbs Basis First Derivative Degree 1
dNu_check: 'NDArray' = gradient(Nu, u, axis=1)

p = 1
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))
dNu = basis_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(u, dNu_check[i, :], label=f'dN_{i}^{p}')
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

#%%
# Plot the Nurbs Basis First Derivative Degree 2
dNu_check: 'NDArray' = gradient(Nu, u, axis=1)

p = 2
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))
dNu = basis_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(u, dNu_check[i, :], label=f'dN_{i}^{p}')
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

#%%
# Plot the Nurbs Basis First Derivative Degree 3
dNu_check: 'NDArray' = gradient(Nu, u, axis=1)

p = 3
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))
dNu = basis_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(u, dNu_check[i, :], label=f'dN_{i}^{p}')
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

#%%
# Plot the Nurbs Basis First Derivative Degree 4
dNu_check: 'NDArray' = gradient(Nu, u, axis=1)

p = 4
kcl = concatenate((full(p, k[0]), k, full(p, k[-1])))
dNu = basis_derivatives(p, kcl, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(u, dNu_check[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()
