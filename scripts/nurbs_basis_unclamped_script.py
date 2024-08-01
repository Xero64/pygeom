#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import asarray, float64, linspace, gradient, zeros
from pygeom.tools.basis import basis_functions, basis_first_derivatives, basis_second_derivatives

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

#%%
# Define Nurbs Basis Functions
k = asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=float64)
u = linspace(-2.0, 10.0, 121, dtype=float64)
u = linspace(0.0, 8.0, 81, dtype=float64)

#%%
# Plot the Nurbs Basis Functions Degree 0
p = 0

Nu = basis_functions(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 1
p = 1

Nu = basis_functions(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Nurbs Basis Functions Degree 2
p = 2

Nu = basis_functions(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

d2Nu = basis_second_derivatives(p, k, u)

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

Nu = basis_functions(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{p}')
_ = ax.legend()

dNu = basis_first_derivatives(p, k, u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{p}')
_ = ax.legend()

d2Nu = basis_second_derivatives(p, k, u)

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
