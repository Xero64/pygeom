#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import float64, gradient, linspace
from pygeom.tools.bernstein import bernstein_derivatives, bernstein_polynomials

if TYPE_CHECKING:
    from numpy.typing import NDArray
    Numeric = Union[float64, NDArray[float64]]

#%%
# Define Parameter
t = linspace(0.0, 1.0, 121, dtype=float64)

#%%
# Plot the Bernstein Polynomials Degree 0
p = 0
Bt = bernstein_polynomials(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Bt.shape[0]):
    ax.plot(t, Bt[i, :], label=f'B_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Polynomials Degree 1
p = 1
Bt = bernstein_polynomials(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Bt.shape[0]):
    ax.plot(t, Bt[i, :], label=f'B_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Derivatives Degree 1
dNu_check: 'NDArray' = gradient(Bt, t, axis=1)

p = 1
dBt = bernstein_derivatives(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dBt.shape[0]):
    ax.plot(t, dBt[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(t, dNu_check[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Polynomials Degree 2
p = 2
Bt = bernstein_polynomials(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Bt.shape[0]):
    ax.plot(t, Bt[i, :], label=f'B_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Derivatives Degree 2
dNu_check: 'NDArray' = gradient(Bt, t, axis=1)

p = 2
dBt = bernstein_derivatives(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dBt.shape[0]):
    ax.plot(t, dBt[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(t, dNu_check[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Polynomials Degree 3
p = 3
Bt = bernstein_polynomials(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Bt.shape[0]):
    ax.plot(t, Bt[i, :], label=f'B_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Derivatives Degree 3
dNu_check: 'NDArray' = gradient(Bt, t, axis=1)

p = 3
dBt = bernstein_derivatives(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dBt.shape[0]):
    ax.plot(t, dBt[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(t, dNu_check[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Polynomials Degree 4
p = 4
Bt = bernstein_polynomials(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Bt.shape[0]):
    ax.plot(t, Bt[i, :], label=f'B_{i}^{p}')
_ = ax.legend()

#%%
# Plot the Bernstein Derivatives Degree 4
dNu_check: 'NDArray' = gradient(Bt, t, axis=1)

p = 4
dBt = bernstein_derivatives(p, t)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dBt.shape[0]):
    ax.plot(t, dBt[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu_check.shape[0]):
    ax.plot(t, dNu_check[i, :], label=f'dB_{i}^{p}')
_ = ax.legend()
