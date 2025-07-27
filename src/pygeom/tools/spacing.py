from typing import TYPE_CHECKING

from numpy import asarray, cos, cumsum, linspace, pi, sqrt, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Consider using linspace and geomspace functions from numpy

def normalise_spacing(spacing: 'NDArray') -> 'NDArray':
    smin = spacing.min()
    smax = spacing.max()
    return (spacing-smin)/(smax-smin)

def semi_cosine_spacing(num: int) -> 'NDArray':
    th = linspace(pi/2, 0, num + 1)
    spc = cos(th)
    spc[0] = 0.0
    return spc

def full_cosine_spacing(num: int) -> 'NDArray':
    th = linspace(pi, 0.0, num + 1)
    spc = (cos(th) + 1.0)/2
    return spc

def equal_spacing(num: int) -> 'NDArray':
    spc = linspace(0.0, 1.0, num + 1)
    return spc

def linear_bias_left(spc: 'NDArray', ratio: float) -> 'NDArray':
    ratio = abs(ratio)
    if ratio > 1.0:
        ratio = 1.0/ratio
    m = 1.0 - ratio
    return spc*(ratio + m*spc)

def linear_bias_right(spc: 'NDArray', ratio: float) -> 'NDArray':
    ratio = abs(ratio)
    if ratio > 1.0:
        ratio = 1.0/ratio
    m = 1.0 - ratio
    return 1.0 - (1.0 - spc)*(ratio + m*(1.0 - spc))

def geometric_series_spacing_ratio(num: int, ratio: float) -> 'NDArray':
    if ratio == 1.0:
        raise ValueError('The input ratio must not equal 1.0.')
    ds0 = (1.0-ratio)/(1.0-ratio**num)
    ds = asarray([ds0*ratio**i for i in range(num)])
    spc = zeros(num+1)
    spc[1:] = cumsum(ds)
    return spc

def geometric_series_spacing_ds0(num: int, ds0t: float,
                                 display: bool=False) -> 'NDArray':
    if ds0t == 1/num:
        return equal_spacing(num)
    else:
        n = num
        b = num*(num-1)/2
        c = num-1/ds0t
        if num == 2:
            ratio = -c/b + 1
            if display:
                print(f'b = {b}')
                print(f'c = {c}')
        else:
            a = num*(num-1)*(num-2)/6
            d = b**2-4*a*c
            if display:
                print(f'a = {a}')
                print(f'b = {b}')
                print(f'c = {c}')
                print(f'd = {d}')
            d = max(0.0, d)
            if display:
                print(f'd = {d}')
            ratio = (b + sqrt(d))/2/a + 1.0
        dsit = 1/ds0t
        dsic = (1.0-ratio**n)/(1.0-ratio)
        Rc = dsit - dsic
        dRdr = (n*ratio**n*(ratio - 1) + ratio*(1 - ratio**n))/(ratio*(ratio - 1)**2)
        delta = Rc/dRdr
        count = 0
        if display:
            print(count, ratio, delta)
        while abs(delta) > 1e-5:
            num_dRdr = n*ratio**n*(ratio - 1) + ratio*(1 - ratio**n)
            den_dRdr = ratio*(ratio - 1)**2
            dRdr = num_dRdr/den_dRdr
            delta = Rc/dRdr
            ratio += delta
            dsic = (1.0-ratio**n)/(1.0-ratio)
            Rc = dsit - dsic
            count += 1
            if display:
                print(count, ratio, delta)
            if count == 100:
                print('Convergence failed in geometric_series_spacing_ds0.')
                break
    ds = asarray([ds0t*ratio**i for i in range(num)])
    spc = zeros(num+1)
    spc[1:] = cumsum(ds)
    return spc
