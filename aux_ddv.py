import xarray as xr
import numpy as np


def moments(pdf, dim, nan_policty="omit"):
    """ Calculates the 2nd, 3rd and 4th moments """
    mean = (pdf.coords[dim]*pdf).integrate(dim)
    var = ((pdf.coords[dim]-mean)**2*pdf).integrate(dim)
    std = np.sqrt(var)

    skew = (((pdf.coords[dim] - mean)/std)**3*pdf).integrate(dim)
    kurt = (((pdf.coords[dim] - mean)/std)**4*pdf).integrate(dim)
    return var, skew, kurt


def skewness(pdf, dim, nan_policty="omit"):
    mean = (pdf.coords[dim]*pdf).integrate(dim)
    var = ((pdf.coords[dim]-mean)**2*pdf).integrate(dim)
    std = np.sqrt(var)
    skew = (((pdf.coords[dim] - mean)/std)**3*pdf).integrate(dim)
    return skew



def swap_dims(da, sdims):
    dims = list(da.dims)
    a, b = dims.index(sdims[0]), dims.index(sdims[1])
    dims[b], dims[a] = dims[a], dims[b]
    return xr.DataArray(da, dims=dims, coords=dict(da.coords))




def conv_xr(da, axes=("z", "itime"), kernel=np.ones((3,3))):
    """ Applies convolve over consecutive planes in axis z """
    from scipy.ndimage import convolve
    conv1 = lambda x: convolve(x, kernel, mode="wrap")

    planes = [ [ xr.apply_ufunc(conv1, da.loc[{axes[0]:p}].loc[{axes[1]:q}]) for p in da.coords[axes[0]] ] for q in da.coords[axes[1]] ]
    new = xr.concat([ xr.concat(plane, dim=axes[0]) for plane in planes ], dim=axes[1])
    return new.transpose(*da.dims)

def ks_test(d1, d2, α=0.1):
    """ 
    Applies Kolmogorov-Smirnoff test
    Returns True if distribuations are the same (cannot reject hypothesis) 
    """
    from scipy import stats

    c_α = lambda α: np.sqrt(-np.log(α)/2)
    n1=len(d1)
    n2=len(d2)
    D, p_val = stats.ks_2samp(d1, d2)
    print(f"KS statistic = {D}, p-value = {p_val}")
    if D <= c_α(α) * np.sqrt((n1+n2)/(n1*n2)):
        return True
    else:
        return False


def gaussian_conv(da, delta=1, dims=["x", "y"], truncate=4, how="manual", 
        full_output=False, **kwargs):
    """ Applies convolution with gaussian kernel """
    from scipy.ndimage import convolve1d
    from scipy.ndimage.filters import gaussian_filter1d

    da = da.copy(deep=True)
    for dim in dims:
        axis = da.dims.index(dim)
        if dim=="z":
            bc="constant"
        else:
            bc="wrap"

        #----
        # Calculate kernel resolution
        s = da.coords[dim]
        ds = s.diff(dim)[0].item()
        #----
        if how=="manual":
            r = np.arange(0, delta*truncate+ds, ds)
            r = np.sort(np.concatenate([-r[1:], r]))

            # calculate 1d kernel
            G = (1/(np.sqrt(2*np.pi)*delta)) * np.exp(-1/2*(r/delta)**2)

            da = xr.apply_ufunc(convolve1d, da, G, kwargs=dict(axis=axis, mode=bc, cval=0))

        elif how=="auto":
            da = xr.apply_ufunc(gaussian_filter1d, da, kwargs=dict(axis=axis, mode=bc, cval=0, sigma=delta/ds, truncate=truncate), **kwargs)

    if full_output and how=="manual":
        return r, G, da
    else:
        return da


def upodd(f, asint=True):
    if asint:
        return int(np.ceil(f) // 2 * 2 + 1)
    else:
        return np.ceil(f) // 2 * 2 + 1

snames = dict(conv_coarse="FS-R",
              conv_atcoarse="NS-R",
              conv_fine="FS-R-F",
              conv_atfine="NS-R-F",
              conv_nccoarse="FS-NR",
              conv_ncatcoarse="NS-NR",
              conv_negcoarse="FS-CR",
              conv_cbig2="FS-R-L",
              conv_csmall="FS-R-S",
              )
pnames = dict(conv_coarse="Free-slip, coarse",
              conv_atcoarse="No-slip, coarse",
              conv_fine="Free-slip, fine",
              conv_atfine="No-slip, fine",
              conv_nccoarse="Free-slip, coarse, $f=0$",
              conv_negcoarse="Free-slip, coarse, $f<0$",
              )
