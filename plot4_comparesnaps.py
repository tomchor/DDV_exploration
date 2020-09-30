import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
from aux_ddv import skewness, snames
mp.rc('font', size=12)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

path = '/data/1/tomaschor/LES05/{}'

grid = "fine"
grid = "allcoarse"
grid = "coriolis"
grid = "resolution"

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse", "conv_negcoarse"]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]
elif grid=="resolution":
    names=["conv_coarse", "conv_cbig2"]


z0 = -1.6
fig, axes = plt.subplots(ncols=2, nrows=int(np.round(len(names)/2)), 
                         figsize=(12, int(np.round(len(names)/2))*5), sharex=True, sharey=True)
axes=axes.flatten()
for nn, name in enumerate(names):
    print(name)

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        lim=150
    else:
        sim = lp.Simulation(path.format(name))
        lim=100
    #-----

    T_conv = sim.z_i / sim.w_star

    #----
    ds=xr.open_dataset(f"data/vort_{name}.nc")
    ds=ds.isel(itime=-1, drop=True).sel(z=z0, method="nearest")
    #----

    #----
    ζ_norm = ds.ζ*T_conv
    ζ_norm.attrs = dict(long_name="Normalized vorticity ($ζ/T_*$)")
    ζ_norm.plot.imshow(ax=axes[nn], y="y", vmin=-lim, vmax=lim, cmap="RdBu_r", rasterized=True, 
                        interpolation="bicubic", cbar_kwargs=dict(shrink=0.5))
    #----

    #----
    axes[nn].set_title(snames[name])
    #----

#----
for ax in axes.flatten():
    ax.grid()
    ax.set_aspect(1)
fig.tight_layout()
fig.savefig(f"figures_check/snapcompare_{grid}_z={z0}.pdf")
#----



