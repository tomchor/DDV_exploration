import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
from aux_ddv import skewness
mp.rc('font', size=12)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

path = '/data/1/tomaschor/LES05/{}'
grid="coriolis"
grid="resolution"


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
for nn, name in enumerate(names):
    print(name)

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))
    #-----

    T_conv = sim.z_i / sim.w_star

    #----
    ds=xr.open_dataset(f"data/vort_{name}.nc")
    ds=ds.isel(itime=-1, drop=True).sel(z=z0, method="nearest")
    #----

    #----
    ds.θ.attrs = dict(long_name=r"Temperature", units="K")
    θ_p = ds.θ - ds.θ.mean()
    θ_p.attrs = dict(long_name=r"Temperature fluctuations", units="K")
    #----

    #----
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)

    (ds.hdiv*T_conv).plot(ax=axes[0,0], y="y", rasterized=True)
    (ds.ζ*T_conv).plot(ax=axes[0,1], y="y", vmin=-200, vmax=200, cmap="RdBu_r", rasterized=True)
    ds.θ.plot(ax=axes[1,0], y="y", rasterized=True)
    θ_p.plot(ax=axes[1,1], y="y", rasterized=True)
    #----

    #----
#    for ax in axes.flatten():
#        ax.grid()
    fig.tight_layout()
    fig.savefig(f"figures_check/snapshots_{name}_z={z0}.pdf")
    #----

