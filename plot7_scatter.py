import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
mp.rc('font', size=14)
mp.rcParams['lines.markersize'] = 3
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import system

path = '/data/1/tomaschor/LES05/{}'
cmap="viridis"
z0=[-.8, -.5, -.3, -.2, -.1, 0]
depth=-1.6

grid="allcoarse"
grid="coriolis"
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
labels = ["d7_m400", "d11_m580"]
for label in labels:
    system(f"mkdir -p figures_{label}")


for nn, name in enumerate(names):
    print(name)
    label = labels[nn]

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))
    #-----

    #----
    # Open dataset
    ds = xr.open_dataset(f"data/jpdfs_{name}.nc")
    ddvs = xr.open_dataset(f"sweep/data/ddvs_{name}_{label}.nc")

    T_conv = ddvs.T_conv

    ds = ds.assign_coords(zeta=ds.zeta*ddvs.T_conv,
                          zeta_filt=ds.zeta_filt*ddvs.T_conv,
                          hdiv=ds.hdiv*ddvs.T_conv,
                          hdiv_filt=ds.hdiv_filt*ddvs.T_conv)
    ddvs["ζ"] = ddvs.ζ*ddvs.T_conv
    ddvs["ζ_filt"] = ddvs.ζ_filt*ddvs.T_conv
    ddvs["hdiv"] = ddvs.hdiv*ddvs.T_conv
    ddvs["hdiv_filt"] = ddvs.hdiv_filt*ddvs.T_conv

    p90 = np.percentile(ddvs.mass, 90)
    ddvs90 = ddvs.where(ddvs.mass>=p90, drop=True)
    #----

   
    #----
    faux2, axes = plt.subplots(ncols=3, figsize=(15,5))
    axes[0].scatter(ddvs.hdiv, abs(ddvs.ζ), label="", color="k")
    axes[1].scatter(ddvs.hdiv_filt, abs(ddvs.ζ), label="", color="k")
    axes[2].scatter(ddvs.ζ_filt, ddvs.ζ, label="", color="k")

    axes[0].set_xlabel("Normalized horizontal divergence")
    axes[1].set_xlabel("Normalized filtered horizontal divergence")
    axes[2].set_xlabel("Normalized filtered vertical vorticity")

    axes[0].set_ylabel("Abs. value of normalized vertical vorticity")
    axes[1].set_ylabel("Abs. value of normalized vertical vorticity")
    axes[2].set_ylabel("Normalized vertical vorticity")

    axes[0].set_xlim(-80, 80)
    axes[1].set_xlim(-8, 8)
    axes[2].set_xlim(-8, 8)
    axes[0].set_ylim(-50, 500)
    axes[1].set_ylim(-50, 500)
    axes[2].set_ylim(-600, 600)
    #----

    #----
    for ax in axes:
        ax.grid(True)
        ax.axvline(x=0, color="k", lw=2)
        ax.axhline(y=0, color="k", lw=2)
        ax.xaxis.set_major_locator(MaxNLocator(symmetric=True))
    faux2.tight_layout()
    faux2.savefig(f"figures_{label}/scatter_vort_{name}_{label}.pdf", pad_inches=0,  bbox_inches="tight")
    #----
