import numpy as np
import lespy as lp
import xarray as xr
from matplotlib import pyplot as plt
from aux_ddv import snames
from aux_plot import cm
import matplotlib as mp
mp.rc('font', size=13)

path = '/data/1/tomaschor/LES05/{}'
cmap = "RdBu_r"
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


#-----
# create limits
vlim=100
wlim=2
#-----


for nn, name in enumerate(names):
    print(name)
    #----
    # Preamble
    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))
    #-----

    w_star = lp.physics.w_star(sim)
    T_conv = sim.inv_depth/w_star
    #----

    #----
    # Open datasets
    ds = xr.open_dataset(f'data/vort_{name}.nc')
    ds_filt = xr.open_dataset(f"data/surf_filtered_{name}.nc")
    #----

    #----
    # Select surface
    ds = ds.sel(z=depth, itime=np.inf, method="nearest")
    ds_filt = ds_filt.sel(z=depth, itime=np.inf, method="nearest")
    #----

    #----
    ζ = ds.ζ*T_conv
    ζ.attrs = dict(long_name=r"Normalized vorticity ($\zeta\times T_*$)")
    #----

    #----
    fig, axes = plt.subplots(ncols=2, figsize=(16,8))
    ζ.plot.imshow(ax=axes[0], x="x", vmin=-vlim, vmax=vlim, 
                       interpolation="bicubic", cmap=cmap, rasterized=True, cbar_kwargs=dict(pad=1e-2, shrink=0.7))
    ζ.plot.imshow(ax=axes[1], x="x", vmin=-vlim, vmax=vlim, 
                       interpolation="bicubic", cmap=cmap, rasterized=True, cbar_kwargs=dict(pad=1e-2, shrink=0.7))

    ds_filt.hdiv.plot.contourf(x="x", levels=[-1e3, 0., 1e3], ax=axes[0],
                                   linestyles="dashed", cmap=cm, add_colorbar=False)
    ds.hdiv.plot.contourf(x="x", levels=[-1e3, 0., 1e3], ax=axes[1],
                                   linestyles="dashed", cmap=cm, add_colorbar=False)

    #img=ds_filt.hdiv.plot.contour(x="x", levels=[0.], ax=axes[0], colors="0.2", linestyles="dashed", linewidths=0.5)
    #axes[0].clabel(img, colors="0.2", fmt='%1.1f')

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_title(snames[name])

    plt.tight_layout()
    plt.savefig("propfigs/vort_hdiv_{}.pdf".format(name), bbox_inches="tight", pad_inches=0)
    plt.close()
    #----


