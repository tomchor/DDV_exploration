import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
from aux_ddv import moments
mp.rc('font', size=14)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from aux_plot import latex_float
from os import system

path = '/data/1/tomaschor/LES05/{}'
cmap="viridis"
z0=[-.8, -.5, -.3, -.2, -.1, 0]
depth=-1.6

grid="allcoarse"
#grid="coriolis"
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
    names=["conv_coarse", "conv_cbig2", "conv_csmall"]
labels = ["d7_m400", "d11_m580"]
for label in labels:
    system(f"mkdir -p figures_{label}")


fig,        axes        = plt.subplots(nrows=int(np.round(len(names)/2)), ncols=2, sharey=True, sharex=True, figsize=(12, len(names)//2*6))
fig_filt,   axes_filt   = plt.subplots(nrows=int(np.round(len(names)/2)), ncols=2, sharey=True, sharex=True, figsize=(12, len(names)//2*6))
axes=axes.flatten()
axes_filt=axes_filt.flatten()
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
    # Define plot limits
    if "coarse" in name:
        amp_ζ = 8e-2
        amp_ζ_filt = 1e-3
        amp_d = 100 # 3e-2
        amp_d_filt = 6 # 1.8e-3

        ζ_lim_filt = [-amp_ζ_filt, 2*amp_ζ_filt]

    elif "fine" in name:
        amp_ζ = 1e-1
        amp_ζ_filt = 5e-4
        amp_d = 5e-2
        amp_d_filt = 1.5e-3

        ζ_lim_filt = [-amp_ζ_filt, amp_ζ_filt]
    #----

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
    ddvs["hdiv"] = ddvs.hdiv*ddvs.T_conv
    ddvs["hdiv_filt"] = ddvs.hdiv_filt*ddvs.T_conv

    p90 = np.percentile(ddvs.mass, 90)
    ddvs90 = ddvs.where(ddvs.mass>=p90, drop=True)
    #----

    #----
    ddvs.ζ.attrs = dict(long_name=r"Normalized vorticity ($\zeta \times T_*$)")
    ddvs.hdiv.attrs = dict(long_name=r"Normalized divergence ($\nabla\cdot u_h \times T_*$)")
    ddvs.hdiv_filt.attrs = dict(long_name=r"Normalized filtered divergence ($\nabla\cdot \widetilde{u_h} \times T_*$)")
    #----

    #----
    # Plot PDFs
    pdf_ζ = ds.sel(z=z0, method="nearest").jpdf.integrate("hdiv")
    pdf_ζ_filt = ds.sel(z=z0, method="nearest").jpdf_filt.integrate("hdiv_filt")

    pdf_ζ.attrs = dict(long_name="PDF of vorticity")
    pdf_ζ_filt.attrs = dict(long_name="PDF of filtered vorticity")

    pdf_ζ.sel(z=slice(-.85, 0)).plot.line(x="zeta", yscale="log", ax=axes[nn])
    pdf_ζ_filt.sel(z=slice(-.85, 0)).plot.line(x="zeta_filt", yscale="log", ax=axes_filt[nn])
    #----


    #----
    axes[nn].set_title(name)
    axes_filt[nn].set_title(name)
    #----

    #----
    # Now we start plotting JPDFs
    #----

    #----
    # Pad the zero values with something else
    jpdf_cross = ds.sel(z=depth/sim.z_i, method="nearest").jpdf_cross
    jpdf_cross = jpdf_cross.where(jpdf_cross>1e-10, 1e-10)

    jpdf_cross.attrs = dict(long_name="Joint PDF of hor. div. and filtered hor. div.")
    jpdf_cross.hdiv.attrs = dict(long_name="Normalized Horizontal divergence")
    jpdf_cross.hdiv_filt.attrs = dict(long_name="Filtered normalized Horizontal divergence")
    #----

    #----
    # Cross
    faux, ax = plt.subplots(figsize=(10,6))

    jpdf_cross.plot.imshow(ax=ax, x="hdiv", norm=LogNorm(), rasterized=True,
                    xlim=[-amp_d, amp_d], ylim=[-amp_d_filt, amp_d_filt], 
                    cmap=cmap, vmin=1e1, vmax=1e5,
                    interpolation="bicubic")
    cs=jpdf_cross.plot.contour(x="hdiv", levels=[1e3, 1e4], 
                               colors="0.8", linestyles="dashed", alpha=1,
                               linewidths=3)

    ax.axvline(x=0, color="k", lw=2); ax.axhline(y=0, color="k", lw=2)
    ax.clabel(cs, fmt=latex_float)
    ax.plot(ddvs.hdiv, ddvs.hdiv_filt, ".", color="dimgray")
    #ax.plot(ddvs90.hdiv, ddvs90.hdiv_filt, ".", color="red")

    ax.set_title("")
    faux.tight_layout()
    faux.savefig(f"figures_{label}/jpdf_cross_{name}_{label}.pdf", pad_inches=0, bbox_inches="tight")
    #----

    
for i, ax in enumerate(axes):
    ax.grid()
    try:
        if i != 1: ax.get_legend().remove()
    except:
        pass
fig.tight_layout()


for i, ax in enumerate(axes_filt):
    ax.grid()
    try:
        if i != 1: ax.get_legend().remove()
    except:
        pass
fig_filt.tight_layout()


fig.savefig(f"figures_{label}/pdf_zeta_{grid}_{label}.pdf")
fig_filt.savefig(f"figures_{label}/pdf_zeta_filt_{grid}_{label}.pdf")
