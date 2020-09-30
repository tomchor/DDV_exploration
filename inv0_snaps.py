import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
from aux_ddv import moments
mp.rc('font', size=12)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from aux_plot import latex_float, set_share_axes
from matplotlib.patches import Circle
from os import system

path = '/data/1/tomaschor/LES05/{}'
grid="allcoarse"
cmap="viridis"
z0=[-.8, -.5, -.3, -.2, -.1, 0]
depth=-1.6

label="d7_m260"
system(f"mkdir -p figures_check_{label}")

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse",]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]

for nn, name in enumerate(names):
    print(name)
    sim = lp.Simulation(path.format(name))
    out = lp.Output(path.format(name)+'/output')

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
    jpdfs = xr.open_dataset(f"data/jpdfs_{name}.nc")
    ddvs = xr.open_dataset(f"sweep/data/ddvs_{name}_{label}.nc")
    snaps = xr.open_dataset(f"data/vort_{name}.nc")
    filtsnaps = xr.open_dataset(f"data/surf_filtered_{name}.nc")

    T_conv = ddvs.T_conv

    jpdfs = jpdfs.assign_coords(zeta=jpdfs.zeta*ddvs.T_conv,
                                zeta_filt=jpdfs.zeta_filt*ddvs.T_conv,
                                hdiv=jpdfs.hdiv*ddvs.T_conv,
                                hdiv_filt=jpdfs.hdiv_filt*ddvs.T_conv)

    snaps["hdiv"] = snaps.hdiv * T_conv
    snaps["ζ"] = snaps.ζ * T_conv

    filtsnaps["hdiv"] = filtsnaps.hdiv * T_conv
    filtsnaps["ζ"] = filtsnaps.ζ * T_conv

    ddvs["ζ"] = ddvs.ζ*ddvs.T_conv
    ddvs["ζ_filt"] = ddvs.ζ_filt*ddvs.T_conv
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
    # Now we start plotting JPDFs
    #----

    #----
    # Pad the zero values with something else
    jpdf_cross = jpdfs.sel(z=depth/sim.z_i, method="nearest").jpdf_cross
    jpdf_cross = jpdf_cross.where(jpdf_cross>1e-10, 1e-10)

    jpdf_cross.attrs = dict(long_name="Joint PDF of hor. div. and filtered hor. div.")
    jpdf_cross.hdiv.attrs = dict(long_name="Normalized Horizontal divergence")
    jpdf_cross.hdiv_filt.attrs = dict(long_name="Filtered normalized Horizontal divergence")
    #----

    #----
    # Plot cross JPDF
    itime = snaps.itime[-1]
    iddvs = ddvs.where(ddvs.itime==itime, drop=True)
    #iddvs = ddvs.where((ddvs.itime==itime) & (ddvs.hdiv>0), drop=True)
    isnaps = snaps.sel(itime=itime, z=depth, method="nearest")
    ifiltsnaps = filtsnaps.sel(itime=itime, z=depth, method="nearest")

    faux, axes = plt.subplots(ncols=2, nrows=3, figsize=(16,21))
    axes = axes.flatten()
    set_share_axes(axes[:4], sharex=True, sharey=True)

    jpdf_cross.plot.imshow(ax=axes[5], x="hdiv", norm=LogNorm(), rasterized=True,
                    xlim=[-amp_d, amp_d], ylim=[-amp_d_filt, amp_d_filt], 
                    cmap=cmap, vmin=1e1, vmax=1e5,
                    interpolation="bicubic")
    cs=jpdf_cross.plot.contour(ax=axes[5], x="hdiv", levels=[1e3, 1e4], 
                               colors="0.8", linestyles="dashed", alpha=1,
                               linewidths=3)
    #----

    #----
    # Plot snapshots
    axes[5].axvline(x=0, color="k", lw=2); axes[5].axhline(y=0, color="k", lw=2)
    axes[5].plot(iddvs.hdiv, iddvs.hdiv_filt, ".", color="dimgray")
    #----


    #----
    # Plot DDVs on top
    isnaps.hdiv.plot.imshow(ax=axes[0], y="y", vmin=-75, vmax=75, cmap="RdBu_r")
    ifiltsnaps.hdiv.plot.imshow(ax=axes[1], y="y",)
    isnaps.ζ.plot.imshow(ax=axes[2], y="y", vmin=-100, vmax=100, cmap="RdBu_r")
    ifiltsnaps.ζ.plot.imshow(ax=axes[3], y="y", vmin=-3, vmax=3, cmap="RdBu_r", cbar_kwargs=dict(label=r"$\zeta\times T_*$"))
    (ifiltsnaps/T_conv/sim.freq_coriolis).ζ.plot.imshow(ax=axes[4], y="y", vmin=-5, vmax=5, cmap="RdBu_r", cbar_kwargs=dict(label=r"$\zeta/f$"))


    for ax in axes[:4]:
        ax.set_aspect("equal")
        for d in iddvs.ddv:
            ddv = iddvs.sel(ddv=d)
            circle = Circle((ddv.X, ddv.Y), ddv.size*sim.domain.dx*2, facecolor="none", edgecolor="k")
            ax.add_artist(circle)
    #----

    faux.tight_layout()
    faux.savefig(f"figures_check_{label}/investigate_{itime.item()}_{name}.pdf", pad_inches=0, bbox_inches="tight")
    #----

    
