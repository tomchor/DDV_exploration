import numpy as np
import lespy as lp
import xarray as xr
import matplotlib as mp
from aux_ddv import moments, snames
mp.rc('font', size=14)
import matplotlib.pyplot as plt
from os import system

path = '/data/1/tomaschor/LES05/{}'
cmap="viridis"
z0=[-.8, -.5, -.3, -.2, -.1, 0]
depth=-1.6

grid="allcoarse"
grid="coriolis"

label="d7_m260"
system(f"mkdir -p figures_{label}")

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse", "conv_negcoarse"]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]


fig, axes = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=True, figsize=(12, 5))
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
    ds = xr.open_dataset(f"data/jpdfs_{name}.nc")
    ddvs = xr.open_dataset(f"sweep/data/ddvs_{name}_{label}.nc")

    T_conv = ddvs.T_conv
    f0 = sim.freq_coriolis * T_conv

    ds = ds.assign_coords(zeta=ds.zeta*T_conv,
                          zeta_filt=ds.zeta_filt*T_conv,
                          hdiv=ds.hdiv*T_conv,
                          hdiv_filt=ds.hdiv_filt*T_conv)
    ds = ds/T_conv**2
    #----

    #----
    # Plot PDFs
    pdf_ζ = ds.sel(z=depth/sim.z_i, method="nearest").jpdf.integrate("hdiv")
    cdf_ζ = pdf_ζ.cumsum("zeta") * (pdf_ζ.zeta.diff("zeta")[0])

    pdf_ζ.attrs = dict(long_name="PDF of vorticity")
    cdf_ζ.attrs = dict(long_name="CDF of vorticity")
    #----

    #----
    # Plot original CDF
    cdf_ζ.plot(ax=axes[0,0], x="zeta", label=snames[name])
    pdf_ζ.plot(ax=axes[1,0], x="zeta", label=snames[name], yscale="log")
    #----

    #----
    # Plot modified CDF
    cdf_ζ.assign_coords(zeta=cdf_ζ.zeta+f0).plot(ax=axes[0,1], x="zeta", label=snames[name])
    pdf_ζ.assign_coords(zeta=cdf_ζ.zeta+f0).plot(ax=axes[1,1], x="zeta", label=snames[name], yscale="log")
    #----

axes[0,0].set_title("Original results")
axes[0,1].set_title("Results including $f$")

for ax in axes[:,1]:
    ax.set_xlabel(r"$\zeta+f$")
for ax in axes[:,0]:
    ax.set_xlabel(r"$\zeta$")

for i, ax in enumerate(axes.flatten()):
    ax.grid()
    ax.legend()
    ax.set_xlim(-200,200)
fig.tight_layout()

fig.savefig(f"figures_check/cdf_zeta_{grid}_{label}.pdf")
