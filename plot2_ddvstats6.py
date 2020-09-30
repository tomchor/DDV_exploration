import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import lespy as lp
from scipy.stats import norm
from aux_ddv import snames
import matplotlib as mp
mp.rc('font', size=14)
from scipy.stats import ks_2samp
from os import system

path = '/data/1/tomaschor/LES05/{}'

grid = "fine"
grid = "allcoarse"
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
    names=["conv_coarse", "conv_cbig2", "conv_csmall"]
labels = ["d7_m400", "d11_m580", "d5_m460"]
for label in labels:
    system(f"mkdir -p figures_{label}")

fig, axes = plt.subplots(nrows=1, ncols=len(names), figsize=(len(names)*3, len(names)), sharex=True, sharey=True)
axes = axes.flatten()
fig2, axes2 = plt.subplots(nrows=1, ncols=len(names), figsize=(len(names)*3, len(names)), sharex=True, sharey=True)
axes2 = axes2.flatten()
for nn, name in enumerate(names):
    print(name)
    label = labels[nn]

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))
    #-----

    w_star = lp.physics.w_star(sim)
    T_conv = sim.inv_depth/w_star

    #-----
    ds = xr.open_dataset(f"sweep/data/ddvs_{name}_{label}.nc")
    #-----

    #-----
    ds_pos = ds.where(ds.ζ>0, drop=True)
    ds_neg = ds.where(ds.ζ<0, drop=True)
    #-----

    #-----
    ζ_pos = ds_pos.ζ * ds.T_conv
    ζ_neg = ds_neg.ζ * ds.T_conv

    print(name)
    print(f"ζ_pos length: {ζ_pos.shape}")
    print(f"ζ_neg length: {ζ_neg.shape}")
    try:
        Dn, pval = ks_2samp(abs(ζ_neg), ζ_pos)
        c005=1.224; n=len(ζ_neg); m=len(ζ_pos)
        print(f"Result from Kolgorov-Smirnof test:\n{ks_2samp(abs(ζ_neg), ζ_pos)}")
        print(f"At level 5%, are the distributions different?", Dn>1.224*np.sqrt((n+m)/(n*m)))
    except ZeroDivisionError:
        pass
    print()
    #-----

    #-----
    # PLOT VORTICITY ζ
    #-----

    #-----
    # Get histogram
    histζ_pos, ζed_pos = np.histogram(ζ_pos, bins=50)
    histζ_neg, ζed_neg = np.histogram(ζ_neg, bins=50)

    ζc_pos = (ζed_pos[:-1] + ζed_pos[1:])/2
    ζc_neg = (ζed_neg[:-1] + ζed_neg[1:])/2
    #-----

    #-----
    # Normalize to get histogram per snapshot
    Nt = len(np.unique(ds.itime))
    histζ_pos = histζ_pos/Nt
    histζ_neg = histζ_neg/Nt
    #-----

    #-----
    # Plot histograms
    axes[nn].bar(ζc_pos, height=histζ_pos, width=np.diff(ζed_pos)[0], 
                 color="royalblue", label=f"Pos.: {len(ζ_pos)}")
    axes[nn].bar(ζc_neg, height=histζ_neg, width=np.diff(ζed_neg)[0], 
                 color="darkorange", label=f"Neg.: {len(ζ_neg)}")
    #-----

    #-----
    axes[nn].set_title(f"Simulation: {snames[name]}\nTotal occurrences: {len(ds.ζ)}")
    axes[nn].set_xlabel(r"$\zeta \times T_*$")
    #-----

    #-----
    # PLOT SIZE
    #-----

    #-----
    # Get histogram
    histS_pos, Sed_pos = np.histogram(ds_pos.size*sim.domain.dx*2, bins=50, density=True)
    histS_neg, Sed_neg = np.histogram(ds_neg.size*sim.domain.dx*2, bins=50, density=True)

    Sc_pos = (Sed_pos[:-1] + Sed_pos[1:])/2
    Sc_neg = (Sed_neg[:-1] + Sed_neg[1:])/2
    #-----

    #-----
    # Normalize to get histogram per snapshot
    Nt = len(np.unique(ds.itime))
    histS_pos = histS_pos/Nt
    histS_neg = histS_neg/Nt
    #-----

    #-----
    # Plot histograms
    axes2[nn].bar(Sc_pos, height=histS_pos, width=np.diff(Sed_pos)[0], 
                 color="royalblue", label=f"Pos.", alpha=0.6)
    axes2[nn].bar(Sc_neg, height=histS_neg, width=np.diff(Sed_neg)[0], 
                 color="darkorange", label=f"Neg.", alpha=0.6)
    #-----

    #-----
    axes2[nn].set_title(f"Simulation: {snames[name]}\nOccurrences: {len(ds.ζ)}")
    axes2[nn].set_xlabel(r"Diameter (m)")
    #-----
 

for ax in axes.flatten():
    ax.grid()
    ax.set_xlim(-500, 500)
    ax.legend(loc=2, prop={'size': 10})
fig.tight_layout()
fig.savefig(f"figures_{label}/DDV_hist_{grid}_{label}.pdf", pad_inches=0, bbox_inches="tight")


for ax in axes2.flatten():
    ax.grid()
    ax.set_xlim(0, 5)
    ax.legend(loc=2, prop={'size': 10})
fig2.tight_layout()
fig2.savefig(f"figures_{label}/size_hist_{grid}_{label}.pdf", pad_inches=0, bbox_inches="tight")


