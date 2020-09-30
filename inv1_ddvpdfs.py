import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import lespy as lp
from aux_ddv import snames
import matplotlib as mp
mp.rc('font', size=12)
from os import system

path = '/data/1/tomaschor/LES05/{}'

grid = "fine"
grid = "allcoarse"

label="d7_m260"
system(f"mkdir -p figures_{label}")

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse",]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]

fig, axes = plt.subplots(ncols=len(names), figsize=(len(names)*3, len(names)), sharex=True, sharey=True)
axes = axes.flatten()
for nn, name in enumerate(names):
    sim = lp.Simulation(path.format(name))
    out = lp.Output(path.format(name)+'/output')
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
        print(f"At level 0.1%, are the distributions the same?", ks_test(abs(ζ_neg), ζ_pos, α=0.001))
        print(f"At level 1%, are the distributions the same?", ks_test(abs(ζ_neg), ζ_pos, α=0.01))
        print(f"At level 10%, are the distributions the same?", ks_test(abs(ζ_neg), ζ_pos, α=0.05))
    except ZeroDivisionError:
        pass
    print()
    #-----

    #-----
    # PLOT VORTICITY ζ
    #-----

    #-----
    # Get histogram
    histζ_pos, ζed_pos = np.histogram(ζ_pos, bins=50, density=True)
    histζ_neg, ζed_neg = np.histogram(-ζ_neg, bins=50, density=True)

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
                 color="royalblue", label=f"Pos.: {len(ζ_pos)}", alpha=0.5)
    axes[nn].bar(ζc_neg, height=histζ_neg, width=np.diff(ζed_neg)[0], 
                 color="darkorange", label=f"Neg.: {len(ζ_neg)}", alpha=0.5)
    #-----

    #-----
    axes[nn].set_title(f"Simulation: {snames[name]}\nOccurrences: {len(ds.ζ)}")
    axes[nn].set_xlabel(r"$|\zeta| \times T_*$")
    #-----


for ax in axes.flatten():
    ax.grid()
    ax.set_xlim(-500, 500)
    ax.legend(loc=2, prop={'size': 8})
fig.tight_layout()
fig.savefig(f"figures_check_{label}/DDV_pdf_{grid}_{label}.pdf", pad_inches=0, bbox_inches="tight")


