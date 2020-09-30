import numpy as np
import lespy as lp
import xarray as xr
from matplotlib import pyplot as plt
from dask.distributed import Client


path = '/data/1/tomaschor/LES05/{}'
grid = "neg"
grid = "resolution"

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse",]
elif grid=="neg":
    names=["conv_coarse", "conv_negcoarse",]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]
elif grid=="resolution":
    names=["conv_cbig2"]

#client = Client(n_workers=10, threads_per_worker=2, memory_limit='8GB')
#exit()

for nn, name in enumerate(names):
    print(name)

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))
    #-----

    #----
    # Adjust z coordinate
    ds = xr.open_dataset('data/vort_{}.nc'.format(name), chunks=dict(itime=2))
#    ds = xr.open_dataset('data/vort_{}.nc'.format(name))

    ds = ds.sel(z=-1.6, method="nearest")
    #----

    #----
    ζ_mean = ds.ζ.mean(("y", "x"))
    ζ_std = ds.ζ.std(("x", "y"))
    ζ_skew = ( ( (ds.ζ - ζ_mean) / ζ_std )**3 ).mean(("x", "y"))
    #----

    #----
    fig, ax1 = plt.subplots()
    c="darkblue"
    ζ_skew.plot(ax=ax1, x="itime", label="Skewness", color=c)
    ax1.tick_params(axis='y', labelcolor=c)
    ax1.set_ylabel("Skewness")

    c="darkorange"
    ax2 = ax1.twinx()
    ζ_std.plot(ax=ax2, x="itime", label="STD", color=c)
    ax2.tick_params(axis='y', labelcolor=c)
    ax2.set_ylabel("STD (1/s)")

    for ax in fig.axes:
        ax.legend()
        ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(f"figures_check/skew_evol.png")
    #----

