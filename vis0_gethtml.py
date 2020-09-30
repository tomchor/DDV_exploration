import numpy as np
import lespy as lp
import xarray as xr
#import matplotlib as mp
#mp.rc('font', size=20)

path = '/data/1/tomaschor/LES05/{}'
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine",]
names=["conv_coarse", "conv_nccoarse"]
vlim = 100
cmap="seismic"

Nt = 5
for nn, name in enumerate(names):
    print(name)
    sim = lp.Simulation(path.format(name))
    out = lp.Output(path.format(name)+'/output')

    ds = xr.open_dataset('data/vort_{}.nc'.format(name))
    ds_filt = xr.open_dataset('data/surf_filtered_{}.nc'.format(name))

    #----
    # Restrict size of data
    ds = ds.isel(itime=slice(-Nt, None))
    ds_filt = ds_filt.isel(itime=slice(-Nt, None))
    ds = ds.sel(z=ds_filt.z)
    #----

    #----
    T_conv = sim.inv_depth/lp.physics.w_star(sim)
    vclim=(-vlim, vlim)
    vclim_filt=(-vlim/10, vlim/10)
    #----
    
    #----
    # From simulation time to hours
    if 0:
        vort.coords["itime"] = vort.itime*sim.dt/60/60
        vort = vort.rename(dict(itime="hours"))
        vort_filt.coords["itime"] = vort_filt.itime*sim.dt/60/60
        vort_filt = vort_filt.rename(dict(itime="hours"))
    #----
    exit()

    lp.plot2.animate_hv((vort*T_conv), simulation=sim, saveas=f"htmls/vort_{name}.html",
            clabel=r"$\zeta \; |h| / w_*$", clim=vclim, cmap=cmap, fps=15, dpi=120, interpolation="none")

