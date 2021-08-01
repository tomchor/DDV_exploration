import numpy as np
import lespy as lp
import xarray as xr
from dask.diagnostics import ProgressBar

path = '/data/1/tomaschor/LES05/{}'
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse", "conv_ncatcoarse", "conv_cbig2", "conv_negcoarse"]


Δ = 15 # std for gaussian filter (meters)
kernel="gaussian"
for nn, name in enumerate(names):
    #+++++
    print(name)
    ds = xr.open_dataset(f'data/vort_{name}.nc', chunks=dict(itime=1))
    #-----

    #+++++ Change behavior if it's Chao's code
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output')
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
    #-----


    #+++++
    # Adjustments
    w_star = lp.physics.w_star(sim)
    T_conv = sim.inv_depth/w_star

    if 0:
        levs = lp.utils.nearest(ds.z, -sim.z_i*np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1/20, 0]))
        ds = ds.sel(z=levs)
    #-----

    #++++ Filters fields with a Gaussian convolution
    print("convolving ζ")
    if 0: # This filters ζ only where w is negative
        ζx_int = ds.ζ.where(ds.w>0, np.nan).interpolate_na("x", period=sim.domain.lx)
        ζy_int = ds.ζ.where(ds.w>0, np.nan).interpolate_na("y", period=sim.domain.ly)
        ζ_to_filter = (ζx_int + ζy_int)/2
    else:
        ζ_to_filter = ds.ζ
    ζ_filt = lp.vector.gaussian_conv(ζ_to_filter, δ=Δ, dims=["x", "y"], truncate=3, how="auto")

    print("convolving w")
    w_filt = lp.vector.gaussian_conv(ds.w, δ=Δ, dims=["x", "y"], truncate=3, how="auto")

    print("convolving ∇•u")
    hdiv_filt = lp.vector.gaussian_conv(ds.hdiv, δ=Δ, dims=["x", "y"], truncate=3, how="auto")

    ds_del = xr.Dataset(dict(w=w_filt, hdiv=hdiv_filt, ζ=ζ_filt))
    ds_del.attrs = dict(T_conv=T_conv, Δ=Δ)

    outname = f"data/surf_filtered_{name}.nc"
    delayed_nc = ds_del.to_netcdf(outname, compute=False)
    with ProgressBar():
        results = delayed_nc.compute()
    print(f"Done saving to {outname}")
    #-----

