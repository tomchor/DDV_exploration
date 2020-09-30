import numpy as np
import lespy as lp
import xarray as xr

path = '/data/1/tomaschor/LES05/{}'
names = ["conv_coarse", "conv_atcoarse"]
names = ["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_atcoarse", "conv_fine", "conv_atfine", "conv_ncatcoarse",]
names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse"]
names = ["conv_negcoarse",]
names = ["conv_cbig2",]
names=["conv_csmall",]


Delta = 50
kernel="gaussian"
for nn, name in enumerate(names):
    #-----
    print(name)
    ds = xr.open_dataset(f'data/vort_{name}.nc')#.chunk(dict(itime=2))
    #-----

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output')
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
    #-----

    w_star = lp.physics.w_star(sim)
    T_conv = sim.inv_depth/w_star

    #-----
    # Pick a time
    if 0:
        levs = lp.utils.nearest(ds.z, -sim.z_i*np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1/20, 0]))
        ds = ds.sel(z=levs)
    #-----

    #-----
    print("convolving ζ")
    #ds0 = ds.sel(z=0, itime=np.inf, method="nearest")
    ζx_int = ds.ζ.where(ds.w>0, np.nan).interpolate_na("x", period=sim.domain.lx)
    ζy_int = ds.ζ.where(ds.w>0, np.nan).interpolate_na("y", period=sim.domain.ly)
    ζ_pos = (ζx_int + ζy_int)/2
    ζ_filt = lp.vector.gaussian_conv(ζ_pos, delta=Delta, dims=["x", "y"])
    print("convolving w")
    w_filt = lp.vector.gaussian_conv(ds.w, delta=Delta, dims=["x", "y"])
    print("convolving ∇•u")
    hdiv_filt = lp.vector.gaussian_conv(ds.hdiv, delta=Delta, dims=["x", "y"])

    ds_del = xr.Dataset(dict(w=w_filt, hdiv=hdiv_filt, ζ=ζ_filt))
    ds_del.attrs = dict(T_conv=T_conv, Delta=Delta)
    ds_del.to_netcdf(f"data/surf_filtered_{name}.nc")
    #-----

