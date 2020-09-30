import lespy as lp
import matplotlib
matplotlib.use('TkAgg')
from xmovie import Movie
import numpy as np
import xarray as xr

paths = '/data/1/tomaschor/LES05/{}'
names=["conv_cbig2",]
names=["conv_csmall",]

Nt = 3000
#Nt = 98
nn = 1
for nj, name in enumerate(names):
    print(name)

    #------
    # Prelim variables
    path = paths.format(name)

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output')
        times = np.vectorize(lp.utils.nameParser)(out.binaries.field_.dropna().iloc[::-1])[:Nt*nn:nn][::-1]

        wθ_res_path = path+"/result/aver_wt.out"
        wθ_sgs_path = path+"/result/aver_sgs_t3.out"
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
        times = np.vectorize(lp.utils.nameParser)(out.binaries.uvw_jt.dropna().iloc[::-1])[:Nt*nn:nn][::-1]

        wθ_res_path = path+"/output/aver_wt.out"
        wθ_sgs_path = path+"/output/aver_sgs_t3.out"
    #-----


    if not sim.stokes_flag:
        sim.La_t = np.inf
    else:
        sim.La_t = np.nan
    #------
    print("Reading snapshots")
    nz = 4
    u,v,w = out.compose_uvw(simulation=sim, times=times, nz=nz)


    print("Reading averaged profs")
    wθ_res = lp.read_aver(wθ_res_path, simulation=sim, dims=["ndtime", "z_w"])*sim.u_scale*sim.t_scale
    wθ_sgs = lp.read_aver(wθ_sgs_path, simulation=sim, dims=["ndtime", "z_w"])*sim.u_scale*sim.t_scale

    #------
    print("Getting total fluxes")
    wθ = (wθ_res + wθ_sgs).isel(ndtime=slice(-Nt,None))
    #------

    #------
    wθ = wθ.assign_coords(ndtime=w.itime.values).rename(ndtime="itime")
    #------

    #------
    print("Getting datset")
    ds = xr.Dataset(dict(w=w.isel(z=nz-1), wθ=wθ/sim.wt_s))
    ds = ds.assign_coords(itime=np.round(ds.itime*sim.dt/60/60, decimals=3)).rename(itime="time")
    ds.time.attrs = dict(units="hour")
    #------

    def plot_func(ds, fig, tt):
        """
        Plot a w snapshot on the left, and a wθ mean snapshot on the right
        """
        axes = fig.subplots(ncols=2, gridspec_kw=dict(width_ratios=[5,1]))
        ds = ds.isel(time=tt)

        ds.w.plot(y="y", ax=axes[0], vmin=-2.5e-2, vmax=2.5e-2, cmap="RdBu_r")
        ds.wθ.plot(y="z", ax=axes[1])

        axes[1].grid(True)
        axes[1].set_xlim(-0.3, 1)

        fig.tight_layout()
        return

    mov = Movie(ds, plot_func, framedim="time")
    print("saving")
    mov.save(f"anims/evolution_{name}.mp4",
             overwrite_existing=True,
             framerate=15,)

