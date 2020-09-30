import numpy as np
import lespy as lp
import xarray as xr
from aux_ddv import snames
import xmovie as xm
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

path = '/data/1/tomaschor/LES05/{}'

cmap="RdBu_r"
depth=-1.6
options = dict(cmap=cmap, cbar_kwargs=dict(extend="both", shrink=0.7), interpolation="bicubic", y="y")

grid="allcoarse"
grid="resolution"
if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse",]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]
elif grid=="resolution":
    names=["conv_cbig2", "conv_coarse", ]
names=["conv_cbig2",]
names=["conv_csmall",]



levels = -np.array([8, 10, 15,])

Nt=3000
evfine = 6
tchunk = 2

for nn, name in enumerate(names):
    print(name)
    sname = snames[name]

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output')
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
    #-----

    T_conv = sim.z_i/sim.w_star

    #-----
    ev=1
    times = out.binaries[-Nt*ev:][::-1][::ev].index[::-1]
    #-----

    #-----
    # Get the velocities on parallel at the appropriate times
    u,v,w = out.compose_uvw(simulation=sim, apply_to_z=False, times=times, nz=3,)

    du = xr.Dataset(dict(u=u, v=v))
    du = du.sel(z=depth, method="nearest")

    hours = np.around((u.itime - u.itime[0])*sim.dt/60/60, decimals=2)
    du = du.assign_coords(itime=hours).rename(itime="hours")
    #-----

    #-----
    # Get the velocity gradient tensor
    vel_grad = lp.vector.velgrad_tensor2d([du.u, du.v], simulation=sim, real=False)
    #-----

    #------
    # Calculate vertical vorticty for every point
    print("Calculating vertical vort")
    ζ = vel_grad[1,0] - vel_grad[0,1]
    #------

    #------
    # Calculate Okubo-Weiss parameter for every point
    print("Calculating horizontal divergence")
    hdiv = vel_grad[0,0] + vel_grad[1,1]
    #------

    #------
    ds = xr.Dataset(dict(ζ=ζ, hdiv=hdiv))
    ds = ds*T_conv
    ds.hours.attrs = dict(long_name="Time", units="hour")
    ds.ζ.attrs = dict(long_name=r"Normalized vertical vorticity ($\zeta\times T_*$)")
    ds.hdiv.attrs = dict(long_name=r"Normalized horizontal divergence (Div $\times T_*$)")
    #------

    def plotfunc(ds, fig, tt):
        fig.set_size_inches(18., 8)
        axes = fig.subplots(ncols=2)

        dsi = ds.isel(hours=tt)
        
        dsi.ζ.plot.imshow(ax=axes[0], vmin=-100, vmax=100, **options)
        dsi.hdiv.plot.imshow(ax=axes[1], vmin=-60, vmax=60, **options)

        for ax in axes:
            ax.set_aspect(1)

        fig.tight_layout()
        fig.suptitle(f"Simulation: {sname}", fontsize=26)
        return


    mov = xm.Movie(ds, plotfunc, framedim="hours")

    print("Saving...")
    mov.save(f"anims/zeta_hdiv_{name}.mp4", overwrite_existing=True, framerate=15)
    print("Done")

