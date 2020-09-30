import numpy as np
import lespy as lp
import xarray as xr
#from dask.distributed import Client
#client = Client(memory_limit='20GB', processes=False, n_workers=2)


path = '/data/1/tomaschor/LES05/{}'

names=["conv_coarse", "conv_atcoarse"]
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse",]
names=["conv_negcoarse",]
names=["conv_cbig2",]
names=["conv_csmall",]

Nts=[30, 30, 30, 30]
evfine = 5
tchunk = 2



#def runmain():
#    nn, name = 0, names[0]
for nn, name in enumerate(names):
    print(name)
    Nt = Nts[nn]

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output')
        depth = 50
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
        depth = 90
    #-----

    #-----
    if "fine" in name:
        ev=evfine
    else:
        ev=1
    times = out.binaries[-Nt*ev:][::-1][::ev].index[::-1]
    #-----

    #-----
    # Get the velocities on parallel at the appropriate times
    u,v,w = out.compose_uvw(simulation=sim, apply_to_z=False, times=times, nz=int(depth/sim.domain.dz), chunksize=tchunk)
    θ = out.compose_theta(simulation=sim, apply_to_z=False, times=times, nz=int(depth/sim.domain.dz), chunksize=tchunk)
    #-----

    #-----
    # Get only the top of the
    du = xr.Dataset(dict(u=u, v=v, θ=θ))
    if len(du.z)>200:
        z_slim = xr.concat([ du.z.sel(z=slice(0, -30)), du.z.sel(z=slice(-30, None, 5)) ], dim="z")
        du = du.sel(z=z_slim)
    du["w"] = w.interp(coords=dict(z=du.z))
    print("Sorting")
    du = du.sortby("z")
    print("Done sorting")
    u, v, w, θ = 1,1,1,1
    del u, v, w, θ
    #-----

    #-----
    # Get the velocity gradient tensor
    try:
        vel_grad = lp.vector.velgrad_tensor2d([du.u, du.v], simulation=sim)
    except ValueError:
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

    print("Saving...", name)
    ζ.attrs = dict(long_name="Vorticity", short_name="ζ", units="1/s")
    hdiv.attrs = dict(long_name="Horizontal divergence", short_name="Div", units="1/s")
    ds = xr.Dataset(dict(ζ=ζ, hdiv=hdiv, w=du.w, θ=du.θ))
    ds.to_netcdf(f'data/vort_{name}.nc')

#    exit()
