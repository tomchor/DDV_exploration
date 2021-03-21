import numpy as np
import lespy as lp
import xarray as xr
from aux01_ddvutils import chunk4d
import dask

#++++ First run this command on IPython
# memory_limit is the limit per worker
#from dask.distributed import Client
#client = Client(n_workers=22, memory_limit='4GB', threads_per_worker=1, processes=True)
#----


path = '/data/1/tomaschor/LES05/{}'

names=["conv_coarse", "conv_atcoarse"]
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse",]
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse", "conv_cbig2", "conv_negcoarse"]



#def runmain():
#    nn, name = 0, names[0]
for nn, name in enumerate(names):
    print(name)
    output_path = path.format(name)+'/output'

    #+++++ Read data
    du = xr.open_dataset(output_path+f'/out.{name}_full.nc')
    du = du.isel(itime=slice(-50, None))
    #-----

    #+++++ Make the data smaller (focus on first 35 m)
    z_slim = xr.concat([ du.z.sel(z=slice(0, -35)), du.z.sel(z=slice(-35, None, 5)) ], dim="z")
    du = du.sel(z=z_slim)
    du = chunk4d(du, time_var="itime", round_func=np.ceil)
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        du["w"] = du.w.interp(zF=du.z)
    #-----

    #+++++ Focus on the surface if the dataset is too large
    print("Sorting")
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # Avoid warning of big slice
        du = du.sortby("z")
    print("Done sorting")
    #-----

    #-----
    # Get the velocity gradient tensor
    try:
        vel_grad = lp.vector.velgrad_tensor2d([du.u, du.v], simulation=None)
    except ValueError:
        vel_grad = lp.vector.velgrad_tensor2d([du.u, du.v], simulation=None, real=False)
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
