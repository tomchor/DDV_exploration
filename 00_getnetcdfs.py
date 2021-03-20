import numpy as np
import lespy as lp
import xarray as xr
import os
from aux00_siminfo import useful_attrs


path = '/data/1/tomaschor/LES05/{}'

names=["conv_coarse", "conv_atcoarse"]
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse",]
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse", "conv_cbig2", "conv_negcoarse"]
names=["conv_cbig2", "conv_negcoarse",]

Nts=[100]*len(names)

for nn, name in enumerate(names):
    print(name)
    Nt = Nts[nn]
    output_path = path.format(name)+'/output'

    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(output_path)
        depth = 50
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(output_path)
        depth = 90
    #-----

    #-----
    ev=1
    times = out.binaries[-Nt*ev:][::-1][::ev].index[::-1]
    #-----

    #++++
    # Get the velocities on parallel at the appropriate times
    for time in times:
        #++++ create reading options
        if "cbig2" not in name:
            uvw_file, θ_file = out.binaries.loc[time][["uvw_jt", "theta_jt"]]
            nc_file = uvw_file.replace("uvw_jt", f"out.{name}_").replace(".sbin", ".nc")
        else:
            field_file, = out.binaries.loc[time][["field_"]]
            nc_file = field_file.replace("field_", f"out.{name}_").replace(".out", ".nc")
        opts = dict(simulation=sim, nz=(sim.nz_tot+1)*3//4)
        #----

        #++++ Read the files
        u, v, w = out.compose_uvw(times=[time], **opts)
        θ = out.compose_theta(times=[time], **opts)
        #----

        #++++ Rename w since it's on a different location
        w = w.rename(z="zF")
        #----

        #+++ create file and save it
        dsout = xr.Dataset(dict(u=u, v=v, w=w, θ=θ))
        dsout.attrs = { k : v for k, v in sim.__dict__.items() if k in useful_attrs } # Keep only useful attrs
        dsout.attrs = { k : v for k, v in dsout.attrs.items() if v is not None } # Remove None values
        dsout.attrs = { k : (v if (type(v) is not bool) else int(v)) for k, v in dsout.attrs.items() } # Remove bool values

        print(f"Creating {nc_file}")
        dsout.to_netcdf(nc_file)

    indiv_nc_files = output_path+"/out.*_????????.nc"
    fullds = xr.open_mfdataset(indiv_nc_files)
    full_name = nc_file[:-11]+"full.nc"
    print(f"\nWriting full dataset to {full_name}")
    fullds.to_netcdf(full_name)

    print("You can remove the unused files with the last command")
    os.system(f"ls {indiv_nc_files}")
    #-----


