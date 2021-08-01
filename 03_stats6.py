import numpy as np
import lespy as lp
import xarray as xr
from dask.diagnostics import ProgressBar

path = '/data/1/tomaschor/LES05/{}'
names=["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine", "conv_nccoarse", "conv_ncatcoarse", "conv_cbig2", "conv_negcoarse"]

#++++
# Define plot limits
amp_ζ = 1e-1
amp_ζ_filt = 1e-3
amp_d = 8e-2
amp_d_filt = 1.8e-3
    
bins_ζ = np.linspace(-amp_ζ, amp_ζ, 150)
bins_ζ_filt = np.linspace(-amp_ζ_filt, 2*amp_ζ_filt, 150)
bins_d = np.linspace(-amp_d, amp_d, 120)
bins_d_filt = np.linspace(-amp_d_filt, amp_d_filt, 120)
    
cen_ζ = (bins_ζ[1:] + bins_ζ[:-1])/2
cen_ζ_filt = (bins_ζ_filt[1:] + bins_ζ_filt[:-1])/2
cen_d = (bins_d[1:] + bins_d[:-1])/2
cen_d_filt = (bins_d_filt[1:] + bins_d_filt[:-1])/2
#----


for nn, name in enumerate(names):
    print(); print(name)

    #+++++ Open simulation
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
    else:
        sim = lp.Simulation(path.format(name))

    ds = xr.open_dataset('data/vort_{}.nc'.format(name), chunks=dict(itime=1))
    ds_filt = xr.open_dataset('data/surf_filtered_{}.nc'.format(name), chunks=dict(itime=1))
    #-----

    #++++ Normalize z coordinate
    ds = ds.assign_coords(z=ds.z/sim.z_i)
    ds_filt = ds_filt.assign_coords(z=ds_filt.z/sim.z_i)

    ds = ds.sel(z=slice(-1.1, 0))
    ds_filt = ds_filt.sel(z=slice(-1.1, 0))
    #----

    #+++++ Calculate histogram height-by-height
    jpdfs, jpdfs_filt, jpdfs_cross = [], [], []
    for z in ds_filt.z:

        #+++++ We do the PDF of unfiltered vars first
        jpdf, _, _, = np.histogram2d(ds.sel(z=z).ζ.data.flatten(), 
                                     ds.sel(z=z).hdiv.data.flatten(),
                                     bins=[bins_ζ, bins_d], density=True)
        jpdf = xr.DataArray(jpdf, dims=["zeta", "hdiv"], 
                            coords=dict(zeta=cen_ζ, hdiv=cen_d))
        jpdf = jpdf.expand_dims(dim="z").assign_coords(z=[z])
        jpdfs.append(jpdf)
        #-----

        #+++++ Then we do the PDF of filtered vars
        jpdf_filt, _, _, = np.histogram2d(ds_filt.sel(z=z).ζ.data.flatten(), 
                                          ds_filt.sel(z=z).hdiv.data.flatten(),
                                          bins=[bins_ζ_filt, bins_d_filt], density=True)
        jpdf_filt = xr.DataArray(jpdf_filt, dims=["zeta_filt", "hdiv_filt"], 
                                 coords=dict(zeta_filt=cen_ζ_filt, hdiv_filt=cen_d_filt))
        jpdf_filt = jpdf_filt.expand_dims(dim="z").assign_coords(z=[z])
        jpdfs_filt.append(jpdf_filt)
        #-----

        #+++++
        # The we do filtered hdiv versus unfiltered vorticity
        jpdf_cross, _, _, = np.histogram2d(ds.sel(z=z).hdiv.data.flatten(), 
                                           ds_filt.sel(z=z).hdiv.data.flatten(), 
                                           bins=[bins_d, bins_d_filt], density=True)
        jpdf_cross = xr.DataArray(jpdf_cross, dims=["hdiv", "hdiv_filt"], 
                            coords=dict(hdiv=cen_d, hdiv_filt=cen_d_filt))
        jpdf_cross = jpdf_cross.expand_dims(dim="z").assign_coords(z=[z])
        jpdfs_cross.append(jpdf_cross)
        #-----

    jpdfs = xr.concat(jpdfs, dim="z")
    jpdfs_filt = xr.concat(jpdfs_filt, dim="z")
    jpdfs_cross = xr.concat(jpdfs_cross, dim="z")
    #----

    #++++ Name variable names properly
    jpdfs.attrs = dict(long_name="Joint PDF of vorticity and hor. div.")
    jpdfs_filt.attrs = dict(long_name="Joint PDF of filtered vorticity and hor. div.")
    jpdfs_cross.attrs = dict(long_name="Joint PDF of hor. div. and filtered hor. div.")
    #----

    #++++ Put everything together and save it to disk
    JPDFs = xr.Dataset(dict(jpdf=jpdfs, jpdf_filt=jpdfs_filt, jpdf_cross=jpdfs_cross))

    JPDFs.z.attrs = dict(short_name="z", long_name="Normalized depth (z/h)")
    JPDFs.zeta.attrs = dict(short_name="zeta", long_name="Vorticity (ζ)")
    JPDFs.hdiv.attrs = dict(short_name="hdiv", long_name="Horizontal divergence")
    JPDFs.zeta_filt.attrs = dict(short_name="zeta_filt", long_name="Filtered vorticity (ζ)")
    JPDFs.hdiv_filt.attrs = dict(short_name="hdiv_filt", long_name="Filtered horizontal divergence")

    print("Saving netcdf...")
    outname = f"data/jpdfs_{name}.nc"
    delayed_nc = JPDFs.to_netcdf(outname, compute=False)
    with ProgressBar():
        results = delayed_nc.compute()
    print(f"Done saving to {outname}")
    #----


