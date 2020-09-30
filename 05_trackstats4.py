import xarray as xr
import numpy as np
import lespy as lp
from os import system

path = '/data/1/tomaschor/LES05/{}'
names = ["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_nccoarse"]
names=["conv_ncatcoarse",]
names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse"]
names = ["conv_negcoarse",]

names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse", "conv_cbig2",]
names = ["conv_csmall",]

radius = 4
labels=["d7_m280", "d7_m340", "d7_m400", "d7_m460"]
labels=["d5_m340"]
for label in labels:
    print(label)
    system(f"mkdir -p figures_{label}")
    system(f"mkdir -p data_{label}")
    for nn, name in enumerate(names):
        #-----
        if "cbig2" in name:
            sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        else:
            sim = lp.Simulation(path.format(name))
        #-----

        w_star = lp.physics.w_star(sim)
        T_conv = sim.inv_depth/w_star
    
        #-----
        ds = xr.open_dataset(f"data/vort_{name}.nc")
        ddvs = xr.open_dataset(f"sweep/data/ddvs_{name}_{label}.nc")
        filt = xr.open_dataset(f"data/surf_filtered_{name}.nc")
        #-----
    
        #-----
        if 1:
            ds = ds.sel(z=slice(-sim.z_i/3, 0))
        #-----
    
        #-----
        dx, dy = sim.domain.dx, sim.domain.dy
        xof=np.arange(0, radius, dx)
        xof=np.sort(np.unique([*xof, *-xof]))
        yof=np.arange(0, radius, dy)
        yof=np.sort(np.unique([*yof, *-yof]))
        #-----
    
        #-----
        # Get data in DDVs
        zetas, ws = [], []
        print("getting DDV stats")
        count=0
        for fr, itime in enumerate(ds.itime):
            print(fr)
            ddvs_t = ddvs.where(ddvs.itime==itime).dropna("ddv")
            X = ddvs_t.X.values
            Y = ddvs_t.Y.values
    
            for x, y in zip(X, Y):
                #print(x, y)
                ζ_ddv = ds.ζ.sel(itime=itime).interp(dict(x=x+xof, y=y+yof)).assign_coords(x=xof, y=yof)
                w_ddv = ds.w.sel(itime=itime).interp(dict(x=x+xof, y=y+yof)).assign_coords(x=xof, y=yof)
    
                zetas.append(ζ_ddv)
                ws.append(w_ddv)
    
                w_ij = filt.w.sel(z=-sim.domain.dz/2, itime=itime).interp(dict(x=x, y=y,))
                if w_ij<0:
                    count+=1
    
        print(f"\nRatio of ddvs in convergence zones for {name}: {count/len(ddvs.ddv)}\n")
        ζ_ddvs = xr.concat(zetas, dim="ddv")
        w_ddvs = xr.concat(ws, dim="ddv")
        ratio_inside = count/len(ddvs.ddv)
        #-----
    
        #-----
        dsout = xr.Dataset(dict(ζ_ddv=ζ_ddvs, w_ddv=w_ddvs), attrs=dict(ratio_inside=ratio_inside, T_conv=T_conv))
        dsout.to_netcdf(f"data_{label}/3d_ddv_{name}.nc")
        #-----
        
