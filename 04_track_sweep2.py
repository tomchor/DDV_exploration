import xarray as xr
import trackpy as tp
import numpy as np
import pandas as pd
import pims
import lespy as lp
from aux_ddv import upodd
from os import system

path = '/data/1/tomaschor/LES05/{}'
#names = ["conv_coarse", "conv_atcoarse"]
names = ["conv_fine", "conv_atfine"]
names = ["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse"]
names = ["conv_negcoarse",]
names = ["conv_cbig2",]
names = ["conv_csmall",]

#names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse", "conv_cbig2",]

depth = -1.6 # meters
diams = [ 5, 7, 9, 11, 13 ]
masses = np.array([160, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760]) # In m**2/s
for diam in diams:
    for vortex_dcirc in circulations:
        print(f"diam: {diam}\nmass: {vortex_dcirc}")
        for nn, name in enumerate(names):
            #-----
            if "cbig2" in name:
                sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
            else:
                sim = lp.Simulation(path.format(name))
            #-----

            w_star = lp.physics.w_star(sim)
            T_conv = sim.inv_depth/w_star
        
            pixelsize = diam
            vortex_dsize = diam*sim.domain.dx
            vortex_pmass = vortex_dcirc/sim.Δx/sim.Δy
            print(f"Actual size of vortices:", diam*sim.domain.dx)
            print()
            options = dict(minmass=vortex_pmass, max_iterations=300, engine="numba", characterize=True)
            #-----
        
            #-----
            print("Reading data from disk")
            ds = xr.open_dataset(f"data/vort_{name}.nc")
            ds_filt = xr.open_dataset(f"data/surf_filtered_{name}.nc")
        
            ds = ds.sel(z=depth, method="nearest")
            ds_filt = ds_filt.sel(z=depth, method="nearest")
            #-----
        
            #------
            print("Rescaling data")
            ζ = ds.ζ * T_conv
            ζ_pos = ζ.where(ζ>0, 0)
            ζ_neg = abs(ζ.where(ζ<0, 0))
            #-----
        
        
            #-----
            # Identify the DDVs
            frames_pos = pims.Frame(abs(ζ_pos.transpose("itime", "y", "x")).values)
            frames_neg = pims.Frame(abs(ζ_neg.transpose("itime", "y", "x")).values)
        
            f_pos = tp.batch(frames_pos, pixelsize, **options);
            f_neg = tp.batch(frames_neg, pixelsize, **options);
            print(f"len(positive): {len(f_pos)}, len(negative): {len(f_neg)}")
        
            if len(f_pos)==0:
                continue
            else:
                t_pos = tp.link_df(f_pos, 5, memory=3)

            if len(f_neg)==0:
                t = t_pos
            else:
                t_neg = tp.link_df(f_neg, 5, memory=3)
                t_neg.index = t_neg.index + max(t_pos.index) + 1
                t_neg.particle = t_neg.particle + max(t_pos.particle) + 1
                t = pd.concat([t_pos, t_neg])
            #-----
        
            #-----
            # Interpolating location to physical units
            Y = np.interp(t.y, np.arange(0, len(ds.y)), ds.y)
            X = np.interp(t.x, np.arange(0, len(ds.x)), ds.x)
    
            t["X"] = X; t["Y"] = Y
            t["itime"] = ds.ζ.itime.isel(itime=t.frame)
            t.index.name="ddv"
            #-----

            #-----
            # Get rid of spurious DDVs that are too diffuseand too weak
            if 0:
                s90 = np.percentile(t["size"], 90)
                m10 = np.percentile(t["mass"], 10)
                t = t[((t['mass'] > m10) | (t['size'] < s90))]
            #-----
        
            #-----
            hdiv_ddvs, hdiv_filt_ddvs, ζ_ddvs, ζ_filt_ddvs = [], [], [], []
            print("Getting hdiv and ζ in cores",)
            for index in t.index:
                print(end='.')
                ddv = t.loc[index]
        
                ζ_ddv = ds.ζ.sel(itime=ddv.itime).interp(x=ddv.X, y=ddv.Y)
                ζ_filt_ddv = ds_filt.ζ.sel(itime=ddv.itime).interp(x=ddv.X, y=ddv.Y)
                hdiv_ddv = ds.hdiv.sel(itime=ddv.itime).interp(x=ddv.X, y=ddv.Y)
                hdiv_filt_ddv = ds_filt.hdiv.sel(itime=ddv.itime).interp(x=ddv.X, y=ddv.Y)
        
                ζ_ddvs.append(ζ_ddv.item())
                ζ_filt_ddvs.append(ζ_filt_ddv.item())
                hdiv_ddvs.append(hdiv_ddv.item())
                hdiv_filt_ddvs.append(hdiv_filt_ddv.item())
            t["ζ"] = ζ_ddvs
            t["ζ_filt"] = ζ_filt_ddvs
            t["hdiv"] = hdiv_ddvs
            t["hdiv_filt"] = hdiv_filt_ddvs
            print("done")
            #-----
        
        
            #-----
            ds_ddvs = xr.Dataset.from_dataframe(t)
            ds_ddvs.attrs = dict(T_conv=T_conv, vortex_dsize=vortex_dsize, vortex_dcirc=vortex_dcirc)
            label = f"_d{pixelsize}_m{vortex_dcirc}"

            system(f"mkdir -p data")
            system(f"mkdir -p track_check")

            ds_ddvs.to_netcdf(f"sweep/data/ddvs_{name}{label}.nc")
            #-----

            if 0:
                from matplotlib import pyplot as plt
                from matplotlib.patches import Circle
                for it, itime in enumerate(ds.itime):
                    print(itime.item())
                    ζ[it].plot.imshow(y="y", figsize=(10,10), cmap="seismic")
                    ax=plt.gca()
                    fig=plt.gcf()
                    for idx, row in t[t.frame==it].iterrows():
                        circle = Circle((row.X, row.Y), 2*row["size"]*sim.domain.dx, facecolor="none", edgecolor="k")
                        ax.add_artist(circle)
        
            #        plt.show()
            #        exit()
                    fig.savefig(f"track_check{label}/track_{name}_{it}.png")
                    plt.close()
    
            elif 0:
                from matplotlib import pyplot as plt
                from matplotlib.patches import Circle
    
                it = len(ζ.itime) -1
                ζ.isel(itime=it).plot.imshow(y="y", figsize=(10,10), cmap="seismic",)
                ax=plt.gca()
                fig=plt.gcf()
                for idx, row in t[t.frame==it].iterrows():
                    circle = Circle((row.X, row.Y), 2*row["size"]*sim.domain.dx, facecolor="none", edgecolor="k")
                    ax.add_artist(circle)
            
                ax.set_aspect("equal")
                fig.savefig(f"sweep/track_check/track_{name}{label}.png")
                plt.close()
            
    
    
            
