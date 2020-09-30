import xarray as xr
import numpy as np
import lespy as lp
from os import system
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

path = '/data/1/tomaschor/LES05/{}'
names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse"]
names = ["conv_negcoarse",]
names = ["conv_cbig2",]

#names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse", "conv_cbig2",]

depth = -1.6 # meters
diams = [ 5, 7, 9, 11, 13 ]
circulations = np.array([160, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760]) # In m**2/s
for diam in diams:
    pixelsize=diam
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
        
            #-----
            ds = xr.open_dataset(f"data/vort_{name}.nc")
            ds = ds.sel(z=depth, method="nearest")
            ζ = ds.ζ * T_conv

            label = f"_d{pixelsize}_m{vortex_dcirc}"
            try:
                ds_ddvs = xr.open_dataset(f"sweep/data/ddvs_{name}{label}.nc")
            except:
                print(label, "No data!")
            #-----

    
            it = ds_ddvs.frame.max().item()
            ζ.isel(itime=it).plot.imshow(y="y", figsize=(10,10), cmap="seismic",)
            ax=plt.gca()
            fig=plt.gcf()
            t = ds_ddvs.to_dataframe()
            for idx, row in t[t.frame==it].iterrows():
                circle = Circle((row.X, row.Y), 2*row["size"]*sim.domain.dx, facecolor="none", edgecolor="k")
                ax.add_artist(circle)
            
            ax.set_aspect("equal")
            fig.savefig(f"sweep/track_check/track_{name}{label}.png")
            plt.close()
            
    
    
            
