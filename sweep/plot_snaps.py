import xarray as xr
import numpy as np
import lespy as lp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

path = '/data/1/tomaschor/LES05/{}'
#names = ["conv_coarse", "conv_atcoarse"]
names = ["conv_fine", "conv_atfine"]
names = ["conv_coarse", "conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_atcoarse", "conv_fine", "conv_atfine"]
names = ["conv_coarse", "conv_nccoarse", "conv_atcoarse", "conv_ncatcoarse"]
names = ["conv_negcoarse",]
names = ["conv_csmall",]

depth = -1.6 # meters
#depth = 0 # meters
diams = [ 5, 7, 9, 11, 13 ]
masses = np.array([160, 220, 280, 340, 400, 460, 520, 580, 640, 700, 760]) # In m**2/s
for diam in diams:
    for mass in masses:
        print(f"diam: {diam}\nmass: {mass}")
        for nn, name in enumerate(names):
            #-----
            if "cbig2" in name:
                sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
            else:
                sim = lp.Simulation(path.format(name))
            #-----

            #-----
            w_star = lp.physics.w_star(sim)
            T_conv = sim.inv_depth/w_star
            label = f"_d{diam}_m{mass}"
            #-----
 
            #-----
            ds = xr.open_dataset(f"../data/vort_{name}.nc")
            ds_ddvs = xr.open_dataset(f"data/ddvs_{name}{label}.nc")

            ds = ds.sel(z=depth, method="nearest")
            #-----

            #-----
            ζ = ds.ζ * T_conv
            #-----

    
            it = len(ζ.itime) -1
            ζ.isel(itime=it).plot.imshow(y="y", figsize=(10,10), cmap="seismic",)
            ax=plt.gca()
            fig=plt.gcf()
            t = ds_ddvs.to_dataframe()
            for idx, row in t[t.frame==it].iterrows():
                circle = Circle((row.X, row.Y), 2*row["size"]*sim.domain.dx, facecolor="none", edgecolor="k")
                ax.add_artist(circle)
            
            ax.set_aspect("equal")
            fig.savefig(f"track_check/track_{name}{label}.png")
            plt.close()
            
#plt.show()
    
    
 
