import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import lespy as lp
from scipy.stats import norm
import matplotlib as mp
mp.rc('font', size=16)

path = '/data/1/tomaschor/LES05/{}'
grid = "coarse"

if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse",]

zeta_e = np.linspace(-1e3, 1e3, 200)
zeta_c = (zeta_e[1:] + zeta_e[:-1])/2

fig, axes = plt.subplots(ncols=2, nrows=len(names), figsize=(12, len(names)*6))
fig2, axes2 = plt.subplots(ncols=len(names), figsize=(len(names)*5, 5))
fig3, axes3 = plt.subplots(ncols=2, figsize=(12, 8))
for nn, name in enumerate(names):
    sim = lp.Simulation(path.format(name))
    out = lp.Output(path.format(name)+'/output')
    w_star = lp.physics.w_star(sim)
    T_conv = sim.inv_depth/w_star

    #-----
    ds = xr.open_dataset(f"data/3d_ddv_{name}.nc")
    surf = xr.open_dataset(f"data/vort_{name}.nc")
    #-----

    #-----
    avgds = ds.mean("ddv")
    avgds.sel(z=slice(-sim.z_i/10, 0)).transpose().to_netcdf(f"data/avg_3d_{name}.nc")
    #-----

    #-----
    avgds.ζ_ddv.attrs = dict(long_name=r"$ζ \times T_*$", units="")
    avgds.w_ddv.attrs = dict(long_name=r"$w / T_*$", units="")
    #-----

    #-----
    # Plot DDV data
    avgds.sel(y=0).ζ_ddv.plot(y="z", ax=axes[nn, 0], vmin=0, vmax=300)
    avgds.sel(y=0).w_ddv.plot(y="z", ax=axes[nn, 1], vmin=-0.01, vmax=0.01, cmap="RdBu_r")
    #-----

    #-----
    axes[nn,0].set_title(f"Simulation: {name}")
    axes[nn,1].set_title(f"Simulation: {name}")
    #-----

    #-----
    rv = norm(scale=surf.ζ.isel(z=-1).std()*T_conv)
    norm_vort_z = surf.ζ*T_conv
    norm_vort_z.attrs = dict(long_name=r"Normalized vertical vorticity ($ζ \times T_*$)", units="")
    norm_vort_z.isel(z=-1).plot.hist(bins=zeta_e, density=True, ax=axes2[nn])
    axes2[nn].semilogy(zeta_e, rv.pdf(zeta_e), 'k-', lw=2, label='Normal distribution')
    axes2[nn].set_title(f"PDF. Simulation: {name}")
    #-----

    #-----
    avgds.ζ_ddv.sel(x=0, y=0).plot(y="z", ax=axes3[0], label=f"Simulation: {name}", lw=3)
    avgds.w_ddv.sel(x=0, y=0).plot(y="z", ax=axes3[1], label=f"Simulation: {name}", lw=3)
    #-----

for ax in axes.flatten():
    ax.grid()
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-15, None)
fig.tight_layout()
fig.savefig(f"figures/DDV_crosssec_{grid}.png")

for ax in axes2.flatten():
    ax.grid()
    ax.set_ylim(1e-8, 1e-1)
    ax.set_xlim(zeta_e[0], zeta_e[-1])
#    ax.legend()
fig2.tight_layout()
fig2.savefig(f"figures/hist_{grid}.png")

for ax in axes3.flatten():
    ax.grid()
    ax.legend()
    ax.set_ylim(-15, None)
fig3.tight_layout()
fig3.savefig(f"figures/profiles_{grid}.png")

