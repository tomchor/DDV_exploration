import numpy as np
import lespy as lp
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mp
from aux_ddv import moments, snames
mp.rc('font', size=12)
cs = ["C0", "C1", "C2", "C4"]

path = '/data/1/tomaschor/LES05/{}'
grid="allcoarse"
#grid="coriolis"
grid="resolution"


if grid=="coarse":
    names=["conv_coarse", "conv_atcoarse"]
elif grid=="fine":
    names=["conv_fine", "conv_atfine"]
elif grid=="coriolis":
    names=["conv_coarse", "conv_nccoarse", "conv_negcoarse"]
elif grid=="allcoarse":
    names=["conv_coarse", "conv_atcoarse", "conv_nccoarse", "conv_ncatcoarse"]
elif grid=="resolution":
    names=["conv_coarse", "conv_cbig2", "conv_csmall"]


fig_skew, axes_skew = plt.subplots(ncols=3, figsize=(12, 6))
for nn, name in enumerate(names):
    print(name)
    c=cs[nn]
    #-----
    if "cbig2" in name:
        sim = lp.Simulation_sp(path.format(name)+'/readin/param.nml')
        out = lp.Output_sp(path.format(name)+'/output_50k')
        u2 = lp.read_aver(path.format(name)+'/result/aver_u2.out', 
                          simulation=sim, dims=["ndtime", "z_u"])*sim.u_scale**2
        w2 = lp.read_aver(path.format(name)+'/result/aver_w2.out', 
                          simulation=sim, dims=["ndtime", "z_w"])*sim.u_scale**2
        θ = 2*sim.t_scale - lp.read_aver(path.format(name)+'/result/aver_theta.out', 
                                         simulation=sim, dims=["ndtime", "z_u"])*sim.t_init
    #-----
    else:
        sim = lp.Simulation(path.format(name))
        out = lp.Output(path.format(name)+'/output')
        u2 = lp.read_aver(path.format(name)+'/output/aver_u2.out', 
                          simulation=sim, dims=["ndtime", "z_u"])*sim.u_scale**2
        w2 = lp.read_aver(path.format(name)+'/output/aver_w2.out', 
                          simulation=sim, dims=["ndtime", "z_w"])*sim.u_scale**2
        θ = 2*sim.t_scale - lp.read_aver(path.format(name)+'/output/aver_theta.out', 
                                         simulation=sim, dims=["ndtime", "z_u"])*sim.t_init
    #-----

    T_conv = sim.inv_depth/sim.w_star

    #----
    #----

    #----
    # Define plot limits
    if "coarse" in name:
        amp_ζ = 8e-2
        amp_ζ_filt = 1e-3
        amp_d = 3e-2
        amp_d_filt = 1.8e-3

        ζ_lim_filt = [-amp_ζ_filt, 2*amp_ζ_filt]

    elif "fine" in name:
        amp_ζ = 1e-1
        amp_ζ_filt = 5e-4
        amp_d = 5e-2
        amp_d_filt = 1.5e-3

        ζ_lim_filt = [-amp_ζ_filt, amp_ζ_filt]
    #----

    #----
    # Open dataset
    ds = xr.open_dataset(f"data/jpdfs_{name}.nc")
    # Deal with different grids
    jpdf = ds.jpdf#.dropna("zeta", how="all").dropna("hdiv", how="all")
    jpdf_filt = ds.jpdf_filt#.dropna("zeta", how="all").dropna("hdiv", how="all")
    jpdf_cross = ds.jpdf_cross#.dropna("zeta", how="all").dropna("hdiv", how="all")
    #----

    #----
    # Plot PDFs
    pdf_ζ = jpdf.integrate("hdiv")
    pdf_ζ_filt = jpdf_filt.integrate("hdiv_filt")

    pdf_ζ.attrs = dict(long_name="PDF of vorticity")
    pdf_ζ_filt.attrs = dict(long_name="PDF of filtered vorticity")
    #----

    #----
    # Plot Skewness
    var_ζ, skew_ζ, kurt_ζ = moments(pdf_ζ, "zeta")

    var_ζ.attrs = dict(long_name="Variance (1/s)")
    skew_ζ.attrs = dict(long_name="Skewness")
    kurt_ζ.attrs = dict(long_name="Kurtosis")
    
    var_ζ .plot(ax=axes_skew[0], y="z", label=snames[name], lw=3, ls="--", c=c)
    skew_ζ.plot(ax=axes_skew[1], y="z", label=snames[name], lw=3, ls="--", c=c)
    kurt_ζ.plot(ax=axes_skew[2], y="z", label=snames[name], lw=3, ls="--", c=c)
    #----


axes_skew[2].axvline(x=3, color="k", ls="-.", label="Ku = 3")
axes_skew[2].set_xlim(0, None)

for i, ax in enumerate(axes_skew):
    ax.grid()
    ax.legend()
    ax.set_ylim(-.8, 0)
    ax.locator_params(axis='x', nbins=3)
#    if i in [0,1]:
#        ax.set_xlabel("")
fig_skew.tight_layout(pad=0)
fig_skew.savefig(f"figures/moments_z_{grid}.png")
