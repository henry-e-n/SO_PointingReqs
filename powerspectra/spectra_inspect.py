# %%
import numpy as np
import matplotlib.pyplot as plt
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils

# %%
# Load the spectra from the data file
spec_file = "/shared_home/henachman/home/SO_PointingReqs/powerspectra/HPtoast_mr_result_spectra_spin0and2/spectra_split_0xsplit_0_run_HEALPIX.dat"
clfile = "/shared_home/henachman/home/software/pspy/tutorials/data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
test_dir = "HPtoast_mr_result_spectra_spin0and2"

lmax=3000

lb, Db_spec = so_spectra.read_ps(spec_file)

l, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)
# ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory, spectra=spectra)


# %%
spectra = ["TT"]
plt.figure(figsize=(10,8))
for c,f in enumerate(spectra):
    plt.plot(lb, Db_spec, "o")
    # plt.plot(lb, ps_theory_b[f], color="lightblue", label="binned theory")
    plt.plot(l, ps_theory[f], color="grey", label="theory")
    plt.ylabel(r"$D^{%s}_{\ell}$" % f, fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    if c == 0:
        plt.legend()
plt.xlim(1000, 3000)
plt.semilogy() #ylim(0, 2000)
plt.savefig("%s/TTspectra_zoomview.png" % (test_dir), bbox_inches="tight")
plt.show()
plt.clf()
plt.close()
