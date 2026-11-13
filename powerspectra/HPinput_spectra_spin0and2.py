# %% MARK: Imports
# This script is an adapted copy of the tuto_spectra_spin0and2.py script from pspy.
"""
This script tests the spectra computation for spin 0 and 2 fields.
The first part is done on CAR pixellisation and the second in HEALPIX pixellisation.
This illustrates how pspy functions are blind to pixellisation.
The CAR survey mask is defined with equatorial coordinates 'ra0', 'ra1', 'dec0', 'dec1' and a resolution of 'res' arcminute.
The HEALPIX survey mask is a disk centered on longitude 'lon' and latitude 'lat' in degree  with a given 'radius' the resolution is given by 'nside'.
For each pixellisation we simulate 2 splits with 'rms_uKarcmin_T' noise rms in temperature.
We also include a point source mask with 'source_mask_nholes' holes of size 'source_mask_radius' in arcminute
We appodize the survey mask with an apodisation 'apo_radius_degree_survey' and the point source mask with an apodisation 'apo_radius_degree_mask'
We compute the spectra between the 2 splits of data up to 'lmax'.
"""

import matplotlib
matplotlib.use("Agg")
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
import numpy as np
import healpy as hp
import pylab as plt
plt.style.use("/shared_home/henachman/.matplotlib/stylelib/HNCustomMono.mplstyle")
import os
from tqdm import tqdm

pbar = tqdm(total=100, unit="MB")


# %% MARK: Survey Params
#We then specify the HEALPIX survey parameter, it will be a disk of radius 'radius' degree centered on longitude 'lon' degree and latitude 'lat' degree
# It will have a resolution 'nside'
lon, lat = 0, -1.0
radius = 12
# ncomp=3 mean that we are going to use spin0 and 2 field
ncomp = 3
# specify the order of the spectra, this will be the order used in pspy
# note that if you are doing cross correlation between galaxy and kappa for example, you should follow a similar structure
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
# clfile are the camb lensed power spectra
clfile = "/shared_home/henachman/home/software/pspy/tutorials/data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat"
path_to_fits = "/shared_home/henachman/data/sim_outputs/CMB/BigSquare_lruK_day1_f090_261112_232415/filterbin_filtered_map.fits"
# path_to_fits = "/shared_home/henachman/home/SO_PointingReqs/powerspectra/HP_result_spectra_spin0and2/split_0_run_HEALPIX.fits"
# n_splits stand for the number of splits we want to simulate
n_splits = 1 # HN - I think the splits just change noise realizations. But I could replace splits with different splits of the TOD...in theory
# the maximum multipole to consider
lmax = 3000
# the number of iteration in map2alm
niter = 10
# the noise on the spin0 component, if not specified, the noise in polarisation wil be sqrt(2)x that
rms_uKarcmin_T = 20e-5 # making noise very small because units are wrong for now
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey = 1
# the number of holes in the point source mask
source_mask_nholes = 50 # 100
# the radius of the holes (in arcminutes)
source_mask_radius = 5 # 10
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask = 0.3
# if you want to write down the data file
write_data = True

test_dir = "HPtoast_mr_result_spectra_spin0and2"
try:
    os.makedirs(test_dir)
except:
    pass


# MARK: Binning file
# create a binningfile with format, lmin,lmax,lmean
pspy_utils.create_binning_file(bin_size=25, n_bins=1000, file_name="%s/binning.dat" % test_dir)
binning_file = "%s/binning.dat" % test_dir

pbar.update(10)

# %% MARK: Read CMB
cmb = so_map.read_map(
    file=path_to_fits,
    coordinate="C",
)
nside = cmb.nside
print(f"nside: {nside}")
pbar.update(10)

# MARK: Window Function
# the binary template for the window functionpixels
# for Healpix we set pixel inside the disk at 1 and pixel outside at zero
binary_healpix = so_map.healpix_template(ncomp=1, nside=nside)
vec = hp.pixelfunc.ang2vec(lon, lat, lonlat=True)
disc = hp.query_disc(nside, vec, radius=radius * np.pi / 180)
binary_healpix.data[disc] = 1

# for the CAR survey we will use an apodisation type designed for rectangle maps, and a C1 apodisation for the healpix maps
apo_type = "C1"
run = "run_HEALPIX"

# %% MARK: Start PS est
# Ok let's loop on the CAR and HEALPIX case, starting with CAR
print(">Start %s" % run)

# %% MARK: Gen. a CMB 
#First let's generate a CMB realisation

splitlist = []
nameList = []
pbar.update(10)

# %% MARK: Gen. Splits
print(">Generate sims")
#let's make 2 splits out of it, each with 'rms_uKarcmin_T' rms in T and 'rms_uKarcmin_T'xsqrt(2) rms in pol
for i in range(n_splits):
    split = cmb.copy()
    noise = so_map.white_noise(split, rms_uKarcmin_T=rms_uKarcmin_T)
    split.data += noise.data
    splitlist += [split]
    nameList += ["split_%d" % i]
    split.plot(hp_gnomv=(lon, lat, 2000, 1), file_name="%s/split_%d_%s"%(test_dir, i, run))
    if write_data==True:
        split.write_map("%s/split_%d_%s.fits"%(test_dir, i, run))

pbar.update(20)

# %% MARK: Window & Appodization
print(">Generate window")
#we then apodize the survey mask
window = so_window.create_apodization(binary_healpix, apo_type=apo_type, apo_radius_degree=apo_radius_degree_survey)
#we create a point source mask
mask = so_map.simulate_source_mask(binary_healpix, n_holes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
#... and we apodize it
mask = so_window.create_apodization(mask, apo_type="C1", apo_radius_degree=apo_radius_degree_mask)
#the window is given by the product of the survey window and the mask window
window.data *= mask.data

if write_data == True:
    window.write_map("%s/window_%s.fits"%(test_dir, run))

pbar.update(10)

# %% MARK: Plot Map
#let's look at it

if run == "run_CAR":
    window.plot(file_name="%s/window_%s"%(test_dir, run))
else:
    window.plot(file_name="%s/window_%s"%(test_dir, run), hp_gnomv=(lon, lat, 2000, 1))

# %% MARK: Compute MCM
print(">Compute mcm")

#for spin0 and 2 the window need to be a tuple made of two objects
#the window used for spin0 and the one used for spin 2

window = (window, window)

# Plot the windowed map
windowed_map = cmb.copy()
if cmb.ncomp == 3:
    windowed_map.data[0] = cmb.data[0]*window[0].data
    windowed_map.data[1] = cmb.data[1]*window[1].data
    windowed_map.data[2] = cmb.data[2]*window[1].data

if cmb.ncomp == 1:
    windowed_map.data = cmb.data * window.data

windowed_map.plot(file_name="%s/windowedmap_%s"%(test_dir, run), hp_gnomv=(lon, lat, 2000, 1))

#the windows (for spin0 and spin2) are going to couple mode together, we compute a mode coupling matrix in order to undo this effect

mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window, binning_file, lmax=lmax, type="Dl", niter=niter)
if write_data == True:
    mbb_inv_array = so_mcm.coupling_dict_to_array(mbb_inv)
    np.savetxt("%s/mbb_inv_%s.dat" % (test_dir, run), mbb_inv_array)

# %% MARK: Sph. Trans. and bin
#then we compute the spherical transform of the splits X windows

print (">Compute alms and bin spectra")

almList = []
for s in splitlist:
    almList += [sph_tools.get_alms(s, window, niter, lmax)]

pbar.update(10)

Db_dict = {}
spec_name_list = []
for name1, alm1, c1  in zip(nameList, almList, np.arange(n_splits)):
    for name2, alm2, c2  in zip(nameList, almList, np.arange(n_splits)):
        if c1 > c2: continue
        #we get the unbin spectra, it has modes correlation due to the window function
        l, ps = so_spectra.get_spectra(alm1, alm2, spectra=spectra)
        spec_name = "%sx%s" % (name1, name2)
        #we bin the spectra and apply the inverse of the mode coupling matrix
        lb, Db_dict[spec_name] = so_spectra.bin_spectra(l,
                                                        ps,
                                                        binning_file,
                                                        lmax,
                                                        type="Dl",
                                                        mbb_inv=mbb_inv,
                                                        spectra=spectra)
        spec_name_list += [spec_name]
        #we write the spectra in a file
        if write_data == True:
            so_spectra.write_ps("%s/spectra_%s_%s.dat" % (test_dir, spec_name, run),
                                lb,
                                Db_dict[spec_name],
                                type="Dl",
                                spectra=spectra)

pbar.update(20)

# %% MARK: Theory compare
print(">Compute spectra")

#read a theory spectra, bin it, and compare to the data

l, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)
ps_theory_b = so_mcm.apply_Bbl(Bbl, ps_theory, spectra=spectra)

plt.figure(figsize=(20,15))
for c,f in enumerate(spectra):
    plt.subplot(3, 3, c+1)
    for spec_name in spec_name_list:
        plt.plot(lb, Db_dict[spec_name][f], "o", label="%s" % spec_name)
    plt.plot(lb, ps_theory_b[f], color="lightblue", label="binned theory")
    plt.plot(l, ps_theory[f], color="grey", label="theory")
    plt.ylabel(r"$D^{%s}_{\ell}$" % f, fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    if c == 0:
        plt.legend()
plt.savefig("%s/spectra_%s.png" % (test_dir, run), bbox_inches="tight")
plt.clf()
plt.close()

spectra = ["TT"]
plt.figure(figsize=(10,8))
for c,f in enumerate(spectra):
    for spec_name in spec_name_list:
        plt.plot(lb, Db_dict[spec_name][f], "o", label="%s" % spec_name)
    plt.plot(lb, ps_theory_b[f], color="lightblue", label="binned theory")
    plt.plot(l, ps_theory[f], color="grey", label="theory")
    plt.ylabel(r"$D^{%s}_{\ell}$" % f, fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    if c == 0:
        plt.legend()
plt.savefig("%s/TTspectra_%s.png" % (test_dir, run), bbox_inches="tight")
plt.clf()
plt.close()


plt.figure(figsize=(10,8))
for c,f in enumerate(spectra):
    for spec_name in spec_name_list:
        plt.plot(lb, Db_dict[spec_name][f], "o", label="%s" % spec_name)
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
pbar.update(10)

pbar.close()