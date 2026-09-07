# 

# Date : 03 September 2026

# This is a tutorial for using TOAST via MPI on SO-UK

# Optionally change logging level
import os
os.environ["TOAST_LOGLEVEL"] = "INFO"
# This is needed before importing toast, and should
# match the value passed to the '-t' option of %toast
# os.environ["OMP_NUM_THREADS"] = "8"

# Imports

import numpy as np
import astropy.units as u
from astropy.table import QTable
import pandas as pd
import os
from time import time

import toast
from toast.tests import helpers
import inspect
from plotting_utils import *
import healpy as hp

# Start the timer
time_start = time()


out_dir = 'out_sim_ground'

# MPI communicator
world, procs, rank = toast.mpi.get_world()
comm = helpers.create_comm(world, single_group=True)

env = toast.Environment.get()
print(env)

# MARK: Make a Telescope

site = toast.instrument.GroundSite("atacama", "-22:57:30", "-67:47:10", 5200.0 * u.meter)

spacesite = toast.instrument.SpaceSite("HNSpaceTelescope")

from toast.instrument_sim import (
    fake_rhombihex_focalplane,
    fake_boresight_focalplane,
    fake_hexagon_focalplane,
    plot_focalplane
)

fp_fwhm = 30.0 * u.arcmin

focalplane = fake_rhombihex_focalplane(
    n_pix_rhombus=25,# Number of pixels per rhombus must be a perfect square
    width=8.0 * u.degree, # field of view of the focal plane
    gap=0 * u.radian,
    sample_rate=10.0 * u.Hz,
    epsilon=0.0,
    fwhm=fp_fwhm, # Beam size
    bandcenter=150 * u.GHz,
    bandwidth=20 * u.GHz,
    psd_net=300.0 * u.uK * np.sqrt(1 * u.second),
    psd_fmin=1.0e-5 * u.Hz,
    psd_alpha=1.0,
    psd_fknee=0.05 * u.Hz,
    fwhm_sigma=0.0 * u.arcmin,
    bandcenter_sigma=0 * u.GHz,
    bandwidth_sigma=0 * u.GHz,
    random_seed=123456,
)
fov = focalplane.field_of_view

# We will define colors for our two polarization angles
detpolcol = {
    x: "red" if "A" in x else "blue" for x in focalplane.detectors
}

# Plot the focal plane
_ = plot_focalplane(
    focalplane=focalplane,
    width=1.3 * fov,
    height=1.3 * fov,
    show_labels=True,
    pol_color=detpolcol
)

telescope = toast.Telescope(name="HenryTelescope", focalplane = focalplane, site = site)

# MARK: DATA
data = toast.Data(comm)

sch_file = os.path.join(out_dir, "test_schedule.txt")

schedule = toast.schedule.GroundSchedule()
schedule.read(sch_file)

# Populate observations according to the schedule and telescope.
sim_ground = toast.ops.SimGround(
    telescope=telescope,
    weather="atacama",
    detset_key="pixel",
    schedule=schedule,
    median_weather=True, # no longer random weather, but less chance of an outlier
)

sim_ground.apply(data)

if rank == 0:
    plot_scanning(data.obs[0], s_start=0, s_end=2000)


det_point_azel = toast.ops.PointingDetectorSimple(
    boresight=defaults.boresight_azel,
    quats="quats_azel"
)

det_point_radec = toast.ops.PointingDetectorSimple(
    boresight=defaults.boresight_radec,
    quats="quats_radec"
)

# Pixelization.  Choose a coarse pixelization for this exercise since there is
# a small patch and only a few detectors.

nside = 256
pixels_radec = toast.ops.PixelsHealpix(
    nside=nside,
    nest=True,
    detector_pointing=det_point_radec,
)


# Stokes weights.  This just uses focalplane table properties to treat each detector
# as a linear polarizer with possibly some cross-polar response.

weights_azel = toast.ops.StokesWeights(
    mode="IQU",
    detector_pointing=det_point_azel,
)

weights_radec = toast.ops.StokesWeights(
    mode="IQU",
    detector_pointing=det_point_radec,
)

pixels_radec.apply(data)
weights_radec.apply(data)

pix_dist = toast.ops.BuildPixelDistribution(
    pixel_dist="pixel_dist",
    pixel_pointing=pixels_radec,
)
pix_dist.apply(data)

# MARK: Sky MAP
input_map_file = os.path.join(out_dir, "fake_sky.fits")


# Scan the map
scan_map = toast.ops.ScanHealpixMap(
    file=input_map_file,
    pixel_pointing=pixels_radec,
    stokes_weights=weights_radec,
)
scan_map.apply(data)

ob = data.obs[0]
if rank == 0:
    plot_dets(ob, d_start=140, d_end=None, s_start=0, s_end=2000, view="scanning")



nominal_noise = toast.ops.DefaultNoiseModel()
nominal_noise.apply(data)

# Plot this nominal noise model for the last few detectors
if rank == 0:
    plot_noise_model(
        data.obs[0][nominal_noise.noise_model],
        model_fit=None,
        d_start=100,
        d_end=None
    )

sim_noise = toast.ops.SimNoise(
    noise_model=nominal_noise.noise_model,
)
sim_noise.apply(data)

ground_pickup = toast.ops.SimScanSynchronousSignal(
    detector_pointing=det_point_azel,
    scale=0.001 * u.K,
    stokes_weights=weights_azel,
)
ground_pickup.apply(data)

sim_atm = toast.ops.SimAtmosphere(
    detector_pointing=det_point_azel,
    add_loading=True,
    lmin_center=0.001 * u.m,
    lmin_sigma=0.0001 * u.m,
    lmax_center=1.0 * u.m,
    lmax_sigma=0.1 * u.m,
    xstep=20 * u.m,
    ystep=20 * u.m,
    zstep=20 * u.m,
    zmax=200 * u.m,
    gain=4e-5,
    wind_dist=1000 * u.m,
)
sim_atm.apply(data)


# Plot the last few normal detectors
if rank == 0:
    plot_dets(data.obs[0], d_start=90, d_end=96, s_start=0, s_end=2000, view="scanning")

# Stop the timer
time_end = time()
print(f"Time to complete with {procs} processes : {time_end-time_start}")