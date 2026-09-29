# A simple script to make a schedule for a given SO LAT Observation

import toast
import os
import astropy.units as u
from toast import Telescope
import h5py

from toast.schedule_sim_ground import run_scheduler

def load_from_hdf5(file_path, class_type):

    """
    Load a TOAST object (like a Telescope, Focalplane, or GroundSite) from an HDF5 file.
    """
    
    with h5py.File(file_path, 'r') as f:
        group = f[list(f.keys())[0]]
        obj = class_type.load_hdf5(group)
    
    return obj

# Load the telescope object
path_to_tel = "toast_tutorial/telescope.hdf5"
if os.path.exists(path_to_tel):
    print(f"Loading telescope from {path_to_tel}")
    # if the telescope hdf5 file is provided, load it
    telescope = load_from_hdf5(path_to_tel, Telescope)
else:
    raise FileNotFoundError("Please specify a valid path to the telescope object.")

ra = 290 * u.deg
dec = -60 * u.deg
width = 10 * u.deg


# Define the output file for our schedule 
sch_file = os.path.join("test_schedule.txt")

# This functions takes in the various command line arguments to design a 
# schedule that meets the needs of the observation
# It then outputs the designed schedule as a txt file
run_scheduler(
    opts=[
        "--site-name", telescope.site.name,
        "--telescope", telescope.name,
        "--site-lon", "{}".format(telescope.site.earthloc.lon.to_value(u.degree)),
        "--site-lat", "{}".format(telescope.site.earthloc.lat.to_value(u.degree)),
        "--site-alt", "{}".format(telescope.site.earthloc.height.to_value(u.meter)),
        "--patch", "test,1,{},{},{}".format(ra.value, dec.value, width.value),
        "--start", "2026-09-01 00:00:00",
        "--stop", "2026-09-02 12:00:00",
        "--out", sch_file,
        "--equalize-time",
        "--patch-coord", "C", # C for Equatorial (RA-Dec), E for Ecliptic (Lon Lat), and G for Galactic
        "--el-min", "20",
        "--el-max", "50",
        "--sun-el-max", "90",
        "--sun-avoidance-angle", "30",
        "--moon-avoidance-angle", "0",
        "--ces-max-time", "36000",
        "--fp-radius", "0",
        # "--boresight-angle-step", "180",
        # "--boresight-angle-time", "1440",
        "--time-step-s", "600",
        "--lock-az-range",
        "--elevations", "25, 30, 35, 40, 45",
    ]
)

