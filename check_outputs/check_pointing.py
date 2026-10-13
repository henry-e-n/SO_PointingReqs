# %% Imports
import numpy as np
import h5py as h5
import os
import matplotlib.pyplot as plt
import toast.qarray as qa

import matplotlib
matplotlib.style.use("/shared_home/henachman/.matplotlib/stylelib/HNCustomMono.mplstyle")
# %%
path_to_data = "/shared_home/henachman/data/sim_outputs/Mars/"
job_name = "Mars_2arcmin_f090_20261010_231649"
data_path = os.path.join(path_to_data, job_name, "data", "Mars-0-0_w16.h5")

offset_job_name = "Mars_bigoffset_5arcmin_f090_20261012_221848"
offset_data_path = os.path.join(path_to_data, offset_job_name, "data", "Mars-0-0_w16.h5")


with h5.File(data_path, 'r') as f:
    print(f.keys())
    print(f["shared"].keys())
    boresight_arr = np.array(f["shared"]["boresight_azel"])

with h5.File(offset_data_path, 'r') as f:
    print(f.keys())
    print(f["shared"].keys())
    offset_arr = np.array(f["shared"]["boresight_azel_offset"])

# %%
slc = slice(0, 1000, 10)

bquat = np.array(boresight_arr[slc, :])
qbang = np.zeros((3, bquat.shape[0]), dtype=np.float64)
qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(bquat)
qbang[0] *= 180.0 / np.pi
qbang[1] *= 180.0 / np.pi

oquat = np.array(offset_arr[slc, :])
obang = np.zeros((3, oquat.shape[0]), dtype=np.float64)
obang[0], obang[1], obang[2] = qa.to_lonlat_angles(oquat)
obang[0] *= 180.0 / np.pi
obang[1] *= 180.0 / np.pi


fig, axs = plt.subplots(1, 2, figsize=(16,8))
axs[0].plot(qbang[0], label="Nominal")
axs[0].plot(obang[0], label="Offset")
axs[0].legend()
axs[0].set_xlabel("Time Index")
axs[0].set_ylabel("Lon (deg)")

axs[1].plot(qbang[1])
axs[1].plot(obang[1])
axs[1].set_xlabel("Time Index")
axs[1].set_ylabel("Lat (deg)")
plt.show()
# %%
