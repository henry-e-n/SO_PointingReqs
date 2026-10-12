# %% Imports
import numpy as np
import h5py as h5
import os
import matplotlib.pyplot as plt

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
plt.plot(boresight_arr[:100])
plt.show()