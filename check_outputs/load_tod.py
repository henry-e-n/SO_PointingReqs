# %% 
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import h5py as h5

path_to_tod = "/cephfs/soukdata/user_data/henachman/sim_outputs"
sim_path = "trial_f090_i1_Mars"

full_path = os.path.join(path_to_tod, sim_path)

if not os.path.exists(full_path):
    print(f"Path {full_path} does not exist. Exiting.")
    raise FileNotFoundError(f"Path {full_path} does not exist.")

data_path = os.path.join(full_path, "data")
# %%

# List the h5 files
h5_files = []
for file in os.listdir(data_path):
    if file.endswith(".h5"):
        h5_files.append(file)

print(h5_files)

# Load the first h5_file
with h5.File(os.path.join(data_path, h5_files[0]), 'r') as f:
    print(f.keys())
    detdata = f['detdata']
    signal = detdata['signal'][:]
    n_det, n_samp = signal.shape
    print(n_det, n_samp)
    print(f['shared'].keys())

# %%

plt.plot(signal[:10, :200])
plt.show()

# %% Pointing Visualizations

# boresight_pointing_3dvis(jitter_boreradec, init_boreradec, ob, n_slices=30, step_size=10, rad_threshold=0.15, show_dets=True, out_dir=out_dir)
# boresight_pointing_residualplot(jitter_boreradec, init_boreradec, n_slices=30, step_size=10, out_dir=out_dir)
