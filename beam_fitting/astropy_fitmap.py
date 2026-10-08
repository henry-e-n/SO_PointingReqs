"""
Astropy Fitmap

Author : Henry Nachman
---

This script fits a 2D Gaussian to a source map using regular astropy methods.
"""
# %% Imports
import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patches as patches
import astropy.units as u
from pixell import enmap
from astropy.wcs import WCS


def fit_beam_gaussian(image_data):
    """
    Fits a 2D Gaussian to an image and extracts beam statistics.
    Assumes image_data is a 2D numpy array (background-subtracted).
    """
    # 1. Create a coordinate grid for the image
    print(image_data.shape)
    yp, xp = image_data.shape
    y, x = np.mgrid[:yp, :xp]
    
    # 2. Estimate robust initial guesses from the data
    amplitude_guess = np.max(image_data)
    y_idx, x_idx = np.unravel_index(np.argmax(image_data), image_data.shape)
    
    # Estimate standard deviation (rough guess based on size or dispersion)
    # Adjust this based on your expected beam size if needed
    stddev_guess = max(1.0, np.std(image_data > (amplitude_guess * 0.1)))
    
    # 3. Initialize the 2D Gaussian model
    # theta is counterclockwise rotation angle from the X-axis
    gauss_init = models.Gaussian2D(
        amplitude=amplitude_guess,
        x_mean=float(x_idx),
        y_mean=float(y_idx),
        x_stddev=stddev_guess,
        y_stddev=stddev_guess,
        theta=0.0
    )
    
    # 4. Set up the fitter
    fitter = fitting.TRFLSQFitter() # Trust Region Reflective Least Squares
    
    # 5. Perform the fit
    fitted_model = fitter(gauss_init, x, y, image_data)
    
    # 6. Extract beam statistics
    # Convert standard deviations to Full Width at Half Maximum (FWHM)
    fwhm_x = fitted_model.x_stddev.value * gaussian_sigma_to_fwhm
    fwhm_y = fitted_model.y_stddev.value * gaussian_sigma_to_fwhm
    
    # Identify major vs minor axes based on which width is larger
    if fwhm_x >= fwhm_y:
        fwhm_major = fwhm_x
        fwhm_minor = fwhm_y
        pa = np.degrees(fitted_model.theta.value) % 180
    else:
        fwhm_major = fwhm_y
        fwhm_minor = fwhm_x
        # If axes are flipped, position angle shifts by 90 degrees
        pa = (np.degrees(fitted_model.theta.value) + 90) % 180

    # Compile results into a clean dictionary
    beam_stats = {
        "amplitude": fitted_model.amplitude.value,
        "x_center": fitted_model.x_mean.value,
        "y_center": fitted_model.y_mean.value,
        "fwhm_major_pix": fwhm_major,
        "fwhm_minor_pix": fwhm_minor,
        "position_angle_deg": pa,
        "model_object": fitted_model
    }
    
    return beam_stats

# %%
# Generate a mock skewed beam with noise

maps_to_load = [
    "/shared_home/henachman/data/sim_outputs/simple_hr10arcmin/mapmaker/mlmapmaker_sky_map.fits"
]

with fits.open(maps_to_load[0]) as hdul1:
    data1 = hdul1[0].data
    # 2. Extract WCS from the primary header (or specific HDU, e.g., hdulist[1].header)
    wcs = WCS(hdul1[0].header)
    print(wcs)


true_beam = models.Gaussian2D(
    amplitude=10, x_mean=25, y_mean=23, 
    x_stddev=5, y_stddev=3, theta=np.radians(35)
)
# Run the fitter
results = fit_beam_gaussian(data1[0])

# Print out beam parameters
print("--- Fitted Beam Statistics ---")
print(f"Peak Amplitude: {results['amplitude']:.3f}")
print(f"Center (X, Y):  ({results['x_center']:.2f}, {results['y_center']:.2f})")
print(f"FWHM Major:     {results['fwhm_major_pix']:.2f} pixels")
print(f"FWHM Minor:     {results['fwhm_minor_pix']:.2f} pixels")
print(f"Position Angle: {results['position_angle_deg']:.1f}°")


fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data1[0], origin='lower', cmap='seismic', vmin=-5e-3, vmax=5e-3)

# Plot an ellipse based on the beam results
ellipse = patches.Ellipse(
    xy=(results['x_center'], results['y_center']), width=results['fwhm_major_pix'], height=results['fwhm_minor_pix'], angle=0, edgecolor="yellow", fc="yellow", lw=2
)

# Add the patch to the axes
ax.add_patch(ellipse)

plt.colorbar(im, ax=ax, label='Difference (Map 1 - Map 2)')
plt.title('FITS Map Difference')
plt.xlim(100,300)
plt.ylim(100, 300)
plt.show()

# %%
# Function stolen from 
# https://github.com/simonsobs/mf-cmg-paper-beams/blob/main/abscal_pipeline/abscal_utils.py
def calc_rad_profile(map, binsize=0.5, normalize=True, positive_only=False):
    """Fucntion to calculate the radial profile of an input beam map.
    Assumes that the beam center is at the center of the map. 

    Args:
        map (ndmap): 2D ndmap to profile
        binsize (float, optional): radial bin size
        normalize (bool, optional): Normalizes profile to center pixel
    Returns:
        rad_prof (dictionary): Keys of 'radius', 'profile', and 'bin_stdev'
    """
    rad_prof = {'radius':[], 'profile':[],'bin_stdev':[]}    
    rad = np.array([])
    amps = np.array([])
    stdevs = np.array([])

    # setup
    pos = map.posmap()
    ra = pos[1]
    dec= pos[0]    
    r = np.sqrt(ra**2 + dec**2) # radians
    r = r*u.rad.to(u.arcmin) # convert to arcmin
    r_max = np.nanmax(r)
    r_bins= np.arange(0, r_max, binsize)
    
    # take radial average
    if positive_only:
        for i in range(len(r_bins)-1):
            idx = (r > r_bins[i]) & (r < r_bins[i+1]) & (map >= 0)
            # idx = (r > r_bins[i]) & (r < r_bins[i+1])
            ravg = np.nanmean(map[idx])
            vars = np.nanstd(map[idx])
            
            rad = np.append(rad,(r_bins[i] + r_bins[i+1])/2)
            amps = np.append(amps,ravg)
            stdevs = np.append(stdevs, vars)
    else:
        for i in range(len(r_bins)-1):
            # idx = (r > r_bins[i]) & (r < r_bins[i+1]) & (map >= 0)
            idx = (r > r_bins[i]) & (r < r_bins[i+1])
            ravg = np.nanmean(map[idx])
            vars = np.nanstd(map[idx])
            
            rad = np.append(rad,(r_bins[i] + r_bins[i+1])/2)
            amps = np.append(amps,ravg)
            stdevs = np.append(stdevs, vars)
    
    
    #take peak for normalization as maximum pixel value within 2 arcmin of r=0, so as to avoid degeneracy with FWHM in fitting
    peakidx = r < 2
    peak = np.nanmean(map[peakidx])
    if normalize:
        amps /= peak
        amps= np.append(np.array([1]),amps)
        rad = np.append(np.array([0]), rad)
    
    rad_prof['radius'] = rad
    rad_prof['profile']   = amps
    rad_prof['bin_stdev']= stdevs

    return rad_prof

# %% 
data_ndmap = enmap.read_map(maps_to_load[0])
rad_prof = calc_rad_profile(data_ndmap[0], positive_only=True)

# %%
print(rad_prof)
print(rad_prof.keys())
print(rad_prof['radius'])
print(rad_prof['profile'])

plt.plot(rad_prof['radius'], rad_prof['profile'])
