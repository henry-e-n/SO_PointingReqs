import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py

matplotlib.style.use("/shared_home/henachman/.matplotlib/stylelib/HNCustomMono.mplstyle")

def gnomview_auto_center(data_map, hits_map, **kwargs):
    """
    Plots a healpy gnomview centered automatically on the mean position of hits.
    """
    # 1. Get the NSIDE from the map
    nside = hp.get_nside(hits_map)
    
    # 2. Find the pixel indices where hits are greater than zero
    # If using UNSEEN markers instead, change to: hits_map != hp.UNSEEN
    valid_pix = np.where(hits_map > 0)[0]
    
    if len(valid_pix) == 0:
        raise ValueError("The hits map contains no valid hit pixels.")
        
    # 3. Convert valid pixel indices to spherical coordinates (theta, phi)
    theta, phi = hp.pix2ang(nside, valid_pix)
    
    # 4. Convert spherical coordinates to Cartesian vectors to safely average them
    # (Averaging angles directly can break near the 0/360 boundary)
    vecs = hp.ang2vec(theta, phi)
    mean_vec = np.mean(vecs, axis=0)
    
    # 5. Convert the mean vector back to coordinates (Lon/Lat in degrees)
    mean_theta, mean_phi = hp.vec2ang(mean_vec)
    
    lon = np.degrees(mean_phi)
    lat = 90.0 - np.degrees(mean_theta)
    
    # 6. Pass the computed center to the rotation parameter 'rot'
    # hp.gnomview(data_map, rot=[lon[0], lat[0], 0], **kwargs)
    return [lon[0], lat[0]]

# --- Example Usage ---
def load_fits_map(path_to_job, path_to_data, hits=False, **kwargs):
    path_to_map = f"{path_to_data}{path_to_job}/filterbin_filtered_map.fits"
    hpmap = hp.read_map(path_to_map)
    if hits==True:
        hitmap_path = f"{path_to_data}{path_to_job}/filterbin_filtered_hits.fits"
        hp_hitmap = hp.read_map(hitmap_path)
        rot = gnomview_auto_center(hpmap, hp_hitmap, **kwargs)
        return hpmap, hp_hitmap, rot
    return hpmap

def load_h5_map(path_to_job, path_to_data, hits=True):
    path_to_map = f"{path_to_data}{path_to_job}/filterbin_filtered_map.h5"
    with h5py.File(path_to_map, "r") as f:
        dset = f["map"]
        # nside = int(dset.attrs["NSIDE"])
        # nest = dset.attrs["ORDERING"] == "NESTED"
        I_map, Q_map, U_map = dset[0], dset[1], dset[2]          # stokes I; use dset[1], dset[2] for Q, U

    for map in [I_map, Q_map, U_map]:
        map = map.astype(float)
        map[map == 0] = hp.UNSEEN    # blank the unhit pixels

    hits_map = None
    if hits==True:
        hitmap_path = f"{path_to_data}{path_to_job}/filterbin_filtered_hits.h5"
        with h5py.File(hitmap_path, "r") as f:
            dset = f["map"]
            hits_map = dset[0]
    return I_map, Q_map, U_map, hits_map



def diff_plot(map1, map2, rot, xsize=1000, reso=0.05, cmap='RdBu_r', deg=True, show=True, **kwargs):
    map1_gnom = hp.gnomview(map1, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["C"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True,
        nest=True
    )
    map2_gnom = hp.gnomview(map2, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["C"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True,
        nest=True
        )

    x_size, y_size = np.shape(map1_gnom)
    x_ticks = np.linspace(0, x_size, 7)
    y_ticks = np.linspace(0, y_size, 7)
    def pix_to_coord(pix, reso, x_size, deg):
        if deg:
            return (pix - x_size//2)*reso/60
        else:
            return (pix - x_size//2)*reso

    if deg:
        x_labels = np.round(rot[0]+pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(rot[1]+pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + r"$^\circ$" for i in x_labels[::-1]]
        y_labels_str = [str(i) + r"$^\circ$" for i in y_labels]
    else:
        x_labels = np.round(pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + "'" for i in x_labels[::-1]]
        y_labels_str = [str(i) + "'" for i in y_labels]

    map1_gnom[map1_gnom == hp.UNSEEN] = np.nan
    map2_gnom[map2_gnom == hp.UNSEEN] = np.nan

    # 5. Subtract the two 2D maps
    subtracted_grid = map1_gnom - map2_gnom

    # 6. Plot the final result using standard matplotlib
    fig = plt.figure(figsize=(8,20))
    subfigs = fig.subfigures(3,1)
    # Truth
    subfig_top = subfigs[0]
    ax_top = subfig_top.subplots()
    im_top = ax_top.imshow(map1_gnom, origin='lower', cmap=cmap, **kwargs)
    subfig_top.colorbar(im_top, ax=ax_top, label='Difference Intensity')
    ax_top.set_title("Map 1")
    def set_axes(ax):
        ax.set_xticks(x_ticks, labels=x_labels_str, rotation=45)
        ax.set_yticks(y_ticks, labels=y_labels_str, rotation=45)
        if deg:
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
        else:
            ax.set_xlabel(r"$\delta_{RA}$ (arcmin)")
            ax.set_ylabel(r"$\delta_{Ddec}$ (arcmin)")
    set_axes(ax_top)

    # Output Map
    subfig_mid = subfigs[1]
    ax_mid = subfig_mid.subplots()
    im_mid = ax_mid.imshow(map2_gnom, origin='lower', cmap=cmap, **kwargs)
    subfig_mid.colorbar(im_mid, ax=ax_mid, label='Difference Intensity')
    ax_mid.set_title("Map 2")
    set_axes(ax_mid)
    # Difference
    subfig_bottom = subfigs[2]
    ax_bottom = subfig_bottom.subplots()
    im_bottom = ax_bottom.imshow(subtracted_grid, origin='lower', cmap=cmap, **kwargs)
    subfig_bottom.colorbar(im_bottom, ax=ax_bottom, label='Difference Intensity')
    ax_bottom.set_title("Difference Map\nMap 1 - Map 2")
    set_axes(ax_bottom)
    if show:
        plt.show()
    axs = [ax_top, ax_mid, ax_bottom]
    return fig, axs



def single_plot(hpmap, rot, xsize=1000, reso=0.05, cmap="RdBu_r", deg=True, show=True, hits=None, **kwargs):
    if hits is not None:
        hpmap[np.where(hits==0)] = hp.UNSEEN
   
    map1_gnom = hp.gnomview(hpmap, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["C"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True,
        nest=True,
        badcolor='gray'
    )
    x_size, y_size = np.shape(map1_gnom)
    x_ticks = np.linspace(0, x_size, 7)
    y_ticks = np.linspace(0, y_size, 7)
    def pix_to_coord(pix, reso, x_size, deg):
        if deg:
            return (pix - x_size//2)*reso/60
        else:
            return (pix - x_size//2)*reso

    if deg:
        x_labels = np.round(rot[0]+pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(rot[1]+pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + r"$^\circ$" for i in x_labels[::-1]]
        y_labels_str = [str(i) + r"$^\circ$" for i in y_labels]
    else:
        x_labels = np.round(pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + "'" for i in x_labels[::-1]]
        y_labels_str = [str(i) + "'" for i in y_labels]

    map1_gnom[map1_gnom == hp.UNSEEN] = np.nan
    
    # 6. Plot the final result using standard matplotlib
    fig = plt.figure(figsize=(12,10))
    subfig_top = fig.subfigures(1,1)
    
    ax_top = subfig_top.subplots()
    im_top = ax_top.imshow(map1_gnom, origin='lower', cmap=cmap, **kwargs)
    subfig_top.colorbar(im_top, ax=ax_top, label='Difference Intensity')
    ax_top.set_title("Map 1")
    def set_axes(ax):
        ax.set_xticks(x_ticks, labels=x_labels_str, rotation=45)
        ax.set_yticks(y_ticks, labels=y_labels_str, rotation=45)
        if deg:
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
        else:
            ax.set_xlabel(r"$\delta_{RA}$ (arcmin)")
            ax.set_ylabel(r"$\delta_{Ddec}$ (arcmin)")
    set_axes(ax_top)
    if show:
        plt.show()
    
    return map1_gnom, fig

def get_center(map):
    """
    Finds the ra and dec of the pixel with the max value in the map
    Should be used to center an SSO / source map.

    Note: This assumes that the target source is the brightest thing in the map.


    Arguments:
    map: Intensity map
    
    Returns:
    list: [ra, dec] - ra and dec of max pixel
    """
    max_ind = np.argmax(map)

    # Get the nside of the map
    nside = hp.npix2nside(len(map))

    # Convert pixel index to spherical coord
    theta, phi = hp.pix2ang(nside, max_ind, nest=True)

    # Convert theta and phi rad to deg
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    return [ra, dec]

def get_mol_center(hits_map):
    # 1. Identify which pixels have hits (where hits > 0)
    # (If your map uses hp.UNSEEN for unhit pixels, use: valid_pix = (hits_map != hp.UNSEEN))
    valid_mask = hits_map > 0
    hit_pixels = np.where(valid_mask)[0]
    weights = hits_map[valid_mask]

    # 2. Get the resolution parameter (nside) of your map
    nside = hp.get_nside(hits_map)

    # 3. Convert the hit pixel indices into 3D Cartesian vectors (x, y, z)
    x, y, z = hp.pix2vec(nside, hit_pixels, nest=True)

    # 4. Calculate the weighted average vector to find the true centroid
    center_x = np.average(x, weights=weights)
    center_y = np.average(y, weights=weights)
    center_z = np.average(z, weights=weights)

    # 5. Convert the average vector back to angular coordinates
    # Set lonlat=True to get (Longitude, Latitude) in degrees directly
    lon, lat = hp.vec2ang(np.array([center_x, center_y, center_z]), lonlat=True)
    
    print(f"Center of hits map: {lon}°, {lat}°")
    rot = [lon[0], lat[0], 0]
    return rot

def centroid_center(map, rot_guess, max_pix=False):
    from photutils.centroids import centroid_com, centroid_2dg, centroid_quadratic

    gnommap = hp.gnomview(map, 
                rot=rot_guess,
                xsize=1000,
                reso=0.1,
                return_projected_map=True,
                nest=True
                # no_plot=True,
    )
                # no_plot=True)
    flat_idx = np.argmax(gnommap)
    row, col = divmod(flat_idx, gnommap.shape[1])
    x_size, y_size = np.shape(gnommap)

    def pix_to_coord(pix, reso=0.1, x_size=1000):
        return (pix - x_size//2)*reso/60
    
    if max_pix:
        rot_est = -pix_to_coord(np.array([row, col]))+np.array(rot_guess)


    box_size = 50
    x_cent, y_cent = centroid_2dg(gnommap[int(row)-box_size:int(row)+box_size, int(col)-box_size:int(col)+box_size])
    cent_pix = np.array([col + x_cent, row + y_cent]) # print(f"Centroid at: x = {col + x_cent}, y = {row + y_cent}")
    rot_est = pix_to_coord(cent_pix)+np.array(rot_guess)
    return rot_est

def get_radial_profile(proj_map, Num_pix, pix_size, show=False):
    patch_size = Num_pix*pix_size
    half = int(patch_size/ pix_size // 2)
    cx   = Num_pix // 2

    r_pix  = np.arange(0, half)
    r_amin = r_pix * pix_size

    # Take cuts through the centre pixel
    def radial_cut(beam, cx):
        return beam[cx, cx:cx + half] / beam[cx, cx]

    profile = radial_cut(proj_map, cx)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.plot(r_amin, radial_cut(proj_map, cx), label='Circular Gaussian', lw=2)
        ax.axhline(0.5, color='grey', ls='--', alpha=0.7, label='Half-power')
        ax.set_xlabel('Angular radius (arcmin)')
        ax.set_ylabel('Normalised beam power')
        ax.set_title('Radial Beam Profiles (linear)')
        ax.legend(fontsize=9)
        # ax.set_xlim(0, 5*fwhm_nom)
        ax.grid(True, alpha=0.3)
    
        ax = axes[1]
        ax.plot(r_amin, 10*np.log10(np.maximum(radial_cut(proj_map, cx), 1e-6)), label='Circular Gaussian', lw=2)
        ax.axhline(-3, color='grey', ls='--', alpha=0.7)
        ax.set_xlabel('Angular radius (arcmin)')
        ax.set_ylabel('Logarithmic beam power [dB]')
        ax.set_title('Radial Beam Profiles (log scale — sidelobes visible)')
        ax.legend(fontsize=9)
        # ax.set_xlim(0, 5*fwhm_nom)
        ax.set_ylim(-55, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
    return [r_amin, profile]
    # Radial beam profiles
    
# Dealing with Sparse Maps

def read_sparse(path):
    with h5py.File(path, "r") as f:
        d = f["map"]
        nside = int(d.attrs["NSIDE"])
        nest = d.attrs["ORDERING"] == "NESTED"
        n = d.chunks[1]
        pix, vals = [], []
        for i in range(d.id.get_num_chunks()):
            s = d.id.get_chunk_info(i).chunk_offset[1]
            block = d[:, s:s + n]                  # (3, n)
            hit = np.any(block != 0, axis=0)       # drop unhit pixels inside a chunk
            pix.append(s + np.flatnonzero(hit))
            vals.append(block[:, hit])
    return nside, nest, np.concatenate(pix), np.concatenate(vals, axis=1)


from healpy.projector import GnomonicProj

def sparse_gnomview(nside, nest, pix, vals, rot_est, xsize=600, reso=0.5, est_rot=False, **kw):
    order = np.argsort(pix)
    pix_s = pix[order]
    vals_s = np.append(vals[order].astype(float), np.nan)   # last slot = unhit
    n = pix_s.size
    ra, dec = rot_est

    rot = [ra, dec, 0]
    def lookup(x, y, z):
        p = hp.vec2pix(nside, x, y, z, nest=nest)
        j = np.clip(np.searchsorted(pix_s, p), 0, n - 1)
        pix_num = np.where(pix_s[j] == p, j, n).astype(np.int64)
        return pix_num   # integer index

    proj = GnomonicProj(rot=rot, xsize=xsize, ysize=xsize, reso=reso)
    img = proj.projmap(vals_s, lookup)

    if est_rot:
        flat_idx = np.argmax(img)
        row, col = divmod(flat_idx, img.shape[1])
        img
        def pix_to_coord(pix, reso=0.1, x_size=1000):
            return (pix - x_size//2)*reso/60

        rot_est = np.array([-1, 1, 1])*pix_to_coord(np.array([col, row, 0]), x_size=xsize, reso=reso)+np.array(rot)

        proj_cent = GnomonicProj(rot=rot_est, xsize=xsize, ysize=xsize, reso=reso)    
        img = proj_cent.projmap(vals_s, lookup)

    plt.imshow(img, origin="lower", **kw)
    plt.colorbar()
    return img
