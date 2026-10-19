import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
def load_map(path_to_job, path_to_data, hits=False, **kwargs):
    path_to_map = f"{path_to_data}{path_to_job}/filterbin_filtered_map.fits"
    hpmap = hp.read_map(path_to_map)
    if hits==True:
        hitmap_path = f"{path_to_data}{path_to_job}/filterbin_filtered_hits.fits"
        hp_hitmap = hp.read_map(hitmap_path)
        rot = gnomview_auto_center(hpmap, hp_hitmap, **kwargs)
        return hpmap, hp_hitmap, rot
    return hpmap

def diff_plot(map1, map2, rot, xsize=1000, reso=0.05, cmap='RdBu_r', deg=True, show=True, **kwargs):
    map1_gnom = hp.gnomview(map1, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["G"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True
    )
    map2_gnom = hp.gnomview(map2, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["G"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True
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
        x_labels_str = [str(i) + r"$^\circ$" for i in x_labels]
        y_labels_str = [str(i) + r"$^\circ$" for i in y_labels]
    else:
        x_labels = np.round(pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + "'" for i in x_labels]
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

def single_plot(hpmap, rot, xsize=1000, reso=0.05, cmap="RdBu_r", deg=True, show=True, **kwargs):   
    map1_gnom = hp.gnomview(hpmap, 
        rot=rot, 
        xsize=xsize,
        reso=reso,
        coord=["G"],
        cmap=cmap,
        return_projected_map=True,
        no_plot=True
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
        x_labels_str = [str(i) + r"$^\circ$" for i in x_labels]
        y_labels_str = [str(i) + r"$^\circ$" for i in y_labels]
    else:
        x_labels = np.round(pix_to_coord(x_ticks, reso, x_size, deg), 2)
        y_labels = np.round(pix_to_coord(y_ticks, reso, y_size, deg), 2) 
        x_labels_str = [str(i) + "'" for i in x_labels]
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
    
    return fig
