# Functions to visualize changes to pointing during a TOAST sim.
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import toast.qarray as qa
import matplotlib

path_to_style = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/.matplotlib/stylelib/HNCustomMono.mplstyle"
if os.path.exists(path_to_style):
    matplotlib.style.use(path_to_style)

def calc_radius(coordinate, center_coord):
    return np.sqrt((coordinate[:, 0]-center_coord[0])**2 + (coordinate[:, 1]-center_coord[1])**2 )

def boresight_pointing_3dvis(jitter_boreradec, init_boreradec, ob, n_slices=30, step_size=10, rad_threshold=0.15, show_dets=False, out_dir=""):
    boresight_coords = np.zeros((n_slices, 3))
    init_boresight_coords = np.zeros((n_slices, 3))
    
    fig = go.Figure()
    # 3D figure made with the help of Claude AI
    for i in range(n_slices):
        slc = slice(step_size * i, step_size * (i+1), step_size)

        bquat = np.array(jitter_boreradec[slc, :])
        qbang = np.zeros((3, bquat.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(bquat)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi

        boresight_coords[i] = np.array([qbang[0][0], qbang[1][0], step_size * i], dtype=float)

        bquat = np.array(init_boreradec[slc, :])
        qbang = np.zeros((3, bquat.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(bquat)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi

        init_boresight_coords[i] = np.array([qbang[0][0], qbang[1][0], step_size * i], dtype=float)


        # Take only the first index
        all_dets = ob.local_detectors
        dets_quats = np.array([ob.detdata["quats_radec"][d][0, :] for d in all_dets])
        det_lon, det_lat, det_psi = qa.to_lonlat_angles(dets_quats)
        det_coords = np.column_stack((det_lon, det_lat))
        # Assume the mid lon/lat is the average of the min and max
        center_coord = (
            np.mean((np.max(det_lon), np.min(det_lon))),
            np.mean((np.max(det_lat), np.min(det_lat))),
        )
        det_radii = calc_radius(det_coords, center_coord)
        max_dist = np.max(det_radii)

        dets = np.array(ob.local_detectors)
        far_dets = dets[det_radii > rad_threshold * max_dist]

        qdet = np.array([ob.detdata["quats_radec"][d][slc, :] for d in far_dets])

        n_qdet = len(qdet)
        n_samp = len(qdet[0])
        qdang = np.zeros((n_qdet, 3, n_samp), dtype=np.float64)
        for det in range(n_qdet):
            qdang[det, 0], qdang[det, 1], qdang[det, 2] = qa.to_lonlat_angles(qdet[det])
            qdang[det, 0] *= 180.0 / np.pi
            qdang[det, 1] *= 180.0 / np.pi


        # flatten all far-detectors' points for this slice into a single trace
        xs = qdang[:, 0, 0]#.ravel()
        ys = qdang[:, 1, 0]#.ravel()
        zs = np.full_like(xs, step_size * i)

        if show_dets:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=2,
                        symbol='diamond',
                        opacity=0.5,
                        color=zs,  # color by time (slice index)
                        colorscale="Blues",
                        cmin= step_size,
                        cmax= step_size * (n_slices),
                        # showscale=(i == 0),  # only draw one colorbar, not one per slice
                        # colorbar=dict(title="Time (slice)", x=1.02) if i == 0 else None,
                    ),
                    name=f"slice {i}",
                    showlegend=False,
                )
            )

    bore_x, bore_y, bore_z = boresight_coords[:, 0], boresight_coords[:, 1], boresight_coords[:, 2]
    fig.add_trace(
        go.Scatter3d(
            x=bore_x,
            y=bore_y,
            z=bore_z,
            mode="lines+markers",
            line=dict(color="blue", width=4),  # line itself can't take a colorscale
            marker=dict(
                size=4,
                color=bore_z,  # color by time (slice index)
                colorscale="viridis",
                cmin=0,
                cmax= step_size * n_slices - 1,
                # showscale=True,
                # colorbar=dict(title="Boresight time", x=1.15),
            ),
            name="Jitter Boresight",
        )
    )

    init_bore_x, init_bore_y, init_bore_z = init_boresight_coords[:, 0], init_boresight_coords[:, 1], init_boresight_coords[:, 2]
    fig.add_trace(
        go.Scatter3d(
            x=init_bore_x,
            y=init_bore_y,
            z=init_bore_z,
            mode="lines+markers",
            line=dict(color="indianred", width=4),  # line itself can't take a colorscale
            marker=dict(
                size=4,
                color=init_bore_z,  # color by time (slice index)
                colorscale="Reds",
                cmin=0,
                cmax= step_size * n_slices - 1,
                # showscale=True,
                # colorbar=dict(title="Boresight time", x=1.15),
            ),
            name="Init Boresight",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude (deg)",
            yaxis_title="Latitude (deg)",
            zaxis_title="Slice index",
        ),
        width=900,
        height=900,
        title="Detector pointing per slice",
    )

    fig.write_html(os.path.join(out_dir, "boresight_3Dvis.html"))
    return fig

def boresight_pointing_residualplot(jitter_boreradec, init_boreradec, n_slices=30, step_size=10, out_dir=""):
    boresight_coords = np.zeros((n_slices, 3))
    init_boresight_coords = np.zeros((n_slices, 3))
    
    fig, axs = plt.subplots(1, 1, figsize=(8,6))
    for i in range(n_slices):
        slc = slice(step_size * i, step_size * (i+1), step_size)

        bquat = np.array(jitter_boreradec[slc, :])
        qbang = np.zeros((3, bquat.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(bquat)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi

        boresight_coords[i] = np.array([qbang[0][0], qbang[1][0], step_size * i], dtype=float)

        bquat = np.array(init_boreradec[slc, :])
        qbang = np.zeros((3, bquat.shape[0]), dtype=np.float64)
        qbang[0], qbang[1], qbang[2] = qa.to_lonlat_angles(bquat)
        qbang[0] *= 180.0 / np.pi
        qbang[1] *= 180.0 / np.pi

        init_boresight_coords[i] = np.array([qbang[0][0], qbang[1][0], step_size * i], dtype=float)

    bore_x, bore_y, bore_z = boresight_coords[:, 0], boresight_coords[:, 1], boresight_coords[:, 2]
    init_bore_x, init_bore_y, init_bore_z = init_boresight_coords[:, 0], init_boresight_coords[:, 1], init_boresight_coords[:, 2]

    # Plot the residuals of the boresight pointing in latitude and longitude
    axs.set_title("Boresight jitter residuals")
    axs.plot(bore_z, init_bore_x - bore_x, label="Latitude")
    axs.set_ylabel(r"Residual ($^\circ$)")
    axs.plot(bore_z, init_bore_y - bore_y, label="Longitude")
    axs.set_xlabel("Time Index")
    # Add a second y axis on the right with the radians converted to arcseconds
    ax2 = axs.twinx()
    ax2.set_ylabel(r"Residual (arcsec)")
    ax2.set_ylim(axs.get_ylim()[0] * 3600, axs.get_ylim()[1] * 3600)
    ax2.set_yticks(np.linspace(axs.get_ylim()[0] * 3600, axs.get_ylim()[1] * 3600, 5))
    ax2.set_yticklabels([f"{int(tick)}" for tick in np.linspace(axs.get_ylim()[0] * 3600, axs.get_ylim()[1] * 3600, 5)])
    ax2.grid(False)

    axs.legend(loc="upper right")
    plt.savefig(os.path.join(out_dir, "boresight_residuals.png"), dpi=300, bbox_inches="tight")

    return fig