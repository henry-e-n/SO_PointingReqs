import numpy as np
import toast
from toast.tests import helpers

import argparse
import os


# MPI communicator
world, procs, rank = toast.mpi.get_world()
comm = helpers.create_comm(world, single_group=True)

def main(args):
    path_to_file = os.path.join(args.directory, "mlmapmaker_sky_map.fits")

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"File {path_to_file} does not exist.")

    toast.vis.plot_wcs_maps(
        hitfile=os.path.join(args.directory, "mlmapmaker_sky_hits.fits"),
        mapfile=os.path.join(args.directory, "mlmapmaker_sky_map.fits"),
        format="png",
        cmap=args.cmap,
        truth=args.truth,
        graticule=args.graticule
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a WCS map from a FITS file.")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing the mlmapmaker_sky_map.fits file.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap to use for the plot (default: viridis).",
    )
    parser.add_argument(
        "--truth",
        type=str,
        default=None,
        help="Optional path to a truth map for comparison.",
    )
    parser.add_argument(
        "--graticule",
        action="store_true",
        default=False,
        help="Whether to overlay a graticule on the map.",
    )
    args = parser.parse_args()

    main(args)