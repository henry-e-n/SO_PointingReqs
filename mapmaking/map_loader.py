import numpy as np
import toast
import argparse
import os

def main(args):
    path_to_file = os.path.join(args.directory, "mlmapmaker_sky_map.fits")

    if not os.path.exists(path_to_file):
        raise FileNotFoundError(f"File {path_to_file} does not exist.")

    toast.vis.plot_wcs_maps(
        mapfile=str(path_to_file),
        format="png",
        cmap="RdBu",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a WCS map from a FITS file.")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing the mlmapmaker_sky_map.fits file.",
    )
    args = parser.parse_args()

    main(args)