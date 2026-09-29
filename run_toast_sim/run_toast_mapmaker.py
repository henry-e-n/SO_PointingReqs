
from toast_mapmaker import toast_mapmaker
import argparse

def main(args):
    toast_mapmaker(data_volume=args.data_vol, out_dir=args.out_dir)
    print("Mapmaking complete.")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the TOAST mapmaker on a data volume."
    )
    parser.add_argument(
        "--data_vol",
        type=str,
        required=True,
        help="Path to the data volume to process.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where the output maps will be saved.",
    )

    args = parser.parse_args()
    main(args)