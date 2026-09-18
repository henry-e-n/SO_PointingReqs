# A new TOAST operator for adding a pointing offset to the telescope pointing.
# This is modeled after the HWP Wobble operator.

import numpy as np

import toast
from toast.timing import function_timer
from toast.traits import (
    trait_docs,
    Int,
    Unicode,
    List,
    Float
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults
import toast.qarray as qa


@trait_docs
class PointingJitter(Operator):
    """
    Add a pointing offset to the telescope pointing.
    This offset can be either static (constant) or time-dependent.
    """

    API = Int(0, help="Internal interface version for this operator")

    max_daz = List(
        0.0,
        help="The maximum deflection in Azimuth coordinates (in radians) to be applied to the telescope boresight pointing",
    )

    max_del = Float(
        0.0,
        help="The maximum deflection in Elevation coordinates (in radians) to be applied to the telescope boresight pointing",
    )

    corotator_max = Float(
        0.0,
        help="The maximum deflection in the corotator angle (in radians) to be applied to the telescope boresight pointing",
    )

    sin_amp = Float(
        0.0,
        help="The amplitude of the sinusoidal jitter (in radians) to be applied to the telescope boresight pointing",
    )

    sin_freq = Float(
        help="The frequency of the sinusoidal jitter (in 1/samples) to be applied to the telescope boresight pointing",
    )

    boresight_azel = Unicode(
        defaults.boresight_azel,
        allow_none=True,
        help="Observation shared key for boresight Az/El",
    )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        allow_none=True,
        help="Observation shared key for boresight RA/Dec",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        print(f"\nExecuting PointingJitter operator with max_daz={self.max_daz}, max_del={self.max_del}, and corotator_max={self.corotator_max}")
        log = Logger.get()

        for ob in data.obs:

            # Define the deflection in Xi Eta coordinates
            # Convert the lists to arrays
            num_samples = ob.shared['boresight_azel'].shape[0]
            # Make an array of random values between 0 and max
            az_rand_jitter = np.random.randn(num_samples)*self.max_daz - self.max_daz/2
            el_rand_jitter = np.random.randn(num_samples)*self.max_del - self.max_del/2
            corotator_jitter = np.random.randn(num_samples)*self.corotator_max - self.corotator_max/2
            
            az_sin = self.sin_amp*np.sin(self.sin_freq*np.arange(num_samples))
            el_sin = self.sin_amp*np.sin(self.sin_freq*np.arange(num_samples))

            lon_noise = -az_rand_jitter - az_sin
            lat_noise = el_rand_jitter + el_sin
            psi_noise = el_rand_jitter + el_sin

            if ob.comm_col_rank == 0:
                lon_arr, lat_arr, psi_arr = qa.to_lonlat_angles(ob.shared[self.boresight_azel].data)
                # add the noise term
                lon_arr += lon_noise
                lat_arr += lat_noise
                psi_arr += psi_noise

                # Recompose into quaternions
                bore = qa.from_lonlat_angles(lon_arr, lat_arr, psi_arr)
            else:
                bore = None
            ob.shared[self.boresight_azel].set(bore)

            if ob.comm_col_rank == 0:
                lon_arr, lat_arr, psi_arr = qa.to_lonlat_angles(ob.shared[self.boresight_radec].data)
                # add the noise term
                lon_arr += lon_noise
                lat_arr += lat_noise
                psi_arr += psi_noise

                # Recompose into quaternions
                bore = qa.from_lonlat_angles(lon_arr, lat_arr, psi_arr)
            else:
                bore = None
            ob.shared[self.boresight_radec].set(bore)

            del bore

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": [self.wobble_meta],
            "shared": [self.hwp_angle, self.boresight_radec, self.boresight_azel],
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        return dict()
