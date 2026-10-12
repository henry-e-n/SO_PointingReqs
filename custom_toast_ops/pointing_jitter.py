# A new TOAST operator for adding a pointing offset to the telescope pointing.
# This is modeled after the HWP Wobble operator.
# Author : Henry Nachman

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

    sigma_az = Float(
        0.0,
        help="Gaussian az offset rms [rad] to be applied to the telescope boresight pointing",
    )

    sigma_el = Float(
        0.0,
        help="Gaussian el offset rms [rad] to be applied to the telescope boresight pointing",
    )

    sigma_psi = Float(
            0.0,
            help="Gaussian roll offset (corotator) rms [rad] to be applied to the telescope boresight pointing",
        )

    corotator_max = Float(
        0.0,
        help="The maximum deflection in the corotator angle (in radians) to be applied to the telescope boresight pointing",
    )

    sin_amp = Float(
        1.0,
        help="The amplitude of the sinusoidal jitter (in radians) to be applied to the telescope boresight pointing",
    )

    sin_freq = Float(
        1.0,
        help="The frequency of the sinusoidal jitter (in 1/samples) to be applied to the telescope boresight pointing",
    )

    boresight_azel = Unicode(
        defaults.boresight_azel,
        allow_none=True,
        help="Observation shared key for boresight Az/El",
    )

    boresight_azel_out = Unicode(
            "boresight_azel_jitter",
            allow_none=True,
            help="Output key of modified boresight azel. If None, modify the input key in place",
        )

    boresight_radec = Unicode(
        defaults.boresight_radec,
        allow_none=True,
        help="Observation shared key for boresight RA/Dec",
    )

    boresight_radec_out = Unicode(
                "boresight_radec_jitter",
                allow_none=True,
                help="Output key of modified boresight radec. If None, modify the input key in place",
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        print(f"\nExecuting PointingJitter operator with max_daz={self.max_daz}, max_del={self.max_del}, and corotator_max={self.corotator_max}")
        log = Logger.get()
        x, y, z = np.eye(3) # Creates a diagonal vector for the rotation axes
        for ob in data.obs:

            # Define the deflection in Xi Eta coordinates
            # Convert the lists to arrays
            num_samples = ob.n_local_samples
            first = ob.local_index_offset
            # Draw from random Gaussian
            draws = [
                toast.rng.random(num_samples, key=(self.seed, ob.uid),
                                 counter=(1,first), sampler="gaussian")
                for i in range(3)
            ]
            t = np.arange(first, first + num_samples)

            # Make a sinusoidal term
            sine_term = self.sin_amp * np.sin(self.sin_freq * t)

            # Make the jitter terms
            lon_noise = -self.sigma_az * draws[0] - sine_term
            lat_noise = self.sigma_el * draws[1] + sine_term
            psi_noise = self.sigma_psi * draws[2] # No sine term for corotator

            # q_off = qa.mult(
            #     qa.mult(qa.rotation(y, daz_term), qa.rotation(x, -del_term)),
            #     qa.rotation(z, dpsi_term)
            # )

            lon_arr, lat_arr, psi_arr = qa.to_lonlat_angles(ob.shared[self.boresight_azel].data)
            # add the noise term
            lon_arr += lon_noise
            lat_arr += lat_noise
            psi_arr += psi_noise

            q_off = qa.from_lonlat_angles(lon_arr, lat_arr, psi_arr)

            for key_in, key_out in (
                (self.boresight_azel, self.boresight_azel_out),
                (self.boresight_radec, self.boresight_radec_out),
            ):
                key_out = key_out or key_in
                if key_out not in ob.shared:
                    # Assigns to the output key if it doesn't exist yet
                    ob.shared.create_column(key_out, shape=(num_samples, 4), dtype=np.float64())
                new = None
                if ob.comm_col_rank == 0:
                    new = qa.mult(ob.shared[key_in].data, q_off)
                ob.shared[key_out].set(new, fromrank=0)

            

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.boresight_radec, self.boresight_azel],
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        outs = [
            self.boresight_azel_out or self.boresight_azel,
            self.boresight_radec_out or self.boresight_radec,
        ]
        return_dict = {
            "shared": outs, 
            "detdata": list(),
            "intervals": list(),
            "meta": list()
        }
        return return_dict


"""
            # The old way of doing things
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
            """