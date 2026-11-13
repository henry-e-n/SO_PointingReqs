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
import astropy.units as u

@trait_docs
class PointingJitter(Operator):
    """
    Add a pointing jitter to the telescope pointing. This jitter is defined as a
    radial offset in az el coordinates.

    $\delta_r = \sigma_{az} \cdot \mathcal{Z}_{az}$
    
    $\theta = \mathcal{U}$

    Then the r, $\theta$ coordinates are decomposed into lon lat coordinates. 

    $\delta_{lon} = \delta_r \cos{\theta}$

    $\delta_{lat} = \delta_r \sin{\theta}$
    
    $\delta_{\psi} = \sigma_{\psi} \cdot \mathcal{Z}_{\psi}$

    """

    API = Int(0, help="Internal interface version for this operator")

    seed = Int(0, help="Random number generator seed for the pointing jitter")
    sigma_r = Float(
        0.0,
        help="Gaussian r offset rms [arcsec] to be applied to the telescope boresight pointing",
    )

    variance_r = Float(
        0.0,
        help="The variance in the jitter value as defined by $\sigma_r^2$. This overwrites the sigma_r parameter."
    )

    sigma_psi = Float(
            0.0,
            help="Gaussian roll offset (corotator) rms [arcsec] to be applied to the telescope boresight pointing",
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
        log = Logger.get()
        log.info(f"\nExecuting PointingJitter operator with seed={self.seed}, sigma_r={self.sigma_r}")
        if self.variance_r != 0.0:
            # Variance has been defined - set sigma r to sqrt
            self.sigma_r = np.sqrt((self.variance_r*u.arcsec*u.arcsec)).to(u.rad).value
        self.sigma_r = (self.sigma_r*u.arcsec).to(u.rad).value
        self.sigma_psi = (self.sigma_psi*u.arcsec).to(u.rad).value
        for ob in data.obs:

            # Define the deflection in Xi Eta coordinates
            # Convert the lists to arrays
            num_samples = ob.n_local_samples
            first = ob.local_index_offset
            # Draw from random Gaussian
            draws = [
                toast.rng.random(num_samples, key=(self.seed, ob.uid),
                                 counter=(i,first), sampler="gaussian")
                for i in range(2)
            ]

            theta_draw = [
                toast.rng.random(num_samples, key=(self.seed, ob.uid),
                                 counter=(i, first), sampler="uniform")
                for i in range(1)
            ]
            t = np.arange(first, first + num_samples)

            # Make the jitter terms
            r_sample = self.sigma_r * draws[0]
            theta_sample = theta_draw[0]

            lon_noise = r_sample*np.cos(theta_sample)
            lat_noise = r_sample*np.sin(theta_sample)
            psi_noise = self.sigma_psi * draws[1] # No sine term for corotator

            azel_nom = ob.shared[self.boresight_azel].data
            lon_arr, lat_arr, psi_arr = qa.to_lonlat_angles(azel_nom)
            # add the noise term
            lon_arr += lon_noise
            lat_arr += lat_noise
            psi_arr += psi_noise

            azel_jitter = qa.from_lonlat_angles(lon_arr, lat_arr, psi_arr)
            # azel jitter is the jittered quaternion
            # to get deflection quaternion, we need to multiply by the inverse of the nominal quaternion

            q_off = qa.mult(qa.inv(azel_nom), azel_jitter)

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
                    # Now we get the new jittered pointing (in the given coordinate system)
                    # By multiplying the input nominal pointing by the deflection quaternion
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