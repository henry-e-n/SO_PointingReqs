# A new TOAST operator for adding a pointing offset to the telescope pointing.
# This is modeled after the HWP Wobble operator.

import numpy as np

import toast
from toast.timing import function_timer
from toast.traits import (
    trait_docs,
    Int,
    Float,
    Unicode,
    List,
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults
import toast.qarray as qa
import astropy.units as u

@trait_docs
class PointingOffset(Operator):
    """
    Add a pointing offset to the telescope pointing.
    This offset can be either static (constant) or time-dependent.
    """

    API = Int(0, help="Internal interface version for this operator")

    dxi = Float(
        0.001,
        help="The deflection in Xi coordinates (in arcsec) to be applied to the telescope boresight pointing",
    )

    deta = Float(
        0.001,
        help="The deflection in Eta coordinates (in arcsec) to be applied to the telescope boresight pointing",
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
        log = Logger.get()
        log.info(f"HENRY CHECKPOINT : Executing Pointing Offset with dxi = {self.dxi}arcsec, deta = {self.deta}arcsec")
        self.dxi = (self.dxi*u.arcsec).to(u.rad).value
        self.deta = (self.deta*u.arcsec).to(u.rad).value
        for ob in data.obs:
            num_samples = ob.n_local_samples
            # Define the deflection in Xi Eta coordinates
            # Convert the lists to arrays
            # dxi = np.array(self.dxi)
            # deta = np.array(self.deta)
            # Convert the Xi Eta deflection to quaternions
            q_off = toast.instrument_coords.xieta_to_quat(self.dxi, self.deta, 0.0)
            # Apply deflection to the boresight pointing (in both azel and radec coordinates)

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
            "meta": [self.wobble_meta],
            "shared": [self.hwp_angle, self.boresight_radec, self.boresight_azel],
            "detdata": list(),
            "intervals": list(),
        }
        return req

    def _provides(self):
        return dict()



""" 
# Old Code
if ob.comm_col_rank == 0:
    bore = qa.mult(ob.shared[self.boresight_azel].data, qa.inv(defl_quat))
else:
    bore = None
ob.shared[self.boresight_azel].set(bore)

if ob.comm_col_rank == 0:
    bore = qa.mult(ob.shared[self.boresight_radec].data, qa.inv(defl_quat))
else:
    bore = None
ob.shared[self.boresight_radec].set(bore)

del bore
"""