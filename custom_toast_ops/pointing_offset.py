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
)
from toast.ops.operator import Operator
from toast.utils import Logger
from toast.observation import default_values as defaults
import toast.qarray as qa


@trait_docs
class PointingOffset(Operator):
    """
    Add a pointing offset to the telescope pointing.
    This offset can be either static (constant) or time-dependent.
    """

    API = Int(0, help="Internal interface version for this operator")

    dxi = List(
        [0.0],
        help="The deflection in Xi coordinates (in radians) to be applied to the telescope boresight pointing",
    )

    deta = List(
        [0.0],
        help="The deflection in Eta coordinates (in radians) to be applied to the telescope boresight pointing",
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
        print(f"\nExecuting PointingOffset operator with dxi={self.dxi} and deta={self.deta}")
        log = Logger.get()

        for ob in data.obs:

            # Define the deflection in Xi Eta coordinates
            # Convert the lists to arrays
            dxi = np.array(self.dxi)
            deta = np.array(self.deta)
            # Convert the Xi Eta deflection to quaternions
            defl_quat = toast.instrument_coords.xieta_to_quat(dxi, deta, 0.0)
            print(defl_quat)
            # Apply deflection to the boresight pointing (in both azel and radec coordinates)
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
