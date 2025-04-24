from calmly import CavityAlignment, CavityScanPreProcess
import numpy as np

def cav_sim_factory():
    maxtem = 3
    mis_angle_min=-2e-4
    mis_angle_max=2e-4

    cav_sim = CavityAlignment.two_mirror_cavity(
        Lcav = 1.0,          # Cavity length IC<-->OC
        Rc = (np.inf, 3.0),  # RoCs of IC and OC
        Refl = (0.95, 0.95), # Reflection off IC and OC
        Loss = (0.0, 0.0),   # Internal loss in IC and OC
        StSp = (2.0, 1.0)    # Distances SM1<-->SM2 and SM2<-->IC
    )
    cav_sim.set_maxtem(maxtem)
    
    cav_sim.add_misalignment_axes([
        cav_sim.m.IC.xbeta,
        cav_sim.m.IC.ybeta,
        cav_sim.m.OC.xbeta,
        cav_sim.m.OC.ybeta
    ], [mis_angle_min]*4, [mis_angle_max]*4)
    
    cav_sim.init_all_steering_motors(pos_coef=1.2e-6, neg_coef=1.0e-6, uncertainty=0, low_limit=-1000, high_limit=1000)
    return cav_sim

def scan_processor_factory():
    signal_ref = None

    if signal_ref is None:
        signal_ref = cav_sim_factory().signal_ref
    return CavityScanPreProcess(signal_reference_level = signal_ref)