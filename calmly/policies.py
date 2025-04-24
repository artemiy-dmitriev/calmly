import numpy as np

def iterative_policy(obs : dict) -> int:
    """
    This is a simple heuristic algorithm which mimics the standard manual iterative process for aligning an optical cavity.
    As a policy, it takes an observation space dictionary as the only argument and returns an integer representing the suggested action.
    
    The algorithm goes through all motors consequitively as follows:
    0. Select motor 0.
    1. Move the selected motor in some direction.
    2. If the alignment worsened, move the same motor in the opposite direction.
    3. Repeat the last action until the alignment stops improving.
    4. Switch to the next motor.
    """
    last_motor_index = np.where(obs["last_motor_vector"])[0][0] if sum(obs["last_motor_vector"])>0 else None
    last_motor_dir_mod = obs["last_motor_direction"][0]/2 + 0.5 # {-1,1}->{0,1}
    if last_motor_index is None:
        new_action = 0
    else:
        last_action = int(2*last_motor_index+last_motor_dir_mod)

        if obs["cur_delta_dominance"] < 0:
            new_action = last_action + 1 if last_action != 7 else 0
        else:
            new_action = last_action
    
    return new_action