import numpy as np
import gymnasium as gym
import stable_baselines3
from gymnasium import spaces
from enum import Enum

from .simulator import CavityAlignment
from .preprocessing import CavityScanPreProcess

class Actions_1(Enum):
    DECREASE_SM1_YAW = 0
    INCREASE_SM1_YAW = 1
    DECREASE_SM1_PITCH = 2
    INCREASE_SM1_PITCH = 3
    DECREASE_SM2_YAW = 4
    INCREASE_SM2_YAW = 5
    DECREASE_SM2_PITCH = 6
    INCREASE_SM2_PITCH = 7

class CavityAlignmentEnv(gym.Env):
    def __init__(self,
                 cav_sim : CavityAlignment,
                 scan_preprocess : CavityScanPreProcess,
                 Npeaks = 10,
                 algo : str = "PPO",
                 training : bool = False,
                 Nmisalign : int = 1,
                 N_averages : int = 3
                ):
        super(CavityAlignmentEnv, self).__init__() # Probably not needed?

        self._training = training # Controls the actions only applied when training, e.g. action noise
        self._algorithm = algo # Action noise is to be included only in PPO training (not needed in DQN)
        self._N_averages = N_averages

        self.Npeaks = Npeaks

        # Discrete action space: fixed forward or backward step for each motor (8 in total)
        self.action_space = spaces.Discrete(8)

        # Observation space
        self.observation_space = spaces.Dict({
            "peak_dominances": spaces.Box(low=0.0, high=1.0, shape=(Npeaks,), dtype=np.float32),
            "peak_positions" : spaces.Box(low=0.0, high=1.0, shape=(Npeaks,), dtype=np.float32),
            # "peak_FWHMs" : spaces.Box(low=0.0, high=1.0, shape=(Npeaks,), dtype=np.float32),
            "cur_delta_dominance" : spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "prev_delta_dominance" : spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "last_motor_vector" : spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),  
            "last_motor_direction" : spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "prev_motor_vector" : spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            "prev_motor_direction" : spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "already_changed_direction" : spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # "limits_reached_vector" : spaces.MultiBinary(8), # If adding back, replace with Box
            "motor_counters" : spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
            "motor_usage_distribution" : spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        })

        # Simulation model of the cavity
        self.cav_sim = cav_sim

        # An object used for pre-processing cavity scans
        self.scan_preprocess = scan_preprocess

        # Number of steps without improvement before cutoff
        self._MAX_NO_IMPROVEMENT = 20
        # After the "no improvement" range, the agent will need to climb back to the best result
        # The following constant sets how close it must be to the previously obtained best result for termination
        self._BEST_RESULT_ANNEALING = 0.95
        # Max number of steps to reach the target
        self._MAX_FINAL_COUNTDOWN_STEPS = 10
        # Number of steps without seeing any peaks before giving up
        self._MAX_NO_PEAKS = 20
        # Maximum allowed step count for one motor operation (assuming the same one on each motor)
        # when using SAP
        self._SAP_MAX_COUNTS_PER_STEP=50
        # Number of counts in each motor action in discrete space (to use with PPO or DQN)
        self._DISCRETE_MOTOR_STEP_1 = 3
        # TODO: try using two step sizes (for coarse and fine alignment)

        # The following settings control how the action noise is applied (only during training)
        self._EXPLORATION_START       = 0.25     # Initial chance to use a random action instead of the predicted one
        self._EXPLORATION_END         = 0.01     # Final chance of random action at the end of the exploration period
        self._EXPLORATION_DECAY_STEPS = 200_000  # Number of steps in the exploration period

        # Number of episodes between consecutive misalignments
        self.Nmisalign = Nmisalign
        self._episode_counter = 0
        
        # # Maximum allowed number of steps
        self._MAX_STEPS=1000
        
        self._action_motor = {
            Actions_1.INCREASE_SM1_PITCH.value : 'SM1.ybeta',
            Actions_1.DECREASE_SM1_PITCH.value : 'SM1.ybeta',
            Actions_1.INCREASE_SM1_YAW.value : 'SM1.xbeta',
            Actions_1.DECREASE_SM1_YAW.value : 'SM1.xbeta',
            Actions_1.INCREASE_SM2_PITCH.value : 'SM2.ybeta',
            Actions_1.DECREASE_SM2_PITCH.value : 'SM2.ybeta',
            Actions_1.INCREASE_SM2_YAW.value : 'SM2.xbeta',
            Actions_1.DECREASE_SM2_YAW.value : 'SM2.xbeta'
        }
        self._action_counts = {
            Actions_1.DECREASE_SM1_PITCH.value : -self._DISCRETE_MOTOR_STEP_1,
            Actions_1.INCREASE_SM1_PITCH.value : self._DISCRETE_MOTOR_STEP_1,
            Actions_1.DECREASE_SM1_YAW.value : -self._DISCRETE_MOTOR_STEP_1,
            Actions_1.INCREASE_SM1_YAW.value : self._DISCRETE_MOTOR_STEP_1,
            Actions_1.DECREASE_SM2_PITCH.value : -self._DISCRETE_MOTOR_STEP_1,
            Actions_1.INCREASE_SM2_PITCH.value : self._DISCRETE_MOTOR_STEP_1,
            Actions_1.DECREASE_SM2_YAW.value : -self._DISCRETE_MOTOR_STEP_1,
            Actions_1.INCREASE_SM2_YAW.value : self._DISCRETE_MOTOR_STEP_1
        }

        self.reward_component_keys = [
            'total',
            'single_peak',
            'no_peaks',
            'main_peak_improvement',
            'main_peak_dominance',
            'other_peaks_dominance',
            'time',
            'motor_usage_dist',
            'motor_switching',
            'final_result'
        ]
        
    def _take_cavity_scan(self, write_values_to_model=True):
        """
        This method takes a cavity scan, preprocesses it 
        and stores the results in the model state if `write_values_to_model` is True.
        To be used in reset() and step() methods, but may also be called by the user
        through the wrapper `get_cavity_scan()` which sets write_values=False.
        """
        # Taking a cavity scan
        x, y = self.cav_sim.cavity_scan(ppbw_resolution=10, scan_range_in_fsrs=1.0, noisy=True)
        # Preprocessing the scan
        peak_positions, peak_dominances, peak_heights, peak_FWHMs, x1, y1, FSR = \
        self.scan_preprocess.process_cavity_scan(x, y, bypass_FSR_preprocessing=True, dominance_type='peak_height')
        Npeaks_found = len(peak_positions)

        if Npeaks_found == 0:
            cur_dominance = 0
            cur_r_dominance = 0
        else:
            # Dominance of the highest peak
            cur_dominance = peak_dominances[0]
            if Npeaks_found > 1:
                cur_r_dominance = sum(peak_dominances[1:])
            else:
                cur_r_dominance = 0

        if self._last_action is None:
            # We are calling from reset()
            cur_delta_dominance = None
            cur_delta_r_dominance = None
        else:
            cur_delta_dominance = cur_dominance - self._prev_dominance
            cur_delta_r_dominance = cur_r_dominance - self._prev_r_dominance

        if write_values_to_model:
            # Storing the values in the model
            self._x, self._y = (x, y)
            self._peak_positions, self._peak_dominances, self._peak_heights, self._peak_FWHMs, self._x1, self._y1, self._FSR = \
            (peak_positions, peak_dominances, peak_heights, peak_FWHMs, x1, y1, FSR)
            self._Npeaks_found = Npeaks_found
            
            self._cur_dominance = cur_dominance
            self._cur_r_dominance = cur_r_dominance
            self._cur_delta_dominance = cur_delta_dominance
            self._cur_delta_r_dominance = cur_delta_r_dominance
            
        return x1, y1
        
    def get_cavity_scan(self):
        return self._take_cavity_scan(write_values_to_model=False)

    def action_to_motor_index_and_dir(self, action):
        """Converts action number into a motor index and the direction indicator"""
        
        motor_index = int(action // 2)
        motor_direction  = -1 if action % 2 == 0 else 1

        return motor_index, motor_direction
    
    def _get_obs_and_info(self):
        # If Npeaks_found is less than Npeaks (in observation space), fill the missing parameters with zeros
        # If Npeaks_found is greater than Npeaks, ignore the smallest peaks
        # Returned peaks are sorted by peak height

        # Filling in the observation space with zeros
        obs = {
            "peak_dominances" : np.zeros(self.Npeaks, dtype=np.float32),
            "peak_positions"  : np.zeros(self.Npeaks, dtype=np.float32),
            # "peak_FWHMs"      : np.zeros(self.Npeaks, dtype=np.float32),
            "cur_delta_dominance" : np.zeros(1, dtype=np.float32),
            "prev_delta_dominance" : np.zeros(1, dtype=np.float32),
            "last_motor_vector" : np.zeros(4, dtype = np.float32),
            "last_motor_direction" : np.zeros(1, dtype = np.float32),
            "prev_motor_vector" : np.zeros(4, dtype = np.float32),
            "prev_motor_direction" : np.zeros(1, dtype = np.float32),
            "already_changed_direction" : np.zeros(1, dtype = np.float32),
            # "limits_reached_vector" : np.zeros(8, dtype = np.float32),
            "motor_counters"  : np.zeros(4, dtype=np.float32),
            "motor_usage_distribution"  : np.zeros(4, dtype=np.float32)
        }
        if self._last_action is not None:
            last_motor_index, last_motor_direction = self.action_to_motor_index_and_dir(self._last_action)
            obs["last_motor_vector"][last_motor_index] = 1
            obs["last_motor_direction"][0] = last_motor_direction
            # obs["limits_reached_vector"][self._last_action] = self._last_action_reached_lim
            if self._prev_action is not None:
                prev_motor_index, prev_motor_direction = self.action_to_motor_index_and_dir(self._prev_action)
                obs["prev_motor_vector"][prev_motor_index] = 1
                obs["prev_motor_direction"][0] = prev_motor_direction

        obs["already_changed_direction"][0] = self._already_changed_direction
            
        # Number of peaks to copy
        Ncopy = self._Npeaks_found if  self._Npeaks_found < self.Npeaks else self.Npeaks
        # Copying the peaks into the observation space
        obs["peak_dominances"][:Ncopy] = self._peak_dominances[:Ncopy]
        obs["peak_positions"][:Ncopy]  = self._peak_positions[:Ncopy]
        obs["motor_counters"] = np.array(self.cav_sim.get_all_steering_motor_counters(normalised=True),dtype = np.float32)
        # obs["peak_FWHMs"][:Ncopy]  = self._peak_FWHMs[:Ncopy]

        usage_total = self._motor_usage_counts.sum()
        usage_dist = self._motor_usage_counts/usage_total \
        if usage_total >= 4 \
        else np.ones(4)/4
        obs["motor_usage_distribution"] = usage_dist.astype(np.float32)

        cur_delta_dominance = self._cur_delta_dominance if self._cur_delta_dominance is not None else 0
        obs["cur_delta_dominance"] = np.array([cur_delta_dominance], dtype=np.float32)
        prev_delta_dominance = self._prev_delta_dominance if self._prev_delta_dominance is not None else 0
        obs["prev_delta_dominance"] = np.array([prev_delta_dominance], dtype=np.float32)
        
        # info dict
        info = {
            "x1" : self._x1,
            "y1" : self._y1,
            "FSR" : self._FSR,
            "Npeaks_found" : self._Npeaks_found
        }
        return obs, info
        

    def reset(self, seed=None):
        # Seeding the RNG
        super().reset(seed=seed)

        # Providing the RNG to the simulation
        self.cav_sim.connect_numpy_rng(self.np_random)
        
        # Resetting the counters
        self._best_dominance = 0
        self._no_improvement_steps = 0
        self._no_peak_steps = 0
        self._steps_taken = 0

        if self._episode_counter % self.Nmisalign == 0:
            # Misaligning some of the cavity mirrors
            self.cav_sim.misalign_axes()

        self._last_action = None
        self._prev_action = None
        self._prev_dominance = None
        self._prev_delta_dominance = None
        self._final_dominance_target = None
        self._last_action_reached_lim = False
        self._already_changed_direction = False
        self._final_countdown = None
        self._motor_usage_counts = np.zeros(4, dtype=np.int32)

        # Resetting the steering mirrors and their motors to all zeroes
        self.cav_sim.set_steering_angles()
        self.cav_sim.reset_steering_motors()

        # Taking a cavity scan
        self._take_cavity_scan(write_values_to_model=True)
                
        self._best_dominance = self._cur_dominance

        # Getting an observation
        observation, info = self._get_obs_and_info()
        
        self._episode_counter += 1

        return observation, info

    def step(self, action):

        # Moving the appropriate variables
        self._prev_action = self._last_action
        self._prev_delta_dominance = self._cur_delta_dominance
        self._prev_dominance = self._cur_dominance
        self._prev_r_dominance = self._cur_r_dominance
        
        # Include action noise in the PPO training
        if self._training and (self._algorithm=="PPO") and (self._steps_taken < self._EXPLORATION_DECAY_STEPS):
            current_exploration_chance = self._EXPLORATION_START + \
            (self._EXPLORATION_END - self._EXPLORATION_START)/self._EXPLORATION_DECAY_STEPS * self._steps_taken
            if self.np_random.random() < current_exploration_chance:
                action = self.np_random.integers(self.action_space.n)
        # Moving one motor by a fixed number of counts
        self._last_action_reached_lim = any(
            self.cav_sim.move_steering_motor(self._action_motor[action], self._action_counts[action])
        )

        self._last_action = action

        # Take a cavity scan
        self._take_cavity_scan(write_values_to_model=True)

        ######## TEMPORARY SECTION ########
        ######## REWARD SCALING ###########
        _r0 = 1 # average reward per step (should not matter)
        _ddom_to_r = 846 # 846 # proportionality coefficient r = ddom_to_r* ddom
        _drdom_to_r = _ddom_to_r / 2 # for the rest of the peaks
        _dom_to_r = 5# proportionality coefficient r = dom_to_r* dom
        _rdom_to_r = _dom_to_r / 2 # for the rest of the peaks
        _Nmax_steps = 93*3
        _r0_to_rb = 308
        _no_peaks_penalty = 1 # Additional penalty for zero peaks (will be scaled by _r0)
        _single_peak_reward = 1 # Additional reward for a single peak (will be scaled by _r0)
        ###### END OF TEMP SECTION ########

        # reward = 0
        reward_components = {}
        
        if self._Npeaks_found == 0:
            # counting consecutive steps with no peaks
            self._no_peak_steps += 1
            self._no_improvement_steps = 0

            reward_components['no_peaks'] = - _no_peaks_penalty * _r0 # penalty for loosing all the peaks
        else:
            # resetting the counter for consecutive steps with no peaks
            self._no_peak_steps = 0
            
            if self._Npeaks_found == 1:
                # We have a single peak
                reward_components['single_peak'] = _single_peak_reward * _r0

        # Reward for improving the main peak
        reward_components['main_peak_improvement'] = max(_ddom_to_r * self._cur_delta_dominance * _r0, 0)
        # Reward for keeping the main peak good
        reward_components['main_peak_dominance'] = _dom_to_r * self._cur_dominance * _r0 
        # reward -= _drdom_to_r * self._cur_delta_r_dominance * _r0 # Penalty for improving other peaks
        # Penalty for keeping the other peaks good
        reward_components['other_peaks_dominance'] = - _rdom_to_r * self._cur_r_dominance * _r0 

        # small penalty growing with the number of steps
        reward_components['time'] = - self._steps_taken * _r0 / _Nmax_steps
        
        cur_motor_index, cur_motor_direction = self.action_to_motor_index_and_dir(self._last_action) \
        if self._last_action is not None \
        else (-1, 0)
        prev_motor_index, prev_motor_direction = self.action_to_motor_index_and_dir(self._prev_action) \
        if self._prev_action is not None \
        else (-1, 0)
        # Is it the same motor as the one at the previous step?
        # same_motor = (observation["prev_motor_vector"] == observation["last_motor_vector"]).all()
        same_motor = (cur_motor_index == prev_motor_index)
        # Is it the same motor direction as the one at the previous step?
        # same_direction = (observation["prev_motor_direction"] == observation["last_motor_direction"])
        same_direction = (cur_motor_direction == prev_motor_direction)

        if not same_motor:
            # update the motor usage counter
            motor_index, _ = self.action_to_motor_index_and_dir(self._last_action) \
            if self._last_action is not None \
            else (None,None)
            if motor_index is not None:
                self._motor_usage_counts[motor_index] += 1

        usage_total = self._motor_usage_counts.sum()
        usage_dist = self._motor_usage_counts/usage_total \
        if usage_total >= 4 \
        else np.ones(4)/4
        
        uniform_dist = np.ones_like(usage_dist) / len(usage_dist)
        usage_dist_penalty_unscaled = np.linalg.norm(usage_dist - uniform_dist)
        # Penalty for the deviation of the motor usage distribution from the uniform distribution
        reward_components['motor_usage_dist'] = - usage_dist_penalty_unscaled * 10 * _r0
        
        if self._prev_delta_dominance is None:
            # This means we are at step 1, so we are bypassing this logical block entirely
            pass
        elif self._prev_delta_dominance == 0:
            # This means that the peaks were lost
            # TODO: maybe add a reward if self._cur_delta_dominance > 0 to encourage the restoration
            # although it will already be reflected in the reward for cur_dominance and cur_delta_dominance
            # TODO: check if the history of actions since the last best result is available
            # and apply it in reverse for a large reward if so (should be a separate action)
            pass
        elif self._prev_delta_dominance > 0:
            # Alignment was improved during the previous step
            if same_motor and same_direction:
                # Repeating is the right action -- adding a positive reward
                reward_components['motor_switching'] = 1 * _r0
            else:
                # All other actions are equally bad -- adding the same penalty
                reward_components['motor_switching'] = - 1 * _r0
        elif (self._prev_delta_dominance < 0) and (self._already_changed_direction):
            # Alignment was impaired during the previous step and we had already tried changing the motor direction
            if not same_motor:
                # Changing the motor is the right action -- adding a positive reward
                reward_components['motor_switching'] = 1 * _r0
            else:
                # Staying on the same motor is wrong, regardless of the direction
                # Adding a penalty
                reward_components['motor_switching'] = - 1 * _r0
        elif (self._prev_delta_dominance < 0) and (not self._already_changed_direction):
            # Alignment was impaired during the previous step and we had not yet tried changing the motor direction
            if same_motor and (not same_direction):
                # Changing the direction was the right thing to do, adding a reward
                reward_components['motor_switching'] = 1 * _r0
            elif not same_motor:
                # Changing the motor wasn't the optimal action but not too bad
                # No reward or small penalty
                reward_components['motor_switching'] = - 0.2 * _r0
            else:
                # Repeating the action that impaired the alignment was a bad idea. Adding a penalty
                reward_components['motor_switching'] = - 1 * _r0
                
        # counting consecutive steps with no improvement
        # TODO: possibly track the moving average instead
        # or forget the best peak ratio with time to allow the exporation of the parameter space
        if self._cur_dominance > self._best_dominance:
            self._best_dominance = self._cur_dominance
            self._no_improvement_steps = 0
        # elif (self._Npeaks_found == 1) and (self._cur_dominance > self._best_dominance*self._BEST_RESULT_ANNEALING):
        #     self._no_improvement_steps += 1
        # else:
        #     self._no_improvement_steps = 0 # these steps are only increasing if the alignment stays good
        else:
            self._no_improvement_steps += 1

        terminated = False
        truncated = False
        
        if (self._final_countdown is None) and (self._no_improvement_steps >= self._MAX_NO_IMPROVEMENT):
            if self._Npeaks_found == 1:
                # Assuming that we have found the best dominance. Setting the final target dominance:
                self._final_dominance_target = self._best_dominance * self._BEST_RESULT_ANNEALING
                self._final_countdown = self._MAX_FINAL_COUNTDOWN_STEPS
            else:
                # Assuming that we messed the alignment. Terminating immediately
                terminated = True
                # Penalty for the failed alignment
                reward_components['final_result'] = - _r0_to_rb * _r0
        elif self._final_countdown is not None:
            if self._cur_dominance >= self._final_dominance_target:
                terminated = True
                # final positive bonus
                reward_components['final_result'] = _r0_to_rb * _r0
            elif self._final_countdown == 0:
                # Did not reach the target dominance
                terminated = True
                reward_components['final_result'] = - _r0_to_rb * _r0
            else:
                self._final_countdown -= 1
        
        # checking if the alignment is hopelessly lost
        elif self._no_peak_steps >= self._MAX_NO_PEAKS:
            terminated = True
            # this hsould be strongly discouraged
            reward_components['final_result'] = - _r0_to_rb * _r0

        # truncating if the number of steps exceeds the maximum
        if self._steps_taken>=self._MAX_STEPS:
            truncated = True
            # terminated = True
            # Not necessarily have to add a penaly here if the gradual penalty growing with the step number is introduced
            # reward -= 5*_r0
            reward_components['final_result'] = - _r0_to_rb/5 * _r0

        reward = sum(reward_components.values())
        reward_components['total'] = reward
        
        # Obtaining an observation
        observation, info = self._get_obs_and_info()

        info['terminated'] = terminated
        info['truncated'] = truncated
        for k, v in reward_components.items():
            if k not in self.reward_component_keys:
                raise ValueError(f"""Reward component {k} is not in self.reward_component_keys.
                Modify reward_component_keys in CavityAlignmentEnv.__init__()""")
            info['reward/'+k] = v

        if not same_motor:
            self._already_changed_direction = False
        elif not same_direction:
            self._already_changed_direction = True
        elif self._cur_delta_dominance > 0:
            # If the current direction gives improvement, there is no need to change it in the end
            # TODO: rename self._already_changed_direction to reflect this option better.
            self._already_changed_direction = True
        self._steps_taken += 1

        # reward = np.sign(reward)*(1+np.log(np.abs(reward))) # Log-squeeze the reward

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

class AvgRewardPerStepWrapper(gym.Wrapper):
    """
    A wrapper for CavityAlignmentEnv 
    to calculate the average reward per step metric.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_sum = 0.0
        self.step_count = 0

    def reset(self, **kwargs):
        self.reward_sum = 0.0
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward_sum += reward
        self.step_count += 1

        if terminated or truncated:
            avg_reward_per_step = self.reward_sum / max(self.step_count, 1)
            info["avg_reward_per_step"] = avg_reward_per_step

        return obs, reward, terminated, truncated, info


def get_env_factory(
    cav_sim_factory,
    scan_processor_factory,
    *,
    training=True,
    monitor=False,
    algo="PPO",
    Nmisalign = 1,
    Npeaks = 3,
    maxtem = 3,
    mis_angle_min=-2e-4,
    mis_angle_max=2e-4,
    seed=None,
    use_avg_reward_wrapper: bool = False
):
    """
    Returns a function that builds a new environment by calling appropriate constructors.

    Parameters
    ----------
    cav_alignment_fn : Callable[[], CavityAlignment]
        Function that returns a new CavityAlignment instance.
    scan_processor_fn : Callable[[], CavityScanPreProcess]
        Function that returns a new CavityScanPreProcess instance.
    ...
    """
    def _init():
        cav_sim = cav_sim_factory()
        scan_proc = scan_processor_factory()

        env = CavityAlignmentEnv(
            cav_sim=cav_sim,
            scan_preprocess=scan_proc,
            training=training,
            Npeaks=Npeaks,
            algo=algo,
            Nmisalign=Nmisalign
        )
        if seed is not None:
            env.reset(seed=seed)

        if use_avg_reward_wrapper:
            env = AvgRewardPerStepWrapper(env)
            
        if monitor:
            env = stable_baselines3.common.monitor.Monitor(env)
        
        return env

    return _init


