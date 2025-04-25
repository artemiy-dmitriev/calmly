import importlib.util
import sys
import pickle
import cloudpickle
from pathlib import Path
from types import FunctionType
from typing import Callable, Tuple
import stable_baselines3

import matplotlib.pyplot as plt
import numpy as np

from .env import CavityAlignmentEnv

def load_factories(factory_path: str) -> Tuple[Callable, Callable]:
    """
    Load cav_sim_factory and scan_processor_factory from a Python script (preferred) or pickle.
    
    Example python script:
    ```python
    # my_factories.py
    
    from calmly import CavityAlignment, CavityScanPreProcess
    
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
    ```

    Alternatively, these functions can be pickled:
    ```python
    import cloudpickle

    with open("my_factories.pkl", "wb") as f:
        cloudpickle.dump((cav_sim_factory, scan_processor_factory), f)
    ```
    Use pickle only if you don’t need long-term or cross-platform portability!

    Parameters
    ----------
    factory_path : str
        Path to either:
        - A `.py` file defining `cav_sim_factory()` and `scan_processor_factory()` functions
        - A `.pkl` file storing a tuple of (cav_sim_factory, scan_processor_factory). 

    Returns
    -------
    Tuple[Callable, Callable]
        cav_sim_factory, scan_processor_factory
    """
    path = Path(factory_path)
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("user_factories", str(path))
        module = importlib.util.module_from_spec(spec)
        # sys.modules["user_factories"] = module
        spec.loader.exec_module(module)

        try:
            return module.cav_sim_factory, module.scan_processor_factory
        except AttributeError as e:
            raise RuntimeError(f"Missing factory functions in {factory_path}") from e

    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, tuple) and all(isinstance(x, FunctionType) for x in obj):
            return obj
        else:
            raise TypeError("Pickled file must contain a tuple of two functions.")

    else:
        raise ValueError("Unsupported file type. Use a .py script or .pkl file.")


class EpisodeResultsDict(dict):
    """
    A dictionary to store the results of a simulation of one alignment episode
    supplemented by custom methods e.g. for plotting the cavity scans (`plot_cavity_scans()`) or
    plotting the important metrics (`plot_metrics()`).
    """
    def plot_cavity_scans(
        self,
        scans : list[bool] = None,
        scan_labels : list[str] = None,
        offset : float = 0.1, 
        show : bool = True,
        figsize : tuple[float, float] = None,
        **kwargs
    ):
        """
        This method creates plots of cavity scans taken during the alignment process. If `scans` list is not specified,
        only the initial and the final scans will be shown; otherwise, scans with indices given in `scans` will be shown.
        
        Parameters
        ----------
        scans : list[bool],
            list of alignment steps for which the cavity scans are to be plotted.
            If `None` (default), only the initial and the final scans will be plotted (equivalent to `scans=[0,-1]`).
        scan_labels : list[str],
            An optional list of labels for the cavity scans
        offset : float,
            The preprocessing script always puts the top of the highest peak in the cavity scan in the zero position of the y array,
            which is not great for the visualisation because the main peak gets split in two.
            This variable provides an offset for the x axis in the units of the overall scan range to shift the main peak.
            Default is 0.1.
        show : bool,
            If `True`, the figure will be shown on screen. Default is `True`.
        figsize : tuple[float, float],
            Figure size tuple. See `matplotlib.pyplot.figure()` for details. Default is `None`.
        **kwargs
            Any additional arguments are passed to the matplotlib's `Axes.plot()` method.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        labelling=True
        
        if scans is None:
            scans = [0, -1]
            if scan_labels is None:
                scan_labels = ['Initial', ' Final']

        if scan_labels is None:
            scan_labels = [None]*len(scans)
        
        if scan_labels == [None]*len(scan_labels):
            labelling = False

        
                
        plt_interactive = plt.isinteractive()
        plt.ioff()
            
        fig = plt.figure() if figsize is None else plt.figure(figsize=figsize)
        ax = fig.gca()
        for scan, scan_label in zip(scans, scan_labels):
            offset_N = int(len(self['infos'][scan]['x1']) * offset)
            ax.plot(
                self['infos'][scan]['x1'],
                np.roll(self['infos'][scan]['y1'], offset_N),
                label=scan_label,
                **kwargs)
        if labelling:
            ax.legend()
        ax.set_title('Cavity scans')
        ax.set_xlabel('Scan range in FSRs')
        ax.set_ylabel('Detected signal')
        if show:
            plt.show()
        plt.close(fig)
        if plt_interactive:
            plt.ion()
        return fig, ax

    def plot_metrics(
        self,
        step_lowlim : int = None,
        step_highlim : int = None,
        show : bool = True,
        figsize : tuple[float, float] = (10,7)
    ):
        """
        Plots some important metrics for the simulated episode. Four plots are produced:
        1. Plots of all reward components at every step
        2. Peak dominance of the main peak and peak dominance of all other peaks at every step
        3. Improvement in the main peak dominance at each step
        4. Action taken at each step.
        
        Parameters
        ----------
        step_lowlim : int
            The first step index to be included in the plots. Default is `None` (start from the initial alignment)
        step_highlim : int
            Index of the last step to include in the plots. Default is `None` (do not exclude any steps at the end).
            A typical choice would be `step_highlim=-1` to exclude the last step, which will include a significant
            final bonus reward or penalty depending on the overall result.
        show : bool,
            If `True`, the figure will be shown on screen. Default is `True`.
        figsize : tuple[float, float],
            Figure size tuple. See `matplotlib.pyplot.figure()` for details. Default is `(10,7)`.
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        steps = np.arange(len(self['rewards']))
        
        ddom_list_nonone = [ddom for ddom in self['delta_dominance_list'][step_lowlim:step_highlim] if ddom is not None]
        if len(ddom_list_nonone) != 0:
            ddom_rms = (sum([ddom**2 for ddom in ddom_list_nonone])/len(ddom_list_nonone))**0.5
            ddom_max = max([np.abs(ddom) for ddom in ddom_list_nonone])
        else:
            ddom_rms=0
            ddom_max=0
        
        plt_interactive = plt.isinteractive()
        plt.ioff()
        
        fig = plt.figure(figsize=figsize)
        plt.subplot(221)
        plt.plot(steps[step_lowlim:step_highlim], self['rewards'][step_lowlim:step_highlim], lw=3, label='Total')
        for key in self['reward_components'].keys():
            if key=='total':
                continue
            plt.plot(steps[step_lowlim:step_highlim], self['reward_components'][key][step_lowlim:step_highlim], label=key)
        
        plt.legend(fontsize='xx-small')
        plt.title(f'Reward (tot={sum(self['rewards'][step_lowlim:step_highlim]):.1f}, ave={sum(self['rewards'][step_lowlim:step_highlim])/len(self['rewards'][step_lowlim:step_highlim]):.1f})')
        plt.ylabel('Reward per step')
        axc1 = plt.gca()
        plt.subplot(222)
        plt.plot(steps[step_lowlim:step_highlim], self['dominance_list'][step_lowlim:step_highlim], label='Main peak')
        plt.plot(steps[step_lowlim:step_highlim], self['r_dominance_list'][step_lowlim:step_highlim], label='Other peaks')
        plt.legend()
        plt.title(f"Peak dominance")
        plt.ylabel("Peak dominance")
        axc2 = plt.gca()
        plt.subplot(223, sharex = axc1)
        plt.plot(steps[step_lowlim:step_highlim], self['delta_dominance_list'][step_lowlim:step_highlim])
        plt.title(f'$\\delta$-dominance (rms={ddom_rms:.1e}, max={ddom_max:.1e})')
        plt.ylabel('Dominance improvement')
        plt.xlabel('Step')
        plt.subplot(224, sharex = axc2)
        plt.plot(steps[step_lowlim:step_highlim], self['action_list'][step_lowlim:step_highlim])
        plt.title('Actions taken')
        plt.ylabel('Action number')
        plt.xlabel('Step')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        plt.close(fig)
        if plt_interactive:
            plt.ion()
        
        return fig

def simulate_episode_with_trained_model(
    model : stable_baselines3.common.base_class.BaseAlgorithm | Callable[[dict],int],
    test_env : CavityAlignmentEnv,
    quiet : bool = False,
    Nsteps_to_print_after : int = 50
) -> EpisodeResultsDict:
    """
    This method simulates one alignment episode for the cavity sprecified in the given gymnasium environment `test_env`
    using the specified trained `stable_baselines3` model `model` as the agent to select the actions.

    Returns a dictionary with a summary of results that can be visualised using `plot_sim_episode_results()`.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm | Callable[[dict],int]
        Either an SB3 model used to select actions based on intermediate results or an explicit policy function, which should
        take the observation dict as the argument and return the suggested action to be taken.
    test_env : CavityAlignmentEnv
        `CavityAlignmentEnv` instance that has the gymnasium environment for the cavity model under test.
        Usually can be created using the factory method created by `calmly.get_env_factory()`
    quiet : bool
        if `True`, will print to stdout intermediate statistics after every few steps and the summary of the final result at the end.
        Default is `True`.
    Nsteps_to_print_after : int
        If `quiet` is set to `True`, the intermediate results will be prined to stdout every `Nsteps_to_print_after` steps.
        Default is 50.

    Returns
    -------
    out : EpisodeResultsDict
        Dictionary containing results and metrics, plus some visualisation methods.
    """
    obs, info = test_env.reset()
    terminated = False
    truncated=False
    episode_reward = 0
    step_count=0
    rewards =[]
    action_list = []
    dominance_list = []
    r_dominance_list = []
    delta_dominance_list = []

    observations = [obs]
    infos = [info]
    
    reward_components = {}
    for rc in test_env.reward_component_keys:
        reward_components[rc] = []
    
    while not (terminated or truncated):
        if isinstance(model, stable_baselines3.common.base_class.BaseAlgorithm):
            # model is an SB3 model
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions for testing
        elif isinstance(model, Callable):
            # model is an explicit policy
            action = model(obs)
        else:
            raise TypeError(f'model must be either an SB3 model or an explicit policy, not {type(model)}')
        obs, reward, terminated, truncated, info = test_env.step(int(action))  # Step in the environment
        episode_reward += reward
        step_count += 1
        if (not quiet) and (step_count%Nsteps_to_print_after)==0:
            print('step:', test_env._steps_taken, '\trew:', reward, '\tdom:', test_env._cur_dominance, \
          '\tddom:', test_env._cur_delta_dominance, '\tNpeaks:', info['Npeaks_found'])
        rewards.append(reward)
        action_list.append(int(action))
        dominance_list.append(test_env._cur_dominance)
        r_dominance_list.append(test_env._cur_r_dominance)
        delta_dominance_list.append(test_env._cur_delta_dominance)
        observations.append(obs)
        infos.append(info)
        
        for rc in test_env.reward_component_keys:
            reward_components[rc].append(info.get('reward/'+rc, 0))
    
    if not quiet:
        print(f"Total Episode Reward: {episode_reward}")
        if truncated:
            finish='Truncated'
        elif terminated:
            finish='Terminated'
        else:
            finish='None'
        print("Finished as", finish, 'after', test_env._steps_taken, 'steps')
        print("Final dominance of the main peak:", test_env._cur_dominance)
    
    out = {
        'terminated' : terminated,
        'truncated' : truncated,
        'episode_reward' : episode_reward,
        'rewards' : rewards,
        'action_list' : action_list,
        'dominance_list' : dominance_list,
        'r_dominance_list' : r_dominance_list,
        'delta_dominance_list' : delta_dominance_list,
        'reward_components' : reward_components,
        'observations' : observations,
        'infos' : infos
    }    
    return EpisodeResultsDict(out)

def manual_episode_init(
    env : CavityAlignmentEnv,
    quiet : bool = False
) -> EpisodeResultsDict:
    """
    Initialise an episode with manual alignment for benchmarking.
    
    Parameters
    ----------
    env : CavityAlignmentEnv
        `CavityAlignmentEnv` instance that has the gymnasium environment for the cavity model under test.
        Usually can be created using the factory method created by `calmly.get_env_factory()`
    quiet : bool
        If `True`, no output is provided to stdout. Default is `False`.

    Returns
    -------
    EpisodeResultsDict
        Dictionary containing results and metrics, plus some visualisation methods. 
    """
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0
    rewards =[]
    action_list = []
    dominance_list = []
    r_dominance_list = []
    delta_dominance_list = []
    
    observations = [obs]
    infos = [info]
    
    reward_components = {}
    for rc in env.reward_component_keys:
        reward_components[rc] = []

    out = {
        'terminated' : terminated,
        'truncated' : truncated,
        'episode_reward' : episode_reward,
        'rewards' : rewards,
        'action_list' : action_list,
        'dominance_list' : dominance_list,
        'r_dominance_list' : r_dominance_list,
        'delta_dominance_list' : delta_dominance_list,
        'reward_components' : reward_components,
        'observations' : observations,
        'infos' : infos
    }

    if not quiet:
        reward = None
        print('step:', env._steps_taken, '\trew:', reward, '\tdom:', env._cur_dominance, \
              '\tddom:', env._cur_delta_dominance, '\tNpeaks:', info['Npeaks_found'])
    return EpisodeResultsDict(out)
    

def manual_episode_step(
    env : CavityAlignmentEnv,
    diag : EpisodeResultsDict,
    action : int = None,
    quiet : bool = False
):
    """
    Take an action in an episode with manual alignment. If no `action` is given, will print the results of the last action.

    Parameters
    ----------
    env : CavityAlignmentEnv
        `CavityAlignmentEnv` instance that has the gymnasium environment for the cavity model under test.
    diag : EpisodeResultsDict
        Dictionary containing results and metrics, plus some visualisation methods. 
        This dictionary is normally returned by `manual_episode_init()`.
    action : int
        Number of the action to take.
        Default is `None`, which takes no action but prints the results of the last one (unless `quiet` is `True`).
    quiet : bool
        If `True`, no output is provided to stdout. Default is `False`.
    """
    if (action is None) and (not quiet):
        # Just print out the last results
        obs, info = env._get_obs_and_info()
        reward = None
        print('step:', env._steps_taken, '\trew:', reward, '\tdom:', env._cur_dominance, \
              '\tddom:', env._cur_delta_dominance, '\tNpeaks:', info['Npeaks_found'])
    elif action is not None:
        if diag['terminated'] or diag['truncated']:
            raise ValueError('Alignment has already been completed')
        # Take the action
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        diag['terminated'] = terminated
        diag['truncated'] = truncated
        diag['episode_reward'] += reward
        diag['rewards'].append(reward)
        diag['action_list'].append(action)
        diag['dominance_list'].append(env._cur_dominance)
        diag['r_dominance_list'].append(env._cur_r_dominance)
        diag['delta_dominance_list'].append(env._cur_delta_dominance)
        for rc in env.reward_component_keys:
            diag['reward_components'][rc].append(info.get('reward/'+rc, 0))
        diag['observations'].append(obs)
        diag['infos'].append(info)

        if not quiet:
            if terminated:
                print("======TERMINATED======")
            if truncated:
                print("======TRUNCATED=======") 
            print('step:', env._steps_taken, '\trew:', reward, '\tdom:', env._cur_dominance, \
                  '\tddom:', env._cur_delta_dominance, '\tNpeaks:', info['Npeaks_found'])
        return
