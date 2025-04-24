import numpy as np
import matplotlib.pyplot as plt
import finesse

from .piezomotor import Piezomotor

class CavityAlignment():
    DEFAULT_SCAN_RANGE_IN_FSRS = 2.1 # scan range expressed in the units of one FSR
    DEFAULT_MAXTEM = 5 # maximum order of TEM modes to model. 
    def __init__(self,
                 finesse_model,
                 cavity,
                 steering_mirror_1, steering_mirror_2,
                 scan_readout,
                 scan_actuator,
                 numpy_rng : np.random._generator.Generator = None
                ):
        """
        Parameters
        ----------
        finesse_model : finesse.model.Model
            Finesse model of the system, must at least include the cavity, two steering mirrors, and one readout.
        cavity : finesse.components.cavity.Cavity
            The cavity to be aligned (must be part of `finesse_model`)
        steering_mirror_1, steering_mirror_2 : finesse.components.beamsplitter.Beamsplitter
            Steering mirrors to align and translate the beam
        scan_readout : finesse.detectors.powerdetector.PowerDetector
            Finesse detector used to produce the cavity scans
        scan_actuator : finesse.components.laser.Laser or finesse.components.beamsplitter.Beamsplitter or finesse.components.mirror.Mirror
            Finesse component (Laser or mirror-like) that is being actuated on to produce cavity scans.
        numpy_rng : numpy.random._generator.Generator
            Numpy random number generator to use. If `None` (default), a new one will be initialised.
        """
        self.m   = finesse_model
        self.cav = cavity
        self.sm1 = steering_mirror_1
        self.sm2 = steering_mirror_2
        self.r   = scan_readout
        self.act = scan_actuator
        self.steering_motors = {}

        if self.m.Nhoms == 1:
            """If the current maxtem is zero, we should switch to the default value"""
            self.set_maxtem()

        if numpy_rng is None:
            self.numpy_rng = np.random.default_rng()
        else:
            self.numpy_rng = numpy_rng
        
        self.settings = {
            'noise_mean' : 1.0e-3,
            'noise_std' : 3.0e-4,
            'signal_norm_level' : 1.2 # normalisation factor with respect to the max readout value
        }
        
        # signal reference
        self.signal_ref = self.get_signal_reference(self.settings['signal_norm_level'])
        
        self.misalignment_axes = {}

    @classmethod
    def two_mirror_cavity(cls,
                          Lcav = 1.0,
                          Rc = (np.inf, 3.0),
                          Refl = (0.98, 0.98),
                          Loss = (0.0, 0.0),
                          StSp = (1.0, 2.0),
                          maxtem = None,
                          numpy_rng = None
                         ):
        """
        Alternative constructor which automatically creates a simple predefined Finesse model of a 2-mirror optical cavity
        with a laser and two steering mirrors (SM1 and SM2) at given distances before the cavity, each at 45 deg angle of incidence.
        
        Parameters
        ----------
        Lcav : float
            Cavity length in metres.
        Rc : tuple of float
            2-tuple of the radii of curvature (RoCs) of the input coupler (IC) and the output coupler (OC) of the cavity.
            Default is (np.inf, 3.0).
        Refl : tuple of float
            2-tuple of reflection coefficients of IC and OC. Default is (0.98, 0.98).
        Loss : tuple of float
            2-tuple of loss coefficients of IC and OC. Default is (0.0, 0.0).
        StSp : tuple of float
            2-tuple containing the distance between steering mirror 1 (SM1) and SM2 as the zeroth element
            and the distance between SM2 and IC as the last element.
            Default is (1.0, 2.0).
        maxtem : int
            Maximum order of TEM modes to include in the simulation. If `None`, DEFAULT_MAXTEM will be used.
            Default is None.
        numpy_rng : numpy.random._generator.Generator
            Numpy random number generator to use. If `None` (default), a new one will be initialised.
        """
        
        model = finesse.Model()
        model_script = f"""
        l LASER
        s sLASER LASER.p1 SM1.p1
        bs SM1 R=1 T=0 alpha=45
        s s1 SM1.p2 SM2.p1 L={StSp[0]}
        bs SM2 R=1 T=0 alpha=45
        s s2 SM2.p2 IC.p1 L={StSp[1]}
        m IC R={Refl[0]} L={Loss[0]} Rc={Rc[0]}
        s sCAV IC.p2 OC.p1 L={Lcav}
        m OC R={Refl[1]} L={Loss[1]} Rc={Rc[1]}
        pd TRAN OC.p2.o
        pd REFL IC.p1.o
        
        cav CAV IC.p2
        """
        
        model.parse(model_script)
        if maxtem is not None:
            model.modes(maxtem=maxtem)
        
        return cls(model, model.CAV, model.SM1, model.SM2, model.TRAN, model.LASER, numpy_rng=numpy_rng)

    def get_signal_reference(self, 
                            level : float = 1.0
                           ) -> float:
        """
        Locks to the lowest mode and looks at the signal at the readout assuming plane waves.
        This signal level is then used to normalise the noise in `cavity_scan()`.
        Parameters
        ----------
        level : float
            Additional normalisation factor for the reference. Default is 1.0
        """
        # TODO: instead of using plane waves, this can be improved to lock to a selected mode using PseudoLockCavity as the lock step

        xrtol = 1e-10/self.cav.finesse # relative x tolerance in the units of one FSR
        
        if isinstance(self.act, finesse.components.laser.Laser):
            xatol = self.cav.FSR * xrtol # absolute x-tolerance
        else:
            # TODO: add actuation on the phi of mirrors and beamsplitters
            raise NotImplementedError("Unsupported actuator type")

        # Temporarily storing the maxtem number and switching to the plane wave representation
        orig_maxtem = int(self.m.modes().flatten().max())
        self.m.modes('off')

        # lock and action
        lock = finesse.analysis.actions.Maximize(self.r.name, self.act.f, xatol=xatol) # can use PseudoLockCavity instead
        action = finesse.analysis.actions.Noxaxis(pre_step=lock, name='noxaxis')
        out = self.m.run(action)
        
        # Restoring the original mode configutration
        self.m.modes(maxtem=orig_maxtem)

        return level * out[self.r.name]
        

    def add_misalignment_axes(self,
                              parameter_list : list[finesse.parameter.Parameter | str],
                              min_value_list : list[float],
                              max_value_list : list[float]
                             ) -> None:
        """
        Define axes (usually xbeta and ybeta of cavity mirrors) that are used to provide
        a random misalignment of the cavity in simulations.
        """
        for axis,min_value,max_value in zip(parameter_list, min_value_list, max_value_list):
            self.add_misalignment_axis(axis, min_value, max_value)
    def add_misalignment_axis(self,
                              axis : finesse.parameter.Parameter | str,
                              min_value : float,
                              max_value : float
                             ) -> None:
        """
        Defines one axis (usually xbeta or ybeta of one of the cavity mirrors) that is used to provide
        a random misalignment of the cavity in simulations.
        """

        misalignment_axis_dict = {}
        if isinstance(axis, finesse.parameter.Parameter):
            axis_obj = axis
            axis_name = axis.full_name
        elif isinstance(axis, str):
            axis_name = axis
            axis_obj = self.get_parameter_by_full_name(axis)
        else:
            raise TypeError(f"`axis` must be a string or a Finesse parameter, not {type(axis)}.")

        self.misalignment_axes[axis_name] = (axis_obj, min_value, max_value)
    
    def misalign_axes(self,
                      *args : finesse.parameter.Parameter | str
                     ) -> None:
        """
        Misalign axes of cavity components. The components need to be registered 
        by `add_misalignment_axes()` or `add_misalignment_axis()` first.
        If no arguments are given, all registered misalignment axes will be randomly misaligned.
        If arguments are given, only axes corresponding to the arguments will be misaligned.
        
        Parameters
        ----------
        *args : finesse.parameter.Parameter | str
            Axes to be misaligned. Each entry can be either the corresponding Finesse parameter or its full name.
            Parameters must be already registered in self.misalignment_axes.
        """
        if len(args) == 0:
            args = [k for k in self.misalignment_axes.keys()]
        for axis in args:
            if isinstance(axis, str):
                axis_key = axis
            elif isinstance(axis, finesse.parameter.Parameter):
                axis_key = axis.full_name
            else:
                raise TypeError(f"`axis` must be a string or a Finesse parameter, not {type(axis)}.")

            axis_obj, min_value, max_value = self.misalignment_axes[axis_key]
            axis_obj.value = self.numpy_rng.uniform(low = min_value, high=max_value)

    def set_maxtem(self, maxtem=None):
        """Sets the maximum order of the modes to model in the cavity to maxtem.
        If maxtem is not specified then the default value from CavityAlignment.DEFAULT_MAXTEM is returned."""
        if maxtem is None:
            maxtem = self.DEFAULT_MAXTEM
        
        self.m.modes(maxtem=maxtem)

    def cavity_scan(self,
                    scan_range_in_fsrs=None,
                    npoints=None, ppbw_resolution=None, ppfsr_resolution=None,
                    noisy=False,
                    plot=False, 
                    **kwargs
                   ):
        """Returns data of a cavity scan"""
        if scan_range_in_fsrs is None:
            scan_range_in_fsrs = self.DEFAULT_SCAN_RANGE_IN_FSRS
        scan_range = scan_range_in_fsrs * self.cav.FSR

        mask = np.array([npoints is not None, ppbw_resolution is not None, ppfsr_resolution is not None])
        if mask.sum() != 1:
            raise TypeError("Exactly one of the parameters `npoints`, `ppbw_resolution`, ppfsr_resolution` must be specified")

        if ppbw_resolution is not None:
            npoints = int(ppbw_resolution * self.cav.finesse * scan_range_in_fsrs)
        elif ppfsr_resolution is not None:
            npoints = int(ppfsr_resolution * scan_range_in_fsrs)

        if isinstance(self.act, finesse.components.laser.Laser):
            # Actuating on the laser frequency
            freq_start = 0 # do we ever need it to be non-zero?
            freq_stop  = freq_start + scan_range
            freqs = np.linspace(freq_start, freq_stop, npoints, endpoint=False)
            action = finesse.analysis.actions.Sweep(
                self.act.f,
                freqs,
                False
            )
        else:
            # TODO: add actuation on the phi of mirrors and beamsplitters
            raise NotImplementedError("Unsupported actuator type")
            
        out = self.m.run(action)
        x = out.x[0]
        y = out[self.r.name]

        if noisy:
            # adding noise, currently just a stationary gaussian noise
            # TODO: instead use Finesse to generate realistic noise with given entry points and PSDs.
            y += np.abs( self.numpy_rng.normal(
                loc = self.signal_ref * self.settings['noise_mean'],
                scale = self.signal_ref * self.settings['noise_std'],
                size = len(y)
            ) )

        if plot:
            out.plot(self.r.name, **kwargs)

        return x, y

    def get_all_steering_motor_counters(self, normalised=False):
        """
        Returns the current value of the step counters of all motors.
        If `normalised` is set to `True`, will map the space between the low and the high limits of each motor
        to the [0, 1] interval and return the current position within it.
        """
        counters = []
        params = [self.sm1.xbeta, self.sm1.ybeta, self.sm2.xbeta, self.sm2.ybeta]
        for param in params:
            axis_name = param.full_name
            counters.append( self.steering_motors[axis_name].get_counter(normalised=normalised) )
        return counters
        
    def init_steering_motor(self, axis, pos_coef, **kwargs):
        """
        Initialise a steering motor for angular degree of freedom `axis`.
        `pos_coef` and `**kwargs` are passed to `Piezomotor()`.
        """
        name = axis.full_name
        if not ('numpy_rng' in kwargs):
            kwargs.update({'numpy_rng':self.numpy_rng})
        motor = Piezomotor(axis, pos_coef=pos_coef, **kwargs)
        d = {name : motor}
        self.steering_motors.update(d)

    def init_all_steering_motors(self, pos_coef, **kwargs):
        """
        Initialise identical steering motors for all angular degrees of freedom of the two steering mirrors.
        `pos_coef` and `**kwargs` are passed to `Piezomotor()`.
        """
        params = [self.sm1.xbeta, self.sm1.ybeta, self.sm2.xbeta, self.sm2.ybeta]
        for param in params:
            self.init_steering_motor(param, pos_coef=pos_coef, **kwargs)
    
    def reset_steering_motors(self,
                              motor_list : list[str] | None = None,
                              value_list : list[float] | None = None
                             ) -> None:
        """
        Resets the counters of the steering motors from `motor_list` to `value_list`.
        Note that this does not reset the actual angles on steering mirrors;
        to reset these, use `self.set_steering_angles)
        If `motor_list` is not specified, all steering motors will be reset.
        If `value_list` is not specified, the counters will be reset to zeroes.
        
        """
        if motor_list is None:
            motor_list = [k for k in self.steering_motors]
        Nm = len(motor_list)
        if value_list is None:
            value_list=[0]*Nm
        for motor, v in zip(motor_list, value_list):
            self.steering_motors[motor].reset(N=v)

    def get_parameter_by_full_name(self, full_name):
        """Returns a parameter in the model given by its full name (i.e. as 'component_name.parameter_name')"""
        component_str, parameter_str = full_name.split('.')
        component = self.m.get_element(component_str)

        param=component.parameters[[param.name for param in component.parameters].index(parameter_str)]
        return param

    def move_steering_motors(self, counts):
        """
        `counts` must be a dictionary with keys corresponding to the axis names of the motors
        and values containing the number of counts by which the motor needs to be moved.
        Will move all motors specified in `counts` one by one.

        Returns a dictionary with the same keys as `counts` containing 2-element bool lists,
        where the elements indicate if the corresponding motor has reached the end-of-travel
        low or high limit, respectively.
        """
        lim_dict= {}
        for axis_name, N in zip(counts.keys(), counts.values()):
            lim = self.move_steering_motor(axis_name, N)
            lim_dict[axis_name] = lim
        return lim_dict

    def move_steering_motor(self, axis, N):
        """
        Moves a steering motor attached to `axis` by N counts.
        Returns a list of two binary elements indicating if the motor hit the low or the high limit, respectively
        """
        if isinstance(axis, str):
            axis_name = axis
        elif isinstance(axis, finesse.parameter.Parameter):
            axis_name = axis.full_name
        else:
            raise TypeError(f"`axis` must be a string or a Finesse parameter, not {type(axis)}.")
        
        _, lim = self.steering_motors[axis_name].move_by(N)
        return lim

    def dof_info(self, axis):
        """
        Return a string with information about current value of an angular dof `axis` and motor count
        """
        axis_name = axis.full_name

        out = f"{axis_name} = {axis.value} {axis.units}, "
        if axis_name in self.steering_motors:
            out += f"motor pos = {self.steering_motors[axis_name].counts}"
        else:
            out += "no motor attached"
        return out
        

    def get_convergence_curves(self,
                               axis,
                               x,
                               maxtem_list,
                               cavity_leak_channels,
                               leak_channel_names,
                               lock_to_mode=[0,0],
                               plot=None,
                               plot_logx=False,
                               quiet=False):
        """
        Returns and plots some curves to help estimate the required amount of TEM modes to include in the simulation.
        
        Parameters
        ----------
        axis : finesse.parameter.Parameter
            Axis to misalign (typically xbeta or ybeta of one of the mirrors).
        x : numpy.ndarray
            Array of values to sweep the axis parameter.
        maxtem_list : list of int
            List of the maximum order of TEM modes to include in the simulation for each produced curve
        cavity_leak_channels : list of finesse.components.node.OpticalNode
            list of optical nodes through which the energy flows out of the cavity (e.g. reflection and transmission).
            Temporary detectors will be placed at these nodes.
        leak_channel_names : list of str
            Names of the temporary detectors (also appear on the generated plots). At the moment, nust not coincide with
            any previously added detectors (even if they share the same node).
        lock_to_mode : list of 2 ints
            TEM mode order to whose resonance the cavity will be tuned at each step point of the simulation.
            Default is [0,0] (TEM00).
        plot : str or None
            Possible options are: "all". If "all", will plot the sum of all temporary readouts as the first subplot
            followed by all temporary readouts, each in a separate subplot.
            Default is `None`, which does not plot anything.
        plot_logx : bool
            If `True`, the x axis on each plot will be scaled logarithmically. Default is `False`.
        quiet : bool
            If `True`, will print a message after finisshing the calculation of each entry in `maxtem_list`.
            Default is `True`.
            
        Returns
        -------
        xlist : list of numpy.ndarray
            each numpy array contains values of the x-axis parameter for the simulation with the corresponding value
            from `maxtem_list`.
        readouts: dict
            A dictionary with keys corresponding to the leak channel names.
            Each dictionary value contains a list of numpy arrays with readout results
            that correspond to different values from `maxtem_list`
        """
        orig_angles = (self.sm1.xbeta, self.sm1.ybeta, self.sm2.xbeta, self.sm2.ybeta)
        orig_maxtem = int(self.m.modes().flatten().max())
        self.set_steering_angles()
        pd_script = ""

        xlist = []
        readouts = {}
        
        for node, name in zip(cavity_leak_channels, leak_channel_names):
            pd_script += f"pd {name} {node.full_name}\n"
            readouts[name] = []
            
        self.m.parse(pd_script)
        
        for maxtem in maxtem_list:
            self.m.modes(maxtem=maxtem)
            # lock = finesse.analysis.actions.Maximize(self.r.name, self.act.f)
            lock = finesse.analysis.actions.PseudoLockCavity(self.cav, mode=lock_to_mode)
            action = finesse.analysis.actions.Sweep(
                axis,
                x,
                False,
                pre_step=lock,
                reset_parameter=True
            )
            out = self.m.run(action)

            xlist.append(out.x[0])
            for name in leak_channel_names:
                readouts[name].append(out[name])
        
            if not quiet:
                print(f"maxtem={maxtem} calculation finished")

            
        for node, name in zip(cavity_leak_channels, leak_channel_names):
            self.m.remove(name)
        self.set_steering_angles(orig_angles)
        self.m.modes(maxtem=orig_maxtem)

        if plot == "all":
            plotfunc = plt.semilogx if plot_logx else plt.plot
            nplots = len(leak_channel_names) + 1
            
            plt.figure(figsize = (6,3*nplots))
            plt.subplot(nplots,1,1)
            for j,maxtem in enumerate(maxtem_list):
                y = np.sum([arr_list[j] for arr_list in readouts.values()], axis=0)
                plotfunc(xlist[j], y,label=f"maxtem={maxtem}")
            plt.xlabel(f"{axis.full_name} angle, rad")
            plt.ylabel("Sum of all leak channels")
            plt.title("Convergence breakdown")
            plt.legend()
                
            for i,name in enumerate(leak_channel_names):
                plt.subplot(nplots, 1, i+2)
                for maxtem,x,y in zip(maxtem_list, xlist, readouts[name]):
                    plotfunc(x, y,label=f"maxtem={maxtem}")
                plt.legend()
                plt.xlabel(f"{axis.full_name} angle, rad")
                plt.ylabel(name)
            
            plt.tight_layout()
            plt.show()

        return xlist, readouts

    def set_steering_angles(self, angles=(0.0,0.0,0.0,0.0)):
        """
        Sets the steering angles sm1.xbeta, sm1.ybeta, sm2.xbeta, sm2.ybeta
        according to angles which must be a tuple of 4 values (in the same order).
        Default is all zeros.
        """
        self.sm1.xbeta, self.sm1.ybeta, self.sm2.xbeta, self.sm2.ybeta = angles
    def connect_numpy_rng(self, 
                          numpy_rng : np.random._generator.Generator 
                         ) -> None:
        "Updates the Numpy generator used by the model and any connected motors."
        self.numpy_rng = numpy_rng
        for motor in self.steering_motors.values():
            motor.connect_numpy_rng(numpy_rng)