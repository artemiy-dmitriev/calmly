import numpy as np
import matplotlib.pyplot as plt
import scipy

def part_autocorr(a, i_start=0, i_stop=None, norm_by_width=False):
    """This is not partial autocorrelation (PACF) but rather standard discrete autocorrelation function
    calculated at certain points (values of lag), which correspond to shift values between `i_start` and `i_stop`."""
    N = len(a)
    if i_stop is None:
        i_stop = N-1
    toeplitz_nrows = i_stop-i_start+1
    toeplitz_ncols = N - i_start
    
    # Creating an auxiliary vector padded with zeros on the left (using np.zeros() because it is faster than np.pad())
    a_aux = np.zeros(2*toeplitz_ncols - 1, dtype=a.dtype)
    a_aux[-toeplitz_ncols:] = a[:toeplitz_ncols]
    strides=a_aux.strides[0]
    # Creating a partial Toeplitz matrix using numpy's strides (fast but need to be very careful)
    a_toeplitz = np.lib.stride_tricks.as_strided(a_aux[toeplitz_ncols-1:], (toeplitz_nrows,toeplitz_ncols), (-strides,strides))
    # creating the complex conjugated vector for multiplication
    ac_v = np.conj(a[-toeplitz_ncols:])
    # Using the Einstein summation as it is faster than @
    res = np.einsum('ij,j->i', a_toeplitz, ac_v)
    # Normalisation by the width of the region with non-zero values
    if norm_by_width:
        res /= np.arange(toeplitz_ncols, toeplitz_ncols-toeplitz_nrows, -1)
    
    return res

def part_autocorr_scaled(xdata, ydata, start_x=None, end_x=None, norm_by_width=False):
    """
    Calculates the autocorrelation function for lag values between `start_x` and `end_x`.
    Returns the corresponding array of lags and the autocorrelation function values.
    If `norm_by_width`==True, then the result is divided by the amount of non-zero elements in the calculation.
    """
    if start_x is None:
        start_x = min(xdata)
    if end_x is None:
        end_x = max(xdata)
    minpos = np.argmin(np.abs(start_x-xdata))
    maxpos = np.argmin(np.abs(end_x-xdata))
    acf_x = xdata[minpos:maxpos+1]-xdata[0]
    acf = part_autocorr(ydata, i_start=minpos, i_stop=maxpos, norm_by_width=norm_by_width)
    return acf_x, acf

def find_FSR(xdata, ydata, minFSR=None, maxFSR=None, norm_by_width=False):
    """Looks for the FSR (periodicity) in `ydata` by selectively calculating the autocorrelation function
    between x values `minFSR` and `maxFSR` and locating the maximum.
    
    Returns the FSR value in the units of x.
    
    When working with continuous cavity scans, it is advisable to locate the FSR first using a wider range
    and then keep tracking it using a narrow range to speed up the calculations.
    
    If `norm_by_width`==True, then the result is divided by the amount of non-zero elements in the calculation.
    This can suppress small-lag peaks (e.g. the HOM separation) but exaggerate the peaks at multiple FSRs.
    Therefore, it should be used with maxFSR < 2*FSR where FSR is the actual FSR.

    TODO: possibly add checks if a peak at 1/2 frequency of the obtained one exists and scale to it if so.
    """
    acf_x, acf = part_autocorr_scaled(xdata, ydata, start_x=minFSR, end_x=maxFSR, norm_by_width=norm_by_width)
    return acf_x[np.argmax(acf)]

class CavityScanPreProcess():
    """
    Pre-processing the cavity scans"""
    def __init__(self,
                 params = {},
                 **kwargs
                ):
        """
        Parameters
        ----------
        params: Dictionary to update/add values to the internal parameter dictionary, `params`.
        Parameters can also be passed separately as kwargs, see below for the description of entries in this dict.
        **kwargs : 
            Other parameters to be update or stored in the internal parameter dictionary, `CavityScanPreProcess().params`.
            Note that any parameters specified as kwargs will take preference over those specified in the `params` argument.
            Parameter names and meaning as follows:
        FSRs_per_scan_upper_limit, FSRs_per_scan_lower_limit: float
            Upper/lower limit for the number of FSRs per scan that the measured FSRs are guaranteed to exceed never.
        FSR_track_range : float
            The range around the previous FSR to use for keeping track of the FSR, in the units of one FSR.
        median_filter_size_FWHM_multiplier : float
            Number of FWHMs to use as the size parameter for the median filter used to obtain the noise baseline. Default is 3.0
        median_filter_max_window_in_FSRs : float
            Max size for the median filter window in the units of one FSR.
        peak_width_upper_limit_in_FSRs : float
            The upper limit on the width of cavity peaks (for all detectable modes) in the units of one FSR. Defaiult is 0.1
        peak_height_above_noise_threshold_in_sigmas : float
            The preprocessing script will calculate the noise baseline and the standard deviation (sigma) of noise from this baseline.
            This parameter regulates the minimum height of the cavity peaks above the noise baseline in the units of sigma.
            Default is 3.0 (three sigma deviation)
        minimum_peak_width_in_data_points : int
            Minimum width of the detected peaks in the units of data points. Default is 10. 
            The resolution of scans should be selected appropriately. A warning will be given if the expected FWHM is smaller
            than this parameter.
        peak_width_lower_limit_in_FWHMs : float
            When tracking the peaks, this value sets the minimum width for the peaks
            in the units of FWHM of the highest peak during the last scan. Default is 0.5.
        signal_reference_level : float | NoneType
        """
        self.last_FSR = None
        self.last_FWHM = None

        self.params = {
            'FSRs_per_scan_upper_limit' : 5.0,
            'FSRs_per_scan_lower_limit' : 1.1,
            'FSR_track_range' : 0.4,
            'peak_width_upper_limit_in_FSRs' : 0.1,
            'median_filter_size_FWHM_multiplier' : 20,
            'median_filter_max_window_in_FSRs' : 0.2,
            'peak_height_above_noise_threshold_in_sigmas' : 3.0,
            'minimum_peak_width_in_data_points' : 4,
            'peak_width_lower_limit_in_FWHMs' : 0.5,
            'signal_reference_level' : None
        }
        self.params.update(params)
        self.params.update(kwargs)

    # TODO:
    # 1. Apply the roll so that the array starts at the largest peak before calling find_peaks -- DONE
    # 2. (can be added later) If averaging the scan, average the data at this point to smoothen the noise out -- DONE
    # 3. Concatenate the data with part of itself (peak_width_upper_limit_in_FSRs portion) -- DONE
    # 4. Find the peaks -- DONE
    # 5. Get rid of any extra peaks in the result (We are only concerned by the one that is at the beginning) -- DONE

    # TODO:
    # Add bypassing of some pre-processing steps when using the simulation to speed-up the training. -- DONE

    # TODO:
    # Potentially add a moving average filter before applying the roll to center more precisely at the highest peak.

    # TODO:
    # Identify the mode with the lowest loss in the final state to use as the utlimate reward in the simulation training
    # (penalise if a cavity has been aligned to the wrong mode)

    # TODO:
    # potentially this can be extended to align the cavity to a higher-order mode instead.
    def process_cavity_scan(self,
                            xdata : float | np.ndarray,
                            ydata : float | np.ndarray,
                            dominance_type : str = 'peak_area',
                            bypass_FSR_preprocessing : bool = False,
                            plot : bool = False
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Preprocessing the cavity scan given by `xdata` and `ydata`.
        
        Parameters
        ----------
        xdata, ydata : numpy.ndarray or list of numpy.nd_array
            If 1D arrays, a single scan is assumed.
            If list of 1D arrays, each entry in the list is a separate scan. The results will be averaged.
            The data must be organised such that `xdata` values are in the ascending order.
            `xdata` values must be evenly spread.
        bypass_FSR_preprocessing : bool
            If `True`, it is assumed that the provided cavity scans already cover exactly one FSR each.
            This skips the FSR estimation which should give a speed-up when training using simulated data.
        plot : bool
            If `True`, the processed data is plotted.

        Returns
        -------
        peak_positions : numpy.ndarray
            Positions of the peaks within the FSR (range normalised to [0,1])
            The highest peak should be at 0.
        peak_dominances : numpy.ndarray
            "Dominances" of the peaks (integrals of the peaks within FWHMs divided by the integral of the whole FSR scan y1).
            Range within [0,1].
        peak_heights : numpy.ndarray
            The heights of the peaks, normalised to the highest peak (range within [0,1])
        peak_FWHMs : numpy.ndarray
            Widths of the peaks in the units of one FSR (range within [0,1])
        x1 : numpy.ndarray
            x data of the processed scan, normalised to one FSR (range within [0,1])
        y1 : numpy.ndarray
            y data of the processed scan, normalised to the highest peak (range within [0,1])
        FSR : float
            The FSR in the units of the original x axis (provided for scaling other outputs back to the scan units).
        
        """
        if isinstance(xdata, np.ndarray) and isinstance(ydata, np.ndarray):
            _xdata = [xdata]
            _ydata = [ydata]
        elif isinstance(xdata, list) and isinstance(ydata, list):
            _xdata = xdata
            _ydata = ydata
        else:
            raise TypeError("Both xdata and ydata must be either 1D numpy arrays or lists of 1D numpy arrays.")

        last_FSR = self.last_FSR

        # If we have multiple scans, we detect the FSR in each one,
        # leave only the central part of the scan corresponding to one FSR,
        # scale the up or down aby interpolating and resampling if the number of points is different.
        # rearrange the data at each one to start at the highest peak
        # And take the average (also need to scale in case)
        new_FSR_list = []
        # When avaraging, Npoints will be the number of points used to resample the data
        # Currently the number of points in one FSR in the first scan is used
        Npoints = 0
        y1_list = []

        for x, y in zip(_xdata, _ydata):
            x_span = x[-1] - x[0]
            x_step =  x[1] - x[0]

            if bypass_FSR_preprocessing:
                # Assuming that the scans already span exactly over one FSR
                new_FSR_list.append(x_span+x_step)
                y1 = y
            else:
                # Finding the FSR
                if last_FSR is None:
                    #This is the first run
                    this_FSR = find_FSR(x, y, 
                                        minFSR=1/self.params['FSRs_per_scan_upper_limit']*x_span,
                                        maxFSR=1/self.params['FSRs_per_scan_lower_limit']*x_span,
                                        norm_by_width=False) 
                else:
                    # We have a previous FSR estimation, will only be looking around it
                    maxFSR = last_FSR*(1+self.params['FSR_track_range']/2)
                    maxFSR = maxFSR if maxFSR < 1/self.params['FSRs_per_scan_lower_limit']*x_span else \
                    1/self.params['FSRs_per_scan_lower_limit']*x_span
                    minFSR = last_FSR*(1-self.params['FSR_track_range']/2)
                    minFSR = minFSR if minFSR > 1/self.params['FSRs_per_scan_upper_limit']*x_span else \
                    1/self.params['FSRs_per_scan_upper_limit']*x_span
                    this_FSR = find_FSR(x, y,
                                        minFSR=minFSR,
                                        maxFSR=maxFSR,
                                        norm_by_width = (x_span < 2*last_FSR)
                                       )
                new_FSR_list.append(this_FSR)
                # TODO: identify and process the cases when the tracking fails 
                # (e.g. skip the scan if the FSR is very different fropm the previous values)
    
                # Slicing the data: taking one FSR in the middle of the x range
                left_pos = int(len(x)/2 - this_FSR/2/x_step)
                right_pos = int(len(x)/2 + this_FSR/2/x_step)
                # x1 = x[left_pos:right_pos] # We actually don't need this array
                y1 = y[left_pos:right_pos]

            # Finding the highest peak and rolling the array to start at it
            maxpos = np.argmax(y1)
            y1 = np.roll(y1, -maxpos)

            # # Normalising the array
            # if y1[0]!=0:
            #     y1 = y1/y1[0]

            if Npoints == 0:
                # This is the first scan, storing its number of points
                Npoints = len(y1)
                # Normalising the x range so that (0...1) is one FSR (x_i = i/Npoints, i=0...Npoints-1)
                x_range = np.arange(Npoints) / Npoints
            else:
                # Resampling the data to the first scan if the number of points does not match
                Npoints2 = len(y1)
                if Npoints2 != Npoints:
                    y1 = np.interp(x_range, np.arange(Npoints2)/Npoints2, y1)

            # Appending the y1 array to the list
            y1_list.append(y1)

        # Now we have a normalised x_range and a list of y-value of scans
        # where each scan should start at the highest peak and span over one FSR
        # Now we can average these scans
        y1 = np.mean(np.stack(y1_list), axis=0)
        new_FSR = sum(new_FSR_list)/len(new_FSR_list)
        x1 = x_range # normalised so that FSR = 1

        # Normalising the y array
        if self.params['signal_reference_level'] is not None:
            y1 = y1 / self.params['signal_reference_level']
        else:
            if y1[0]!=0:
                y1 = y1 / y1[0]
            else:
                y1 = y1 / max(y1)

        # TODO: this is where we can identify problems with some of the scans
        # such as centring on the wrong peak (likely if two peaks have similar height)
        # E.g. we can calculate the correlation between y1 and each of the arrays in y1_list,
        # and if there are outlier arrays with low correlation, throw them out and recalculate y1
        # This will likely be a huge processing bottleneck when using the simulated data,
        # so it should only be applied to the real data after the sim-to-real transition

        # Obtaining the noise baseline
        if self.last_FWHM is None:
            expected_FWHM_size = int(self.params['peak_width_upper_limit_in_FSRs'] * len(x1))
        else:
            expected_FWHM_size = int( self.last_FWHM/self.last_FSR * len(x1) )
        
        noise_baseline_size = int(expected_FWHM_size * self.params['median_filter_size_FWHM_multiplier'])
        if noise_baseline_size > self.params['median_filter_max_window_in_FSRs']*len(x1):
            noise_baseline_size = int( self.params['median_filter_max_window_in_FSRs']*len(x1) )

        noise_baseline_size = max(1, noise_baseline_size) # it should not be zero            
        noise_baseline = scipy.ndimage.median_filter(y1, size=noise_baseline_size)

        # Figuring out the height threshold above the baseline
        # Step 1: subtracting the baseline
        diff = y1-noise_baseline
        # Step 2: removing the 3-sigma outliers, e.g. the peaks
        sigma0 = diff.std()
        diff1 = diff[np.abs(diff)<3*sigma0]
        # Step 3: recalculating sigma
        sigma1 = diff1.std()
        # Step 4: calculating the height threshold
        peak_height_low = noise_baseline+self.params['peak_height_above_noise_threshold_in_sigmas']*sigma1

        # Setting the low limit for the peaks' width
        if self.last_FWHM is not None:
            if (expected_FWHM_size*self.params['peak_width_lower_limit_in_FWHMs'] < self.params['minimum_peak_width_in_data_points']):
                # warnings.warn(f"""
                # Expected FWHM ({int(expected_FWHM_size)} pts, {self.last_FWHM} Hz) is too narrow.
                # Use with caution.
                # Increase of resolution is advised if possible.
                # """)
                peak_width_low = int(expected_FWHM_size*self.params['peak_width_lower_limit_in_FWHMs'])
            else:
                peak_width_low = self.params['minimum_peak_width_in_data_points']
        else:
            peak_width_low = self.params['minimum_peak_width_in_data_points']

        # Concatenating the y array with a small portion of itself to ensure the highest peak is handled correctly
        Npoints_add = int(self.params['peak_width_upper_limit_in_FSRs']*len(y1))
        y1c = np.concatenate((y1[-Npoints_add:], y1))
        peak_height_low_c = np.concatenate((peak_height_low[-Npoints_add:], peak_height_low))
                          
        # Finding the peaks
        peaks,props = scipy.signal.find_peaks(
            y1c,
            height = peak_height_low_c,
            width = peak_width_low
        )

        # Deleting any extra peaks found in the concatenated part on the left
        delete_pos = np.where(peaks < Npoints_add)
        peaks = np.delete(peaks, delete_pos)
        for k, v in props.items():
            props[k] = np.delete(v, delete_pos)
        
        Npeaks = len(peaks)

        # Calculating the dominances
        peak_dominances=np.zeros(Npeaks)
        if dominance_type == 'peak_area':
            # Dominances defined as integrals over FWHMs of each peak divided by the inegral over the whole FSR (sum of y1)
            sum_y1 = y1.sum()
            for k in range(Npeaks):
                peak_dominances[k] = y1c[int(props['left_ips'][k]):int(props['right_ips'][k])+1].sum() / sum_y1
        elif dominance_type == 'peak_height':
            # Dominances defined as peak heights divided by the refernce level
            # If the signal reference was specified, y1 will be already normalised by it by now
            if self.params['signal_reference_level'] is None:
                raise ValueError("signal_reference_level in self.params must be initialised if using peak heights as dominances.")

            peak_dominances[:] = props['peak_heights'][:]
        else:
            raise ValueError("Unsupported dominance type.")

        # Converting the peak positions and other properties back to correspond to array y1
        peaks = peaks - Npoints_add
        for k, v in props.items():
            # The following entries have the meaning of positions and need to be converted back to y1
            if k in ['left_ips', 'right_ips', 'left_bases', 'right_bases', 'left_edges', 'right_edges']:
                props[k] = v - Npoints_add

        # TODO: We should have a peak at y1[0] now. If we do not, then something's wrong.
        # Might be worth adding a check and a warning or drop the processing if this happened
        
        # We want to return the peaks sorted by height
        sort_p = props['peak_heights'].argsort()[::-1]
        props = {k: v[sort_p] for k, v in props.items()}
        peaks = peaks[sort_p]

        # Peak positions in fractions of the FSR
        peak_positions = peaks/len(y1)

        # Peak FWHMs in fractions of the FSR
        peak_FWHMs = props['widths']/len(y1)

        # Calculating the new FWHM of the main peak in Hz
        if Npeaks>0:
            new_FWHM = props['widths'][0]*new_FSR/len(y1)
        else:
            new_FWHM = np.nan

        if plot:
            plt.figure()
            plt.plot(x1,y1)
            plt.plot(x1,noise_baseline, ls="--")
            plt.plot(x1,noise_baseline+self.params['peak_height_above_noise_threshold_in_sigmas']*sigma1, ls=":")
            for peak in peaks:
                plt.axvline(x1[peak], ls="--")
            plt.xlabel('Scan variable (units of FSR)')
            plt.ylabel('Normalised cavity scan')
            plt.show()

        # Updating the values used for tracking
        self.last_FSR = new_FSR if new_FSR == new_FSR else None
        self.last_FWHM = new_FWHM if new_FWHM == new_FWHM else None

        return peak_positions, peak_dominances, props['peak_heights'], peak_FWHMs, x1, y1, new_FSR