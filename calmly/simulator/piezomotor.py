import numpy as np
import finesse

class Piezomotor():
    """This class simulates the action of piezoelectric stick-slip motors"""
    def __init__(self,
                 axis,
                 pos_coef,
                 neg_coef=None,
                 uncertainty=0.1,
                 initial_counts=0,
                 low_limit=None,
                 high_limit=None,
                 numpy_rng = None
                ):
        """
        Parameters
        ----------
        axis : finesse.parameter.Parameter or None
            the axis in a Finesse model to which this picomotor is attached.
            If `None`, the object operates as a standalone motor.
        pos_coef, neg_coef : float
            transfer coefficients for the angular motion in positive and negative direction in rad/cnt.
            if `neg_coef` is `None` (default), then both coefficients are assumed to be equal
        uncertainty : float
            standard deviation of the Gaussian distribution that describes the random deviation of the transfer coefficients
            in the units of the mean values (`pos_coef` and `neg_coef`)
        initial_counts : int
            initial value for the motor's step counter. Default is 0.
        low_limit : int | NoneType = None
        high_limit : int | NoneType = None
            Limits for the counter that the motor cannot exceed.
        numpy_rng : numpy.random._generator.Generator
            Numpy random number generator to use. If `None` (default), a new one will be initialised.
        """
        if neg_coef is None:
            neg_coef = pos_coef

        self.axis = axis
        self.pos_coef = pos_coef
        self.neg_coef = neg_coef
        self.uncertainty = uncertainty
        self.low_limit = low_limit
        self.high_limit = high_limit
        # Truncating the initial counter value if outside the bounds
        if (low_limit is not None) and (initial_counts < low_limit):
            initial_counts = low_limit
        elif (high_limit is not None) and (initial_counts > high_limit):
            initial_counts = high_limit
        self.counts = initial_counts
        if numpy_rng is None:
            self.numpy_rng = np.random.default_rng()
        else:
            self.numpy_rng = numpy_rng

    def move_by(self, N):
        """
        Move the motor by N counts.
        
        Parameters
        ----------
        N : int
            Number of counts by which the motor will be moved.
            
        Returns
        -------
        res : float
            Resulting angular change in rad.
        lim : list[bool]
            A 2-element list of boolean values indicating if the low limit (first element)
            or the high limit (second element) for the motor counter were exceeded.
        """

        if not isinstance(N, int):
            raise TypeError("The number of counts must be an integer!")
        
        # Step-count-to-angle coefficient
        coef = self.neg_coef if N<0 else self.pos_coef

        new_counter = self.counts + N

        # Checking the limits
        if (self.low_limit is not None) and (new_counter<self.low_limit):
            new_counter = self.low_limit
            low_limit_reached = 1
        else:
            low_limit_reached = 0

        if (self.high_limit is not None) and (new_counter>self.high_limit):
            new_counter = self.high_limit
            high_limit_reached = 1
        else:
            high_limit_reached = 0
        
        # Adding the uncertainty
        coef = self.numpy_rng.normal(loc=coef, scale=coef*self.uncertainty)

        #Obtaining the angle
        res = coef*N
        if self.axis is not None:
            self.axis.value += res
        self.counts = new_counter
        return res, [low_limit_reached, high_limit_reached]
        
    def reset(self, N=0):
        """Reset the motor's step counter to N. Default is zero"""
        if (self.low_limit is not None) and (N < self.low_limit):
            N = self.low_limit
        elif (self.high_limit is not None) and (N > self.high_limit):
            N = self.high_limit
        self.counts = N
    
    def get_counter(self, normalised=False):
        """
        Returns the current value of the motor's step counter.
        If `normalised` is set to `True`, will map the space between the low and the high limits to the [0, 1] interval
        and return the current position within it.
        """

        counts = self.counts
        if normalised:
            if (self.low_limit is None) or (self.high_limit is None):
                raise Exception("Cannot get a normalised counter value if either of the motor axis limits is not set.")
            else:
                A = 1 / (self.high_limit - self.low_limit)
                B = - self.low_limit / (self.high_limit - self.low_limit)
                counts_normalised = A * self.counts + B
            return counts_normalised
        else:
            return counts

    def connect_numpy_rng(self, 
                          numpy_rng : np.random._generator.Generator 
                         ) -> None:
        "Updates the Numpy generator used by the motor."
        self.numpy_rng = numpy_rng