"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
import functools
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def kl_div(x, y):
    x[np.abs(x-1)<1e-5] = 1 - 1e-3
    x[np.abs(x) <1e-5] = 1e-3
    y[np.abs(y-1)<1e-5] = 1 - 1e-3
    y[np.abs(y) <1e-5] = 1e-3
    ret = np.multiply(x, np.log(np.divide(x,y))) + np.multiply((1-x), np.log(np.divide(1-x, 1-y)))

    return ret

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.total_count = 0
        self.ucb = np.zeros(num_arms)
        self.mean_values = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.total_count < self.num_arms:
            return self.total_count
        else:
            return np.argmax(self.ucb)
        
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_count += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        temp = self.mean_values[arm_index]
        new_mean = ((n - 1) / n) * temp + (1 / n) * reward
        self.mean_values[arm_index] = new_mean

        if self.total_count >= self.num_arms:
            self.ucb= self.mean_values + np.sqrt(np.divide(2*np.log(np.ones(self.num_arms)*self.total_count),self.counts))

        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.c = 0
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.mean_values = np.zeros(num_arms)
        self.kl_ucb_values = np.zeros(num_arms)
        self.rhs_const = np.inf
        self.total_count = 0
        self.eps = 1e-9
        # END EDITING HERE
    def bisection_method(self, f, target, x_low, x_high, err=1e-5):

        MAX_ITERATIONS = 10
        curr_iter = 0
        while curr_iter < MAX_ITERATIONS:
            curr_iter += 1
            mid = (x_low + x_high)/2
            if np.all(mid - x_low < err):
                return mid
            bool_arr = f(mid) > target
            x_high[bool_arr] = mid[bool_arr]
            x_low[np.invert(bool_arr)] = mid[np.invert(bool_arr)]
        return mid

    def give_pull(self):
        if self.total_count < 3*self.num_arms:
            return self.total_count%self.num_arms
        else:  
            self.rhs_const = np.divide(np.ones(self.num_arms)*np.log(self.total_count) + self.c*np.log(np.log(np.ones(self.num_arms)*self.total_count)), self.counts)

            self.kl_ucb_values = self.bisection_method(f = functools.partial(kl_div, np.copy(self.mean_values)),
                                                        target = self.rhs_const,
                                                        x_low =  np.copy(self.mean_values),
                                                        x_high = np.ones(self.num_arms)
                                                        )

            return np.argmax(self.kl_ucb_values)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_count += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.mean_values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.mean_values[arm_index] = new_value
        
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.num_arms = num_arms
        self.successes = [0]*num_arms
        self.failures = [0]*num_arms

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = []
        for i in range(self.num_arms):
            samples.append(np.random.beta(self.successes[i]+1, self.failures[i]+1))
        samples = np.array(samples)
        return np.argmax(samples)
    
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 0:
            self.failures[arm_index] += 1
        else:
            self.successes[arm_index] += 1
        # END EDITING HERE

