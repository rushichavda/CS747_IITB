"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np
import random

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

# class FaultyBanditsAlgo:
#     def __init__(self, num_arms, horizon, fault):
#         # You can add any other variables you need here
#         self.num_arms = num_arms
#         self.horizon = horizon
#         self.fault = fault # probability that the bandit returns a faulty pull
#         # START EDITING HERE
#         self.successes = [1]*num_arms
#         self.failures = [1]*num_arms
#         self.count = np.zeros(num_arms)
#         self.timestamp = 0

#         # END EDITING HERE
    
#     def give_pull(self):
#         # START EDITING HERE
#         if self.timestamp < self.num_arms:
#             return self.timestamp
        
#         else:
#             samples = []
#             for i in range(self.num_arms):

#                 # print(self.successes[i],self.failures[i])
#                 samples.append(np.random.beta(self.successes[i]+1, self.failures[i]))
#             samples = np.array(samples)
#             return np.argmax(samples)
    
#         # END EDITING HERE
    
#     def get_reward(self, arm_index, reward):
#         # START EDITING HERE
#         self.timestamp += 1 
#         if self.timestamp < self.num_arms:
#             if reward == 0:
#                 self.failures[arm_index] += 1
#             else:
#                 self.successes[arm_index] += 1


#         if self.timestamp >= self.num_arms:
#             prob_fault = self.fault
#             prob_fair = 1 - prob_fault

#             # Generate a random number between 0 and 1
#             random_number = random.random()

#             # Make the selection based on the probabilities
#             if random_number < prob_fault:
#                 random_select = random.random()
#                 if random_select < 0.5:
#                     self.failures[arm_index] += 1
#                 else:
#                     self.successes[arm_index] += 1

#             else:
#                 if reward == 0:
#                     self.failures[arm_index] += 1
#                 else:
#                     self.successes[arm_index] += 1
#         #END EDITING HERE

# [802.5, 1667.22, 3409.9, 6919.96, 13947.6, 28014.76, 56168.7, 112492.08, 225189.84]
import functools
def kl_div(x, y):
    x[np.abs(x-1)<1e-5] = 1 - 1e-3
    x[np.abs(x) <1e-5] = 1e-3
    y[np.abs(y-1)<1e-5] = 1 - 1e-3
    y[np.abs(y) <1e-5] = 1e-3
    ret = np.multiply(x, np.log(np.divide(x,y))) + np.multiply((1-x), np.log(np.divide(1-x, 1-y)))

    return ret

class FaultyBanditsAlgo:
#    
    def __init__(self, num_arms, horizon,fault):
        # super().__init__(num_arms, horizon)
        self.horizon = horizon
        # You can add any other variables you need here
        # START EDITING HERE
        self.c = 0
        self.fault = fault
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
        new_value = ((n - 1) / n) * value + (1 / n) * (0.8*reward + 0.1*1)
        self.mean_values[arm_index] = new_value
        