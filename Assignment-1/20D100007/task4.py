"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


# class MultiBanditsAlgo:
#     def __init__(self, num_arms, horizon):
#         # You can add any other variables you need here
#         self.num_arms = num_arms
#         self.horizon = horizon
#         # START EDITING HERE
        
#         # END EDITING HERE
    
#     def give_pull(self):
#         # START EDITING HERE
#         raise NotImplementedError
#         # END EDITING HERE
    
#     def get_reward(self, arm_index, set_pulled, reward):
#         # START EDITING HERE
#         raise NotImplementedError
#         # END EDITING HERE

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE




class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.horizon = horizon
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
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        if reward == 0:
            self.failures[arm_index] += 1
        else:
            self.successes[arm_index] += 1
        # END EDITING HERE


