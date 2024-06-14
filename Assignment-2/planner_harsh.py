import numpy as np
import os
import sys
import argparse
from numpy import argmax
from pulp import *
import functools

class MDP:
    def __init__(self, file, patience=5, eps=1e-8) -> None:
        content = file.readlines()
        self.num_states = int(content[0].split()[1])
        self.num_actions = int(content[1].split()[1])
        self.end_states = [int(elem) for elem in content[2].split()[1:]]
        self.patience = patience
        self.eps = eps
        i = 3
        self.transition_matrix = np.zeros(shape=[self.num_states, self.num_states, self.num_actions], dtype=np.float)
        self.reward_matrix = np.zeros(shape=[self.num_states, self.num_states, self.num_actions], dtype=np.float)
        while content[i].split()[0] == "transition":
            s1 = int(content[i].split()[1])
            ac = int(content[i].split()[2])
            s2 = int(content[i].split()[3])
            rew = float(content[i].split()[4])
            prob = float(content[i].split()[5])
            self.transition_matrix[s1, s2, ac] = prob
            self.reward_matrix[s1, s2, ac] = rew
            i += 1

        self.mdtype = content[i].split()[1]
        self.gamma = float(content[i+1].split()[1])
        

    
    def print_mdp_data(self):
        print("Number of states :", self.num_states)
        print("Number of action :", self.num_states)
        print("Transition matrix :", self.transition_matrix)
        print("Reward matrix :", self.reward_matrix)
        print("Mdtype :",self.mdtype)
        print("Discount factor :", self.gamma)

    def plan(self, algo, print_output=True):
        if algo == "vi":
            # Value Iteration

            operator = self.get_bellman_optimality_op()
            prev_V = np.zeros(shape=[self.num_states, 1], dtype=np.float)
            curr_V = np.zeros(shape=[self.num_states, 1], dtype=np.float)
            curr_patience = 0
            ## Value iteration 
            while curr_patience < self.patience:
                prev_V = curr_V
                curr_V = operator(curr_V)
                if np.linalg.norm(prev_V-curr_V) < self.eps:
                    curr_patience += 1
                else:
                    curr_patience = 0
            ## Policy evaluation
            policy = np.argmax(np.sum((self.gamma*(curr_V.T[:,:,None]) + self.reward_matrix)*self.transition_matrix,axis=1),axis=1)
            V = curr_V.squeeze()

            ## Printing
            for i in range(self.num_states):
                print(V[i], policy[i])
            

        elif algo == "lp":
            # Linear Programming

            problem = LpProblem('MDP_solver', LpMinimize)

            ## Adding decision variable
            decision_variables = []
            for i in range(self.num_states):
                decision_variables.append(LpVariable(f"v({i})", cat=LpContinuous))
            
            ## Objective function
            problem += lpSum(decision_variables)

            ## Adding constraints
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    vars = [(self.gamma*decision_variables[s_p] + self.reward_matrix[s, s_p, a])*self.transition_matrix[s, s_p, a] for s_p in range(self.num_states)]
                    exp = functools.reduce(lambda a, b: a + b, vars)
                    problem += exp <= decision_variables[s]
            
            ## Solving the problem
            status = problem.solve(PULP_CBC_CMD(msg = 0))
            V = []
            for i in range(self.num_states):
                V.append(value(decision_variables[i]))
            V = np.array(V)[:,None]

            ## Evaluating policy
            policy = np.argmax(np.sum((self.gamma*(V.T[:,:,None]) + self.reward_matrix)*self.transition_matrix,axis=1),axis=1)
            V = self.value_function(policy_file=None, policy=policy, print_out=False)
            
            # Printing
            for i in range(self.num_states):
                print(V[i], policy[i])
            
        elif algo == "hpi":
            # Howard's Policy Iteration
            curr_policy = np.random.randint(low=0, high=self.num_actions, size=self.num_states)    
            improvables_states = self.get_improvable_states(curr_policy)
            while sum([len(elem) for elem in improvables_states]) > 0:
                for i, elem in enumerate(improvables_states):
                    if len(elem) > 0:
                        curr_policy[i] = elem[0]
                improvables_states = self.get_improvable_states(curr_policy)
            V = self.value_function(policy_file=None, policy=curr_policy, print_out=False)
            
            ## Printing
            for i in range(self.num_states):
                print(V[i], curr_policy[i])

        else:
            raise Exception("Invalid algorithm(vi/lp/hpi)")
    
    def get_bellman_optimality_op(self):
        def ret_func(V):
            mat = np.sum((self.gamma*V.T[:,:,None] + self.reward_matrix)*self.transition_matrix,axis=1).max(axis=1)
            return mat[:,None]
        return ret_func
    
    def get_improvable_states(self, policy):
        states = []
        for i in range(self.num_states):
            V = self.value_function(policy_file=None, policy=policy, print_out=False)
            tmp_val = np.sum(np.array([(self.gamma*V[s_p] + self.reward_matrix[i, s_p, policy[i]])*self.transition_matrix[i, s_p, policy[i]] for s_p in range(self.num_states)]))
            state_elem = []
            for j in range(self.num_actions):
                curr_val = np.sum(np.array([(self.gamma*V[s_p] + self.reward_matrix[i, s_p, j])*self.transition_matrix[i, s_p, j] for s_p in range(self.num_states)]))
                if curr_val > tmp_val:
                    state_elem.append(j)
            states.append(state_elem)
        
        return states

    def value_function(self, policy_file, policy=None, print_out=True):
        if policy_file is not None:
            content = policy_file.readlines()
            policy = np.zeros(shape=[self.num_states], dtype=np.intc)
            for i in range(self.num_states):
                policy[i] = int(content[i].split()[0])
        problem = LpProblem('MDP_solver', LpMinimize)

        ## Adding decision variable
        decision_variables = []
        for i in range(self.num_states):
            decision_variables.append(LpVariable(f"v({i})", cat=LpContinuous))
        
        ## Objective function(constant)
        problem += 1 

        ## Constraints
        for s in range(self.num_states):
            vars = [(self.gamma*decision_variables[s_p] + self.reward_matrix[s, s_p, policy[s]])*self.transition_matrix[s, s_p, policy[s]] for s_p in range(self.num_states)]
            exp = functools.reduce(lambda a, b: a + b, vars)
            problem += exp >= decision_variables[s]
            problem += exp <= decision_variables[s]

        ## Solving
        status = problem.solve(PULP_CBC_CMD(msg = 0))
        V = []
        for i in range(self.num_states):
            V.append(value(decision_variables[i]))
            if print_out:
                print(V[-1], policy[i])
        
        return np.array(V)

    

def parse_arguments(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str, required=False, help='Path to mdp file')
    parser.add_argument('--algorithm', type=str, required=False, help='Algorithm used for solving the given MDP for value function and optimal policy(vi/hpi/lp)',
                         default="vi")
    parser.add_argument('--policy', type=str, required=False, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    with open(args.mdp, "r") as f:
        mdp = MDP(f)

    # if policy file path provided
    if args.policy is None:
        mdp.plan(args.algorithm)
    else:
        with open(args.policy) as f:
            mdp.value_function(f)
