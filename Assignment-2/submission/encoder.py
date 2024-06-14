import numpy as np
import os
import sys
import argparse
import random

ACTIONS = [0, 1, 2, 3, 1,5, 5, 6, 7, 8, 9]  # Adjust this for football actions
ACTION_DICT = {ac: i for i, ac in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

OUTCOMES = [-1, 0, 1, 2]  # Adjust this for football outcomes
OUTCOME_DICT = {o: i for i, o in enumerate(OUTCOMES)}
NUM_OUTCOMES = len(OUTCOMES)

class MDP_encoder:
    def __init__(self, p, q, opponent_policy, grid_size = 4, goal_length=2):
        self.grid_size = grid_size
        self.goal_length = goal_length
        self.p = p
        self.q = q
        self.opponent_policy = opponent_policy
        self.num_actions = NUM_ACTIONS

        self.opp_dict = {}  # Modify to include football state format
        self.hit = []  # Modify to include football state format
   
        def get_state_index(state):  # Modify to handle football state format
            number = str(state)
            # tuple_len  = [2,2,2,1]
            substrings = [int(number[:2]), int(number[2:4]), int(number[4:6]), int(number[6:])]
            return substrings
        state_index = np.zeros(shape = (16, 16, 16, 2))

        i = 0
        for B1_states in range(1,17):
            for B2_states in range(1,17):
                for R_states in range(1,17):
                    for ball_pos in range(1,3):
                        state_index[B1_states-1][B2_states-1][R_states-1][ball_pos-1] = int(i)
                        # print(f"{B1_states}{B2_states}{R_states}{ball_pos}")
                        i += 1
        # state_index["end"] = 8192
        # state_index["goal"] = 8193

        def state_index_i(B1_states, B2_states, R_states, ball_pos, episode = None):
            if episode == "goal":
                return 8193
            if episode == "end":
                return 8192
            else:            
                return int(state_index[B1_states-1][B2_states-1][R_states-1][ball_pos-1])
        i = True
        with open(self.opponent_policy, "r") as file:
            for line in file:
                stripped = line.strip().split()
                # print(stripped)
                if i :
                    i = False
                    continue
                # print(stripped[0])
                opp_state = get_state_index(stripped[0])
                self.opp_dict[f"{state_index_i(opp_state[0],opp_state[1],opp_state[2],opp_state[3])}"] = [float(stripped[1]),float(stripped[2]),float(stripped[3]),float(stripped[4])]
                
            
        
        self.numstates = 16*16*16*2 + 2
        self.transitions = np.zeros(shape = (self.numstates, NUM_ACTIONS,  self.numstates), dtype = np.float)
        self.rewards = np.zeros(shape = (self.numstates, NUM_ACTIONS, self.numstates), dtype = np.float)

        
        
               
        # r_actions = ["L", "R", "U", "D"]
        for B1_x in range(1,5):
            for B1_y in range(1,5):
                B1_states = B1_x + (B1_y - 1) * 4
                for B2_x in range(1,5):
                    for B2_y in range(1,5):
                        B2_states = B2_x + (B2_y - 1) * 4
                        for R_x in range(1,5):
                            for R_y in range(1,5):
                                R_states = R_x + (R_y - 1) * 4
                                for ball_pos in range(1,3):
                                    state = state_index_i(B1_states,B2_states,R_states,ball_pos)
                                    # actions = list(range(10))
                                    new_b1x, new_b1y, new_b2x, new_b2y = B1_x, B1_y, B2_x, B2_y

                                    for action in range(10):
                                          
                                        if action < 8:
                                          
                                            if action == 0:  #move B1 left
                                                new_b1x = B1_x - 1
                                                new_b1y = B1_y
                                            elif action == 1: #move b1 right
                                                new_b1x = B1_x + 1 
                                                new_b1y = B1_y                                              
                                            elif action == 2: #move b1 up
                                                new_b1y = B1_y - 1
                                                new_b1x = B1_x
                                            elif action == 3: #move b1 down
                                                new_b1y = B1_y + 1
                                                new_b1x = B1_x
                                            
                                            if action == 4: #move b2 left
                                                new_b2x = B2_x - 1
                                                new_b2y = B2_y
                                            elif action == 5: #move b2 right
                                                new_b2x = B2_x + 1
                                                new_b2y = B2_y
                                            elif action == 6: #move b2 up
                                                new_b2y = B2_y - 1
                                                new_b2x = B2_x
                                            elif action == 7: #move b2 down
                                                new_b2y = B2_y + 1
                                                new_b2x = B2_x                                           

                                            r_actprob =  self.opp_dict[str(state)] 
                                           
                                            moves = ["U", "D", "L", "R"]
                                            for i in range(len(moves)):
                                                if moves[i] == "U":
                                                    new_rx = R_x
                                                    new_ry = R_y - 1
                                                elif  moves[i] == "D":
                                                    new_rx = R_x
                                                    new_ry = R_y + 1
                                                elif  moves[i] == "L":
                                                    new_rx = R_x - 1
                                                    new_ry = R_y
                                                elif  moves[i] == "R":
                                                    new_rx = R_x + 1
                                                    new_ry = R_y 
                                                
                                                cond =  (new_b1x > 4 or new_b1y > 4 or new_b2x > 4 or new_b2y > 4 or new_b1x < 1  or new_b1y < 1 or new_b2x < 1 or new_b2y < 1)
                                                if cond:                                                
                                                    new_state = state_index_i(1,2,3,4,"end")
                                                else:
                                                    # print(True)
                                                    new_b1 = new_b1x + (new_b1y - 1) * 4
                                                    new_b2 = new_b2x + (new_b2y - 1) * 4
                                                    new_r = new_rx + (new_ry - 1) * 4
                                                    if new_r < 1 or new_r > 16 :
                                                        break
                                                    new_state = state_index_i(new_b1,new_b2,new_r,ball_pos)
                                                    
                                                
                                                if ball_pos == 1:
                                                    if action < 4:
                                                        success_prob = (1 - 2 * p) * r_actprob[i]   
                                                        failure_prob = r_actprob[i] - success_prob                      
                                                    else:
                                                        success_prob = (1 - p) * r_actprob[i]
                                                        failure_prob = r_actprob[i] - success_prob

                                                    if not cond and (new_b1 == new_r or (new_r == B1_states and new_b1 == R_states)):
                                                        success_prob = success_prob/2
                                                        failure_prob = r_actprob[i] - success_prob
                                                    
                                                    self.transitions[state][action][new_state] = success_prob                                                    
                                                    self.transitions[state][action][state_index_i(1,2,3,4,"end")] = failure_prob
                                                    self.rewards[state][action][new_state] = 0

                                                    if new_state == state_index_i(1,2,3,4,"end"):
                                                        self.transitions[state][action][new_state] = 1
                                                    
                                                if ball_pos == 2:
                                                    if action < 4:
                                                        success_prob = (1 - p) * r_actprob[i]
                                                        failure_prob = r_actprob[i] - success_prob
                                                    else:
                                                        success_prob = (1 - 2 * p) * r_actprob[i]   
                                                        failure_prob = r_actprob[i] - success_prob   

                                                    if not cond and (new_b2 == new_r or (new_r == B2_states and new_b2 == R_states)):
                                                        success_prob = success_prob/2
                                                        failure_prob = r_actprob[i] - success_prob
                                                    
                                                    self.transitions[state][action][new_state] = success_prob
                                                    self.transitions[state][action][state_index_i(1,2,3,4,"end")] = failure_prob
                                                    self.rewards[state][action][new_state] = 0  

                                                    if new_state == state_index_i(1,2,3,4,"end"):
                                                        self.transitions[state][action][new_state] = 1                                            
                                            
                                        if action == 8:
                                        
                                            pass_action = 8 
                                            
                                            r_actprob =  self.opp_dict[str(state)] 
                                            moves = ["U", "D", "L", "R"]
                                            for i in range(len(moves)):
                                                if moves[i] == "U":
                                                    new_rx = R_x
                                                    new_ry = R_y - 1
                                                elif  moves[i] == "D":
                                                    new_rx = R_x
                                                    new_ry = R_y + 1
                                                elif  moves[i] == "L":
                                                    new_rx = R_x - 1
                                                    new_ry = R_y
                                                elif  moves[i] == "R":
                                                    new_rx = R_x + 1
                                                    new_ry = R_y 
                                                pass_success_prob =  (q - 0.1 * max(abs(B1_x - B2_x), abs(B1_y - B2_y))) * r_actprob[i]
                                                pass_failure_prob =  r_actprob[i] - pass_success_prob

                                                if (new_rx - B1_x) * (B2_y - B1_y) == (new_ry - B1_y) * (B2_x - B1_x):
                                                    if (min(B1_x, B2_x) <= new_rx <= max(B1_x, B2_x) and min(B1_y, B2_y) <= new_ry <= max(B1_y, B2_y)):
                                                        pass_success_prob = 0.5 * (q - 0.1 * max(abs(B1_x - B2_x), abs(B1_y - B2_y))) * r_actprob[i]
                                                        pass_failure_prob =  r_actprob[i] - pass_success_prob

                                                
                                                new_r = new_rx + (new_ry - 1) * 4
                                                if ball_pos == 1:
                                                    new_ball_pos = 2
                                                if ball_pos == 2:
                                                    new_ball_pos = 1
                                                if new_r < 1 or new_r > 16 :
                                                        break
                                                new_state = state_index_i(B1_states,B2_states, new_r, new_ball_pos)
                                                
                                                self.transitions[state][pass_action][new_state] = pass_success_prob
                                                self.transitions[state][pass_action][state_index_i(1,2,3,4,"end")] = pass_failure_prob
                                                self.rewards[state][pass_action] = 0  
                                        
                                        if action == 9:
                                            
                                            r_actprob =  self.opp_dict[str(state)] 
                                            moves = ["U", "D", "L", "R"]
                                            for i in range(len(moves)):
                                                if moves[i] == "U":
                                                    new_rx = R_x
                                                    new_ry = R_y - 1
                                                elif  moves[i] == "D":
                                                    new_rx = R_x
                                                    new_ry = R_y + 1
                                                elif  moves[i] == "L":
                                                    new_rx = R_x - 1
                                                    new_ry = R_y
                                                elif  moves[i] == "R":
                                                    new_rx = R_x + 1
                                                    new_ry = R_y  

                                                if ball_pos == 1:
                                                    prob_shoot = (q - 0.2 * (3 - B1_x)) * r_actprob[i]
                                                    fail_shoot = r_actprob[i] - prob_shoot
                                                if ball_pos == 2:
                                                    prob_shoot = (q - 0.2 * (3 - B2_x)) * r_actprob[i]
                                                    fail_shoot = r_actprob[i] - prob_shoot
                                                if new_rx == 4 and new_ry in [2, 3]:
                                                    prob_shoot/=2
                                                    fail_shoot = r_actprob[i] - prob_shoot
                                                    
                                            self.transitions[state][action][state_index_i(1,2,3,4,"goal")] = prob_shoot
                                            self.rewards[state][action][state_index_i(1,2,3,4,"goal")] = 1
                                            self.transitions[state][action][state_index_i(1,2,3,4,"end")] = fail_shoot
                                            self.rewards[state][action][state_index_i(1,2,3,4,"end")] = 0
        
        print("numStates",self.numstates)
        print("numActions",self.num_actions)
        print("end"," ", 8193, 8192)   -> print("end", 8193, 8192)
        
        for i in range(8194):
            for j in range(8194):
                for k in range(10):
                    if self.transitions[i][k][j] != 0:
                        print(f"transition {i} {k} {j} {self.rewards[i][k][j]} {self.transitions[i][k][j]}")
        
        print("mdptype episodic")
        print("discount 1")                              





      

                        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opponent', type=str, help='Path to the opponent policy file')
    parser.add_argument('--p', type=float, help='Parameter p')
    parser.add_argument('--q', type=float, help='Parameter q')

    args = parser.parse_args()
    MDP_encoder(args.p, args.q, args.opponent)
                                          






                                            

                                            
    