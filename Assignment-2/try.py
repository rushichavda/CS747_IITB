import random
import numpy as np
opp_dict = {} 
def get_state_index(state):  # Modify to handle football state format
    number = str(state)
    # tuple_len  = [2,2,2,1]
    substrings = [int(number[:2]), int(number[2:4]), int(number[4:6]), int(number[6:])]
    return substrings
p = 0.1
q = 0.7 
state_index = np.zeros(shape = (16, 16, 16, 2))
i = 0
for B1_states in range(1,17):
    for B2_states in range(1,17):
        for R_states in range(1,17):
            for ball_pos in range(1,3):
                state_index[B1_states-1][B2_states-1][R_states-1][ball_pos-1] = int(i)
                i += 1
# state_index["end"] = 8192
# state_index["goal"] = 8193
numstates = 16*16*16*2 + 2
transitions = np.zeros(shape = (numstates, 10,  numstates), dtype = np.float)
rewards = np.zeros(shape = (numstates, 10, numstates), dtype = np.float)
def state_index_i(B1_states, B2_states, R_states, ball_pos, episode = None):
    if episode == "goal":
        return 8193
    if episode == "end":
        return 8192
    else:            
        return int(state_index[B1_states-1][B2_states-1][R_states-1][ball_pos-1])
i = True
opponent_policy = r"data\football\test-1.txt"
with open(opponent_policy, "r") as file:
    for line in file:
        stripped = line.strip().split()
        # print(stripped)
        if i :
            i = False
            continue
        # print(stripped[0])
        opp_state = get_state_index(stripped[0])
        opp_dict[f"{state_index_i(opp_state[0],opp_state[1],opp_state[2],opp_state[3])}"] = [float(stripped[1]),float(stripped[2]),float(stripped[3]),float(stripped[4])]
        
# print(opp_dict["52171"])
r_actions = ["L", "R", "U", "D"]
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
                            print("the input state is",B1_states, B2_states, R_states, ball_pos, state)
                            # actions = list(range(10))
                            new_b1x, new_b1y, new_b2x, new_b2y = B1_x, B1_y, B2_x, B2_y

                            for action in range(10):
                                    
                                if action < 8:
                                    continue
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

                                    r_actprob =  opp_dict[str(state)] 
                                    print(r_actprob)
                                    selected_act = random.choices(r_actions, r_actprob)[0]
                                    print(selected_act)

                                    if selected_act == "U":
                                        new_rx = R_x
                                        new_ry = R_y - 1
                                    elif selected_act == "D":
                                        new_rx = R_x
                                        new_ry = R_y + 1
                                        
                                    elif selected_act == "L":
                                        new_rx = R_x - 1
                                        new_ry = R_y
                                    elif selected_act == "R":
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
                                        print("The new_state parameters is ", new_b1, new_b2, new_r, new_rx, new_ry, ball_pos)
                                        new_state = state_index_i(new_b1,new_b2,new_r,ball_pos)
                                    
                                    if ball_pos == 1:
                                        if action < 4:
                                            success_prob = 1 - 2 * p                                
                                        else:
                                            success_prob = 1 - p
                                        if not cond and (new_b1 == new_r or (new_r == B1_states and new_b1 == R_states)):
                                            success_prob/=2
                                        print("input to transition matrix", state, action, new_state)
                                        transitions[state][action][new_state] = success_prob
                                        transitions[state][action][state_index_i(1,2,3,4,"end")] = 1 - success_prob
                                        rewards[state][action][new_state] = 0
                                        if new_state == state_index_i(1,2,3,4,"end"):
                                            transitions[state][action][new_state] = 1

                                                # pos_move = ["success", "fail"]
                                                # move_1 = random.choices(pos_move, [success_prob_1, 1-success_prob_1])[0]                                                                                 
                                                # move_2 = random.choices(pos_move, [success_prob_2, 1-success_prob_2])[0]
                                    if ball_pos == 2:
                                        if action < 4:
                                            success_prob = 1 - p
                                        else:
                                            success_prob = 1 - 2 * p
                                        if not cond and (new_b2 == new_r or (new_r == B2_states and new_b2 == R_states)):
                                            success_prob/=2
                                        
                                        transitions[state][action][new_state] = success_prob
                                        transitions[state][action][state_index_i(1,2,3,4,"end")] = 1 - success_prob
                                        rewards[state][action][new_state] = 0  

                                        if new_state == state_index_i(1,2,3,4,"end"):
                                            transitions[state][action][new_state] = 1                                            
                                        
                                    
                                if action == 8:
                                    print("Now i will take action 8")
                                    pass_action = 8 
                                    pass_success_prob =  q - 0.1 * max(abs(B1_x - B2_x), abs(B1_y - B2_y))
                                    
                                    if (R_x - B1_x) * (B2_y - B1_y) == (R_y - B1_y) * (B2_x - B1_x):
                                        if (min(B1_x, B2_x) <= R_x <= max(B1_x, B2_x) and min(B1_y, B2_y) <= R_y <= max(B1_y, B2_y)):
                                            pass_success_prob = 0.5 * (q - 0.1 * max(abs(B1_x - B2_x), abs(B1_y - B2_y)))
                                    
                                    #moving R
                                    r_actprob =  opp_dict[str(state)] 
                                    selected_act = random.choices(r_actions, r_actprob)[0]

                                    if selected_act == "U":
                                        new_rx = R_x
                                        new_ry = R_y - 1
                                    elif selected_act == "D":
                                        new_rx = R_x
                                        new_ry = R_y + 1
                                    elif selected_act == "L":
                                        new_rx = R_x - 1
                                        new_ry = R_y
                                    elif selected_act == "R":
                                        new_rx = R_x + 1
                                        new_ry = R_y
                                    
                                        
                                    new_b1 = new_b1x + (new_b1y - 1) * 4
                                    new_b2 = new_b2x + (new_b2y - 1) * 4
                                    new_r = new_rx + (new_ry - 1) * 4
                                    if ball_pos == 1:
                                        new_ball_pos = 2
                                    if ball_pos == 2:
                                        new_ball_pos = 1
                                    if new_b2 == 18 :
                                        print(True)
                                        break
                                    new_state = state_index_i(B1_states,B2_states, new_r, ball_pos)
                                     
                                    transitions[state][pass_action][new_state] = pass_success_prob
                                    transitions[state][pass_action][state_index_i(1,2,3,4,"end")] = 1 - pass_success_prob
                                    rewards[state][pass_action] = 0  
                                
                                if action == 9:
                                    if ball_pos == 1:
                                        prob_shoot = q - 0.2 * (3 - B1_x)
                                    if ball_pos == 2:
                                        prob_shoot = q - 0.2 * (3 - B2_x)
                                    
                                    if R_x == 4 and R_y in [2, 3]:
                                        prob_shoot/=2
                                    
                                    r_actprob =  opp_dict[str(state)] 
                                    selected_act = random.choices(r_actions, r_actprob)[0]

                                    if selected_act == "U":
                                        new_rx = R_x
                                        new_ry = R_y - 1
                                    elif selected_act == "D":
                                        new_rx = R_x
                                        new_ry = R_y + 1
                                    elif selected_act == "L":
                                        new_rx = R_x - 1
                                        new_ry = R_y
                                    elif selected_act == "R":
                                        new_rx = R_x + 1
                                        new_ry = R_y 
                                    transitions[state][action][state_index_i(1,2,3,4,"goal")] = prob_shoot
                                    transitions[state][action][state_index_i(1,2,3,4,"goal")] = 1
                                    transitions[state][action][state_index_i(1,2,3,4,"end")] = 1 - prob_shoot
                                    transitions[state][action][state_index_i(1,2,3,4,"end")] = 0