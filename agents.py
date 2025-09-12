import pandas as pd
import numpy as np
import random
import state_action_reward as sar


class Agent(object):
    def __init__(self, agent_info:dict):
        """Initializes the agent to get parameters and create an empty q-tables."""

        self.epsilon     = agent_info["epsilon"]
        self.step_size   = agent_info["step_size"]
        self.states      = sar.states()
        self.actions     = sar.actions()
        self.R           = sar.rewards(self.states, self.actions)        

        self.q = pd.DataFrame(
            data    = np.zeros((len(self.states), len(self.actions))), 
            columns = self.actions, 
            index   = self.states
        )
        
        self.visit = self.q.copy()

    
class QLearningAgent(Agent):
    
    def __init__(self, agent_info:dict):        
        
        super().__init__(agent_info)
        self.prev_state  = 0
        self.prev_action = 0
    
    def step(self, state_dict, actions_dict):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        """
        
        # (1) Transform state dictionary into tuple
        state = [i for i in state_dict.values()]
        state = tuple(state)
        
        # (2) Choose action using epsilon greedy
        # (2a) Random action
        if random.random() < self.epsilon:
            
            # action list: ["play_wasabi", "play_highest_nigiri", "play_tempura", 
            #               "play_sashimi", "play_dumpling", "play_pudding", 
            #               "play_highest_maki", "play_chopsticks"]
            
            # Determine which actions are possible based on cards in hand
            actions_possible = [0]*len(actions_dict)
            action = actions_possible

            if wasabi_in_hand != 0:
                actions_possible[0] = 1
            if ((egg_nigiri_in_hand != 0) or (salmon_nigiri_in_hand != 0) or (squid_nigiri_in_hand != 0)):
                actions_possible[1] = 1
            if tempura_in_hand != 0:
                actions_possible[2] = 1
            if sashimi_in_hand != 0:
                actions_possible[3] = 1
            if dumpling_in_hand != 0:
                actions_possible[4] = 1
            if pudding_in_hand != 0:
                actions_possible[5] = 1
            if ((maki_1_in_hand != 0) or (maki_2_in_hand != 0) or (maki_3_in_hand != 0)):
                actions_possible[6] = 1
            if chopsticks_in_hand != 0:
                actions_possible[7] = 1

            # actions_possible = [key for key,val in actions_dict.items() if val != 0]
            actions_possible_idx = [i for i in range(len(actions_possible)) if actions_possible[i]]         
            action[random.choice(actions_possible_idx)] = 1
        
        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0
            
            for i in actions_possible:
                val = self.q.loc[[state],i][0]
                if val >= val_max: 
                    val_max = val
                    action = i
        
        return action
    
    def update(self, state_dict, action):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str
        """
        state = [i for i in state_dict.values()]
        state = tuple(state)
        
        # (1) Set prev_state unless first turn
        if self.prev_state != 0:
            prev_q = self.q.loc[[self.prev_state], self.prev_action][0]
            this_q = self.q.loc[[state], action][0]
            reward = self.R.loc[[state], action][0]
            
            print ("\n")
            print (f'prev_q: {prev_q}')
            print (f'this_q: {this_q}')
            print (f'prev_state: {self.prev_state}')
            print (f'this_state: {state}')
            print (f'prev_action: {self.prev_action}')
            print (f'this_action: {action}')
            print (f'reward: {reward}')
            
            # Calculate new Q-values
            if reward == 0:
                self.q.loc[[self.prev_state], self.prev_action] = prev_q + self.step_size * (reward + this_q - prev_q) 
            else:
                self.q.loc[[self.prev_state], self.prev_action] = prev_q + self.step_size * (reward - prev_q)
                
            self.visit.loc[[self.prev_state], self.prev_action] += 1
            
        # (2) Save and return action/state
        self.prev_state  = state
        self.prev_action = action
      
        
class MonteCarloAgent(Agent):

    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()
    
    def step(self, state_dict, actions_dict):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        """
        
        # (1) Transform state dictionary into tuple
        state = [i for i in state_dict.values()]
        state = tuple(state)
        
        # (2) Choose action using epsilon greedy
        # (2a) Random action
        if random.random() < self.epsilon:
            
            actions_possible = [key for key,val in actions_dict.items() if val != 0]         
            action = random.choice(actions_possible)
        
        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0
            
            for i in actions_possible:
                val = self.q.loc[[state],i][0]
                if val >= val_max: 
                    val_max = val
                    action = i
        
        # (3) Add state-action pair if not seen in this simulation
        if ((state),action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
        
        self.q_seen.append(((state),action))
        self.visit.loc[[state], action] += 1
        
        return action
    
    def update(self, state_dict, action):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str
        """
        
        state  = [i for i in state_dict.values()]
        state  = tuple(state)
        reward = self.R.loc[[state], action][0]
        
        # Update Q-values of all state-action pairs visited in the simulation
        for s,a in zip(self.state_seen, self.action_seen): 
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
            print (self.q.loc[[s],a])
        
        self.state_seen, self.action_seen, self.q_seen = list(), list(), list()