import gym
import numpy as np
env = gym.make("FrozenLake-v1", render_mode="human",map_name="4x4",is_slippery=False)

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 300

map="SFFFFHFHFFFHHFFG"
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
q_table=np.random.uniform(low=0,high=1,size=((16,)+(4,)))
for episode in range(EPISODES):
    state=(env.reset())[0]
    done =False
    truncated=False
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False
    while not done and not truncated:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
    
        new_state,reward,done,truncated,_=env.step(action)
        
        if episode % SHOW_EVERY == 0:
            env.render()
        
        if not done and not truncated:
        
         # Maximum possible Q value in next step (for new state)
         max_future_q = np.max(q_table[new_state])


            # Current Q value (for current state and performed action)
         current_q = q_table[(state,) + (action,)]

            # And here's our equation for a new Q value for current state and action
         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
         q_table[(state,) + (action,)] = new_q
        
        elif(reward==1.0):
         q_table[(state,) + (action,)] =1
         print(f"Goal reached{episode}")
        state = new_state
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:

            epsilon -= epsilon_decay_value
        
    
env.close()





'''




'''
