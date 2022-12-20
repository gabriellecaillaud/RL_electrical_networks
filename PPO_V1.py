import numpy as np
import tensorflow as tf 
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as keras
import keras.losses as kls
from Grid_restoration_AI_Project_v1 import environement
#from grid_environment import reset,


env = environement()
#low = env.observation_space.low
#high = env.observation_space.high

class critic(tf.keras.Model):   #The Critic network outputs the value of a state.
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(324,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v
    

class actor(tf.keras.Model):  #The Actor-network takes the current state as input and outputs probability for each action.
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(324,activation='relu')
    self.a = tf.keras.layers.Dense(21,activation='softmax')    #21=number of distinct switches on which we can take an action (open or close)

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    return a

class agent():                               #Action Selection
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)    #We define our agent class and initialize optimizer and learning rate.
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2         #We also define a clip_pram variable which will be used in the actor loss function.

          
    def act(self,state):
        #print("le np array", np.array([state]))
        prob = self.actor(np.asarray([state]).astype('float32'))
        prob = prob.numpy()
        
        reconf_list = [(16, 29), (29, 16), (8, 9), (9, 8), (22, 35), (35, 22), (20, 22), (22, 20), (14, 33), (33, 14), (24, 10), (10, 24), (33, 31), (31, 33), (11, 9), (9, 11), (13, 12), (12, 13), (25, 11), (11, 25), (18, 5), (5, 18), (24, 23), (23, 24), (20, 19), (19, 20), (21, 36), (36, 21), (20, 21), (21, 20), (31, 30), (30, 31), (32, 13), (13, 32), (27, 26), (26, 27), (5, 4), (4, 5), (23, 19), (19, 23), (7, 27), (27, 7)]
        reconf = {}
        treshold = 0.4

        for i in range(len(prob[0])):
            reconf[reconf_list[2*i]] = 1 if prob[0][i]>treshold else 0
            reconf[reconf_list[2*i+1]] = 1 if prob[0][i]>treshold else 0
        
        return reconf
        #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)    #For action selection, we will be using the TensorFlow probabilities library, which takes probabilities as input and convert them into distribution.
        #action = dist.sample()                 #Then, we use the distribution for action selection.
        #return int(action.numpy()[0])
  


    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op,a  in zip(probability, adv, old_probs, actions):
                        t =  tf.constant(t)
                        #op =  tf.constant(op)
                        #print(f"t{t}")
                        ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        #ratio = tf.math.divide(pb[a],op[a])
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),21))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

def test_reward(env):            #This function will be used to test our agent’s knowledge and returns the total reward for one episode.
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        #print(np.array([state]))                                                        #"DONE" A MODIFIER 
        #print(len(np.array(state)))
        #print(agentoo7.actor(np.array([state])))
        action = agentoo7.act(state)
        next_state, reward, done, _ = env.step(action,0)                      #env.step() : This command will take an action at each step. The action is specified as its parameter. Env.step function returns four parameters, namely observation, reward, done and info. These four are explained below:
                                                                            #observation : an environment-specific object representing your observation of the environment.
                                                                            #reward : amount of reward achieved by the previous action. It is a floating data type value. The scale varies between environments.
                                                                            #done : A boolean value stating whether it’s time to reset the environment again.
                                                                            #info (dict): diagnostic information useful for debugging.
        state = next_state
        total_reward += reward
        

    return total_reward

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    #actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv    


tf.random.set_seed(336699)
agentoo7 = agent()
steps = 5000
ep_reward = []
total_avgr = []
target = False 
best_reward = 0
avg_rewards_list = []


for s in range(steps):           #We will loop for “steps” time i.e we will collect experience for “steps” time.
  if target == True:
          break
  
  done = False
  state = env.reset()
  all_aloss = []
  all_closs = []
  rewards = []
  states = []
  actions = []
  probs = []
  dones = []
  values = []
  print("new episod")

  for e in range(128):         #The next loop is for the number of times agent interacts with environments and we store experiences in different lists.
    action = agentoo7.act(state)
    value = agentoo7.critic(np.array([state])).numpy()
    next_state, reward, done, _ = env.step(action,e)
    dones.append(1-done)
    rewards.append(reward)
    states.append(state)
    #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
    actions.append(action)
    prob = agentoo7.actor(np.array([state]))
    probs.append(prob[0])
    values.append(value[0][0])
    state = next_state
    if done:
      state = env.reset()
  
  state = env.reset()
  value = agentoo7.critic(np.array([state])).numpy()                  #After the above loop, we calculate and add the value of the state next 
  values.append(value[0][0])                                          #to the last state for calculations in the Generalized Advantage Estimation method.
  np.reshape(probs, (len(probs),21))
  probs = np.stack(probs, axis=0)
  
  states, actions,returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)   #Then, we process all the lists in the Generalized Advantage Estimation method to get returns, advantage.
  for epocs in range(10):                                    #We train our networks for 10 epochs.
      al,cl = agentoo7.learn(states, actions, adv, probs, returns)
      # print(f"al{al}") 
      # print(f"cl{cl}")   

  avg_reward = np.mean([test_reward(env) for _ in range(5)])   #After training, we will test our agent on the test environment for five episodes.
  print(f"total test reward is {avg_reward}")
  avg_rewards_list.append(avg_reward)
  if avg_reward > best_reward:                               #If the average reward of test episodes is larger than the target reward set by you then stop otherwise repeat from step one.
        print('best reward=' + str(avg_reward))
        agentoo7.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
        agentoo7.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
        best_reward = avg_reward
  if best_reward == 200:
        target = True
  env.reset()

env.close()
    
  