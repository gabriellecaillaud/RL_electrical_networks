import numpy as np
# gym pour implémenter l'environnement d'apprentissage
from gym import spaces

from stable_baselines3 import PPO, A2C

from alpyne.client.alpyne_client import AlpyneClient
from alpyne.client.model_run import ModelRun
from alpyne.data.spaces import Observation, Action

from alpyne.client.abstract import BaseAlpyneEnv

import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls

# création de l'environnement d'apprentissage avec standards gym à travers l'héritage et implémentation des
# fonctions de BaseAlpyneEnv

class EolienParc(BaseAlpyneEnv):


    def __init__(self, sim: ModelRun):
        super().__init__(sim)
        
    # définie l'espace des observation sous forme d'un vecteur avec les valeurs min et max de chaque champs    
    def _get_observation_space(self) -> spaces.Space:
        low1 = np.zeros(50)
        high1= np.ones(50)
        low2 = np.zeros(3)
        high2 = np.ones(3)
        low3 = np.array([0,0,0])
        high3 = np.array([50,50,50])
        low4 = np.array([-4500])
        high4 = np.array([+4500])
        return spaces.Box(low=np.concatenate((low1,low2,low3,low4)), high=np.concatenate((high1,high2,high3,high4)))
        
    # transformer l'observation récupérer par anylogic sous forme d'un vecteur de type Space de gym
    def _convert_from_observation(self, observation: Observation):
        print("observation.etatequipement")
        print(observation.etatequipement)
        print("observation.disponibleequipe")
        print(observation.disponibleequipe)
        print("observation.emplequipe")
        print(observation.emplequipe)
        print("observation.recompns")
        print(observation.recompns)
        print("FIN")
        return np.array(np.concatenate((observation.etatequipement,observation.disponibleequipe,observation.emplequipe)))

    # définie l'espace d'action sous forme d'un vecteur avec les valeurs min et max de chaque action
    def _get_action_space(self) -> spaces.Space:
        return spaces.Box(low=np.array([0,0,0]), high=np.array([50,50,50]), shape=(3,), dtype=np.int64)

    # transformer l'action prise par l'actor sous forme d'un vecteur à une action de type Action de Alpyne pour
    # la communiquer avec le serveur alpyne
    def _convert_to_action(self, action: np.ndarray) -> Action:
        if(action[0] == action[1]):
            action[1] = 50
        if(action[0] == action[2]):
            action[2] = 50
        if(action[1]== action[2]):
            action[2]= 50
        return Action(req1=int(action[0]), req2=int(action[1]), req3=int(action[2]))
    
    # calcul de la récompense
    def _calc_reward(self, observation: Observation) -> float:
        return observation.recompns

    # définie une état terminale de l'epsode hors un nombre limité de pas
    def _terminal_alternative(self, observation: Observation) -> bool:
        eq = np.array(observation.etatequipement)
        return np.count_nonzero(eq)>=49  
        # arbitrarily chosen small(ish) number


    
    # creation de client Alpyne avec le modele exporté à partir de l'expérience RLExperiment
client = AlpyneClient(r"C:\Users\gabri\Documents\CS\2A\Pole_projet\RL et Anylogic\Export-Test\model.jar", verbose = True, port = 51150)
    # Creation d'un ModelRun avec le template de configuration
cfg = client.configuration_template
    
sim = client.create_reinforcement_learning(cfg)


    # utiliser notre implémentation de l'environnement
env = EolienParc(sim)


low = env.observation_space.low
print('Low', low)
high = env.observation_space.high
print('high', high)

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v
        

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(128,activation='relu')
    self.a = tf.keras.layers.Dense(2,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    a = self.a(x)
    return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

          
    def act(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
  


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
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb[a],op[a])
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

        old_p = tf.reshape(old_p, (len(old_p),2))
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

    
def test_reward(env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(agentoo7.actor(np.array([state])).numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

print("CHECK ligne 201")
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
    actions = np.array(actions, dtype=np.int32)
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
print("check ligne 228")

for s in range(steps):
    if target == True:
          break
    print("step numéro :")
    print(f"s{s}") 
    done = False
    #sim = client.create_reinforcement_learning(cfg)


    # utiliser notre implémentation de l'environnement
    #env = EolienParc(sim)
    state = env.reset()
    print("on a reseté")
    print("STATE", state)
    #On regarde l'état initial
    state = env._get_observation_space
    print("check: reset ok")
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []
    print("new episod")

    for e in range(128):
        print("check ligne 263")
        action = agentoo7.act(state)
        print("\n action \n",action)
        value = agentoo7.critic(np.array([state])).numpy()
        print("\n value \n",value )
        next_state, reward, done, _ = env.step(action)
        dones.append(1-done)
        rewards.append(reward)
        states.append(state)
        print("\n states \n", states)
        #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
        actions.append(action)
        prob = agentoo7.actor(np.array([state]))
        probs.append(prob[0])
        values.append(value[0][0])
        state = next_state
        if done:
            env.reset()
    
    value = agentoo7.critic(np.array([state])).numpy()
    values.append(value[0][0])
    np.reshape(probs, (len(probs),2))
    probs = np.stack(probs, axis=0)

    states, actions,returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)

    for epocs in range(10):
        al,cl = agentoo7.learn(states, actions, adv, probs, returns)
        # print(f"al{al}") 
        # print(f"cl{cl}")   

    avg_reward = np.mean([test_reward(env) for _ in range(5)])
    print(f"total test reward is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            agentoo7.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            agentoo7.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
    if best_reward == 200:
            target = True
    env.close()
    sim = client.create_reinforcement_learning(cfg)


    # utiliser notre implémentation de l'environnement
    env = EolienParc(sim)


env.close()
        
    