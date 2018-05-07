import numpy as np
from forward_kinematics_one_segment import ForwardKinematicsOneSegment
import logging
import pickle
import sys
import os

version = 4
file_path = str("./models/model_one_segment_es%d.p" % version)
log_path = str("./logs/model_one_segment_es%d.log" % version)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
log_hdlr = logging.FileHandler(log_path, mode="w")
log_hdlr.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_hdlr)

def ensure_dir(file_path):
   directory = os.path.dirname(file_path)
   if not os.path.exists(directory):
      os.makedirs(directory)

env = ForwardKinematicsOneSegment(0.075, 0.125, 0.01, 10)
logs_path = "/home/andi/Documents/masterarbeit_imes/code/model"

state_dim = env.state_dim
action_dim = env.action_dim
info_step = 50
save_model_step = 10
# hyper parameters
h1_size = 64
h2_size = 64
npop = 50
sigma = 0.1
alpha = 0.03

# Handling the script
restore = True
demo = True
save_model = True
render = False

if restore:
   model = pickle.load(open(file_path, "rb"))
   print("Model restored.")
else:
   model = {}
   model["W1"] = np.random.randn(state_dim, h1_size) / np.sqrt(state_dim)
   model["b1"] = np.zeros(h1_size)
   model["W2"] = np.random.randn(h1_size, h2_size) / np.sqrt(h1_size)
   model["b2"] = np.zeros(h2_size)
   model["W3"] = np.random.randn(h2_size, action_dim) / np.sqrt(h2_size)
   model["b3"] = np.zeros(action_dim)

def get_action(state, model):
   h1 = np.tanh( np.add( np.matmul(state, model["W1"]), model["b1"]) )
   h2 = np.tanh( np.add( np.matmul(h1, model["W2"]), model["b2"]) )
   action = np.tanh( np.add( np.matmul(h2, model["W3"]), model["b3"]) ) * env.delta_l
   return action


def f(model, render=False, jitter_run=False):
   """function that plays one episode and returns total reward"""
   state = env.reset(0.1, 0.1, 0.1)
   total_reward = 0
   steps = 0
   while True:
      if render: env.render()

      action = get_action(state, model)
      state, reward, done, info = env.step(action[0], action[1], action[2])
      steps += 1
      episode_rewards.append(reward)
      total_reward += reward

      if done:
         logger.debug("{}\t{:.1f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}"
                      .format(episode, total_reward, steps,
                              env.dist_start, env.dist_end, env.dist_start-env.dist_end, info))
         if jitter_run == False:
            print(env.info, end="      ")
            covered_distances.append(env.dist_start-env.dist_end)
         break
   return total_reward

episode_rewards = []
covered_distances = []
if demo:
   for episode in range(10):
      print(f(model, render=True))
   sys.exit("demo finished")

total_episodes = 1000
average_reward = 0
for episode in range(total_episodes):
   N = {}
   for key, value in model.items():
      if "W" in key:
         N[key] = np.random.randn(npop, value.shape[0], value.shape[1])
      elif "b" in key:
         N[key] = np.random.randn(npop, value.shape[0])
   R = np.zeros(npop)

   for traj in range(npop):
      model_jitter = {}
      for key, value in model.items():
         model_jitter[key] = value + sigma*N[key][traj]
      R[traj] = f(model_jitter, jitter_run=True)

   A = (R - np.mean(R)) / np.std(R)

   for key in model:
      if "W" in key:
         model[key] = model[key] + alpha/(npop*sigma) * np.dot(N[key].transpose(1, 2, 0), A)
      elif "b" in key:
         model[key] = model[key] + alpha/(npop*sigma) * np.dot(N[key].transpose(), A)

   cur_reward = f(model, render=render)

   average_reward = 1/(episode+1) *(average_reward*episode + cur_reward)
   print("iter{:4d}, episode reward: {:4.1f}, average reward: {:4.1f}".format(episode, cur_reward, average_reward))

   if episode % save_model_step == 0 and save_model:
      pickle.dump(model, open(file_path, "wb"))