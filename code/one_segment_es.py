import numpy as np
from forward_kinematics_one_segment import ForwardKinematicsOneSegment
import logging
import pickle
import sys


env = ForwardKinematicsOneSegment(0.075, 0.125, 0.01, 10)

version = 3
model_path = str("./models/model_one_segment_es%d.p" % version)
log_path = str("./models/model_one_segment_es%d.log" % version)
#logs_path = "/home/andi/Documents/masterarbeit_imes/code/model"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
log_hdlr = logging.FileHandler(log_path, mode="w")
log_hdlr.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_hdlr)

state_dim = env.state_dim
action_dim = env.action_dim

save_model_step = 100
# hyper parameters
h1_size = 512
h2_size = 512
npop = 50
sigma = 0.1
alpha = 0.03

# Handling the script
restore = False
demo = False
save_model = True

if restore:
   model = pickle.load(open(model_path, "rb"))
   print("model restored")
else:
   model = {}
   if h2_size == None:
      model["W1"] = np.random.randn(state_dim, h1_size) / np.sqrt(state_dim)
      model["W2"] = np.random.randn(h1_size, action_dim) / np.sqrt(h1_size)
   else:
      model["W1"] = np.random.randn(state_dim, h1_size) / np.sqrt(state_dim)
      model["W2"] = np.random.randn(h1_size, h2_size) / np.sqrt(h1_size)
      model["W3"] = np.random.randn(h2_size, action_dim) / np.sqrt(h2_size)

# log hyper params
logger.debug("HYPERPARAMS\n"
             "version: {}\th1_size: {}\th2_size: {}\tnpop: {}\tsigma: {}\t alpha: {}\n"
             "ENVIRONMENT\n"
             "delta_l: {}\tmax_steps: {}\tgoal_eps: {}\n\n"
             .format(version, h1_size, h2_size, npop, sigma, alpha, env.delta_l, env.max_steps, env.eps))
logger.debug("ep\treward\tsteps\tstart_d\tend_d\tcovered\tinfo")

def get_action(state, model):
   h1 = np.tanh( np.matmul(state, model["W1"]) )
   if h2_size == None:
      action = np.matmul(h1, model["W2"])
      action = np.tanh(action) * env.delta_l
   else:
      h2 = np.tanh( np.matmul(h1, model["W2"]))
      action = np.matmul(h2, model["W3"])
      action = np.tanh(action) * env.delta_l
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
      total_reward += reward

      if done:
         goal_reached = True if "GOAL" in info else False
         if jitter_run == False:
            logger.debug("{}\t{:.1f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}"
                         .format(episode, total_reward, steps,
                                 env.dist_start, env.dist_end, env.dist_start-env.dist_end, goal_reached))
            print(env.info, end="      ")
            covered_distances.append(env.dist_start-env.dist_end)
            episode_rewards.append(total_reward)
            goals_reached.append(1) if goal_reached else goals_reached.append(0)
         break
   return total_reward

episode_rewards = []
covered_distances = []
goals_reached = []

# demonstrate loaded or random model
if demo:
   for episode in range(5):
      print(f(model, render=True, append=True))
   sys.exit("demo finished")

total_episodes = 10000
average_reward = 0
for episode in range(total_episodes):
   N = {}
   for key, value in model.items():
      N[key] = np.random.randn(npop, value.shape[0], value.shape[1])
   R = np.zeros(npop)

   for traj in range(npop):
      model_jitter = {}
      for key, value in model.items():
         model_jitter[key] = value + sigma*N[key][traj]
      R[traj] = f(model_jitter, jitter_run=True)

   A = (R - np.mean(R)) / np.std(R)

   for key in model:
      model[key] = model[key] + alpha/(npop*sigma) * np.dot(N[key].transpose(1, 2, 0), A)

   cur_reward = f(model)

   average_reward = 1/(episode+1) *(average_reward*episode + cur_reward)
   print("iter{:4d}, episode reward: {:4.1f}, average reward: {:4.1f}".format(episode, cur_reward, average_reward))

   if episode % save_model_step == 0 and save_model:
      pickle.dump(model, open(model_path, "wb"))

logger.debug("\n\nmeancovered distance: {}\t"
             "goals reached: {}/{}".format(sum(covered_distances)/len(covered_distances), sum(goals_reached), len(goals_reached)))
