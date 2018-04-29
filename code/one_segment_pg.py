import numpy as np
import tensorflow as tf
from forward_kinematics_one_segment import ForwardKinematicsOneSegment
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
log_hdlr = logging.FileHandler("eplog.log", mode="w")
log_hdlr.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(log_hdlr)

env = ForwardKinematicsOneSegment(0.075, 0.125, 0.01, 10)
logs_path = "/home/andi/Documents/masterarbeit_imes/code/model"

n_states = env.n_states
n_actions = env.n_actions
n_action_values = len(env.n_action_values)
n_outputs = n_actions*n_action_values
n_neurons_hl1 = 32
n_neurons_hl2 = 32
discount = env.gamma
learning_rate = 1e-2
info_step = 50
global_ep_step = 500

tf.reset_default_graph()

def discount_and_normalize_rewards(r, gamma=0.99):
   discounted_norm_rewards = np.zeros_like(r)
   running_return = 0
   # Gt = Rt+1 + gamma * Rt+2 + gamma^2 * Rt+3 + ... + gamma^(T-t+1)RT
   for t in reversed(range(len(r))):
      running_return = r[t] + gamma * running_return
      discounted_norm_rewards[t] = running_return

   # subtract mean and divide by standard deviation / normalize rewards
   discounted_norm_rewards -= np.mean(discounted_norm_rewards)
   discounted_norm_rewards /= np.std(discounted_norm_rewards)

   return discounted_norm_rewards

with tf.name_scope("Inputs"):
   tf_states = tf.placeholder(tf.float32, shape=[None, n_states], name="tf_states")
   tf_action1 = tf.placeholder(tf.float32, shape=[None, n_action_values], name="tf_action1")
   tf_action2 = tf.placeholder(tf.float32, shape=[None, n_action_values], name="tf_action2")
   tf_action3 = tf.placeholder(tf.float32, shape=[None, n_action_values], name="tf_action3")
   tf_rewards_discounted = tf.placeholder(tf.float32, shape=[None, ], name="discounted_ep_rewards")
#   tf_rewards = tf.placeholder(tf.float32, shape=[None, ], name="undiscounted_ep_rewards")

with tf.name_scope("Parameters"):
   # see http://cs231n.github.io/neural-networks-2/ for initialization
   W1 = tf.get_variable(name="W1", shape=[n_states, n_neurons_hl1], initializer=tf.keras.initializers.he_normal())
   b1 = tf.get_variable(name="b1", shape=[1, n_neurons_hl1], initializer=tf.zeros_initializer())
   W2 = tf.get_variable(name="W2", shape=[n_neurons_hl1, n_neurons_hl2], initializer=tf.keras.initializers.he_normal())
   b2 = tf.get_variable(name="b2", shape=[1, n_neurons_hl2], initializer=tf.zeros_initializer())
   W3 = tf.get_variable(name="W3", shape=[n_neurons_hl2, n_actions*n_action_values], initializer=tf.keras.initializers.he_normal())
   b3 = tf.get_variable(name="b3", shape=[1, n_actions*n_action_values], initializer=tf.zeros_initializer())

with tf.name_scope("layer1"):
   layer1 = tf.nn.relu(tf.add(tf.matmul(tf_states, W1), b1))
with tf.name_scope("layer2"):
   layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
with tf.name_scope("Output_layer"):
   logits = tf.add(tf.matmul(layer2, W3), b3, name="logits")
   tf_probs1 = tf.nn.softmax(logits[0][0:3], name="probs1")
   tf_probs2 = tf.nn.softmax(logits[0][3:6], name="probs2")
   tf_probs3 = tf.nn.softmax(logits[0][6:9], name="probs3")

with tf.name_scope("Loss"):
   neg_log_prob1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, 0:3], labels=tf_action1)
   neg_log_prob2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, 3:6], labels=tf_action2)
   neg_log_prob3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, 6:9], labels=tf_action3)
   loss = tf.reduce_mean((neg_log_prob1+neg_log_prob2+neg_log_prob3) * tf_rewards_discounted)

with tf.name_scope("Train"):
   train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

restore = False
with tf.Session() as sess:
   saver = tf.train.Saver()
   if restore == True:
      saver.restore(sess, path=logs_path+"/model.ckpt")
   else:
      sess.run(tf.global_variables_initializer())

   logger.debug("HYPERPARAMS\nn_neurons_hl1 = {}\nn_neurons_hl2 = {}\ndiscount = {}\nlearning_rate = {}"
                .format(n_neurons_hl1, n_neurons_hl2, discount, learning_rate))
   logger.debug("#" * 50)
   logger.debug("EP\treward\tsteps\tinfo")
   state = env.reset(l1=0.1, l2=0.1, l3=0.1)
   render = False

   episode_states = []
   episode_actions1 = []
   episode_actions2 = []
   episode_actions3 = []
   episode_rewards = []

   reward_sum = 0
   episode = 0
   max_episodes = 10000
   step = 0
   while episode < max_episodes:
      step += 1
      if render:
         env.render()
      # choose actions stochastically
      input_state = np.reshape(state, (1, n_states)) # reshape state for tf graph
      feed_dict = {tf_states: input_state}
      action_probs1, action_probs2, action_probs3 = sess.run([tf_probs1, tf_probs2, tf_probs3],
                                                             feed_dict=feed_dict)
#      print(tf_probs1.eval(feed_dict), tf_probs2.eval(feed_dict), tf_probs3.eval(feed_dict))

      # action values: -1, 0, 1 drawn stochastically
      action1 = np.random.choice(range(len(action_probs1.ravel())), p=action_probs1.ravel()) - 1
      action2 = np.random.choice(range(len(action_probs2.ravel())), p=action_probs2.ravel()) - 1
      action3 = np.random.choice(range(len(action_probs3.ravel())), p=action_probs3.ravel()) - 1
      # action values: -1, 0, 1 * delta_l -> retract, hold, release tendon
      state_, reward, done, info = env.step(action1*env.delta_l, action2*env.delta_l, action3*env.delta_l)

      episode_states.append(state)
      action_onehot1 = np.identity(n_action_values)[action1+1]
      action_onehot2 = np.identity(n_action_values)[action2+1]
      action_onehot3 = np.identity(n_action_values)[action3+1]
      episode_actions1.append(action_onehot1)
      episode_actions2.append(action_onehot2)
      episode_actions3.append(action_onehot3)
      episode_rewards.append(reward)
      reward_sum += reward

      state = state_

      if done:
         episode += 1

         logger.debug("{}\t{}\t{}\t{}".format(episode, sum(episode_rewards), len(episode_rewards), info))

         disc_norm_ep_rewards = discount_and_normalize_rewards(episode_rewards, env.gamma)
         # train
         feed_dict = {tf_states: episode_states, tf_action1: episode_actions1,
                      tf_action2: episode_actions2, tf_action3: episode_actions3,
                      tf_rewards_discounted: disc_norm_ep_rewards}
#         logits = sess.run(logits, feed_dict={tf_states: episode_states})
#         print(logits)
         sess.run(train_op, feed_dict=feed_dict)

         if episode % info_step == 0:
            print("AVERAGE REWARD EP {}-{}: {}".format(episode-(info_step-1), episode, reward_sum/info_step))

            reward_sum = 0

         if episode % global_ep_step == 0:
            saver.save(sess, save_path=logs_path+"/model.ckpt")

         # reset lists
         episode_states = [];
         episode_actions1 = []; episode_actions2 = []; episode_actions3 = []
         episode_rewards = [];

         state = env.reset(l1=0.1, l2=0.1, l3=0.1)


