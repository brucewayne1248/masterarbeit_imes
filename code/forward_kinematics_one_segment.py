import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin
# libraries needed to render continuum robot
#import matplotlib
import matplotlib.pyplot as plt
import time
from plot_utils import Arrow3D, mypause
# video file
# import cv2 by deleting kinetic python path from sys (else error)
#import sys
#try:
#   sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
#except :
#   print("kinetic path already deleted")
#import cv2

class ForwardKinematicsOneSegment():
   """
   class handling the forward kinematics of a single segment tendon driven continuum robot

   lmin, lmax: min and max tendon length [m]

   d:          pitch distance to cable guides [m]

   n:          number of units (spacer discs) within one segment
   """
   precision_digits = 16 # rounding precision needed for handling the singularity at l1=l2=l3

   def __init__(self, lmin, lmax, d, n):
      self.lmin = lmin
      self.lmax = lmax
      self.d = d
      self.n = n

      self.l1 = None; self.l2 = None; self.l3 = None; # tendon lengths
      self.lengths = None # [l1, l2, l3]
      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.kappa = None # curvature kappa [m^(-1)]
      self.phi = None # angle rotating arc out of x-z plane [rad]
      self.seg_len = None # total arc length [m]

      self.T01 = None # transformation matrix either Bishop or Frenet frames
      self.normal_vec = None # Frenet: pointing towards center point of arc radius # Bishop: aligned with the base frame
      self.binormal_vec = None # tangent_vec x normal_vec
      self.tangent_vec = None # tangent vector of arc
      self.tip_vec = None # robot's tip vector [m] [x, y, z]

      self.fig = None # fig variable used for plotting

      # variables needed in episodic reinforcement learning
      self.state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.reward = None # current reward
      self.done = None # indicates that episode is progress or over
      self.info = None # additional info returned by stepping the environment, indicating goal reached
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.tangent_vec_goal = None # tangent vector of goal position
      self.state_dim = 10
      self.action_dim = 3 # number of actions per time step
      self.delta_l = 0.0001
      self.max_steps = 500 # max steps per episode
      self.eps = 3e-3 # distance tolerance to reach goal
      self.dist_start = None # start distance to goal of current episode
      self.dist_end = None # end distance to goal of current episode
      self.dist = None # current distance to goal

   def reset(self, l1=None, l2=None, l3=None, l1goal=None, l2goal=None, l3goal=None, reset_goal=True):
      """ Resets the environment and updates other variables accordingly. Returns state of new episode. """
      self.l1 = np.random.uniform(self.lmin, self.lmax) if l1 == None else l1
      self.l2 = np.random.uniform(self.lmin, self.lmax) if l2 == None else l2
      self.l3 = np.random.uniform(self.lmin, self.lmax) if l3 == None else l3
      # after resetting tendon lengths, workspace needs to be updated
      self.update_workspace()
      # create goal with a little distance away from tip-vetor
      if (l1goal is not None) and (l2goal is not None) and (l3goal is not None):
         self.set_goal(l1goal, l2goal, l3goal)
      if reset_goal:
         self.goal = self.tip_vec
         while norm(self.goal-self.tip_vec) < 2*self.eps:
            self.set_goal(l1goal, l2goal, l3goal) # set a new goal for the episode
      self.state = self.get_state()
      self.reward = 0
      self.info = "Environment is reset."
      self.done = False
      self.dist_start = norm(self.goal-self.tip_vec)
      self.steps = 0
      return self.state

   def set_goal(self, l1goal=None, l2goal=None, l3goal=None):
      """ Sets the goal to a random point of the robot's workspace [x, y, z] in [m]
      and sets the tangent vector accordingly."""
      l1goal = np.random.uniform(self.lmin, self.lmax) if l1goal == None else l1goal
      l2goal = np.random.uniform(self.lmin, self.lmax) if l2goal == None else l2goal
      l3goal = np.random.uniform(self.lmin, self.lmax) if l3goal == None else l3goal
      kappa, phi, seg_len = self.configuration_space(l1goal, l2goal, l3goal)
      T01 = self.transformation_matrix(kappa, phi, seg_len)
      self.goal = np.matmul(T01, self.base)[0:3]
      self.tangent_vec_goal =  T01[0:3, 2]

   def step(self, delta_l1, delta_l2, delta_l3):
      """Steps the environment and returns new state, reward, done, info."""
      if self.done:
         self.info = "Reset environment with env.reset(), episode is done."
         print(self.info)
         return self.state, self.reward, self.done, self.info
      self.steps += 1

      self.l1 += delta_l1; self.l2 += delta_l2; self.l3 += delta_l3
      # make sure tendon lengths are within min, max
      lengths = [self.l1, self.l2, self.l3]
      for i in range(len(lengths)):
         if lengths[i] < self.lmin:
            lengths[i] = self.lmin
         elif lengths[i] > self.lmax:
            lengths[i] = self.lmax
      self.l1 = lengths[0]; self.l2 = lengths[1]; self.l3 = lengths[2]

      old_dist = self.goal-self.tip_vec; self.old_dist = old_dist
      old_dist_euclid = norm(old_dist); self.old_dist_euclid = old_dist_euclid
      self.update_workspace()
      new_dist = self.goal-self.tip_vec; self.new_dist = new_dist
      new_dist_euclid = norm(new_dist); self.new_dist_euclid = new_dist_euclid

      # handling regular step
      self.state = self.get_state()
      self.reward = -100*(new_dist_euclid-old_dist_euclid)/self.dist_start
      self.info = "EPISODE RUNNING @STEP {} DISTANCE: {:5.2f}mm".format(self.steps, 1000*new_dist_euclid)
      # handling goal reaching case
      if norm(self.tip_vec-self.goal) < self.eps:
         self.done = True
         self.reward = 0.5*float(self.max_steps)
         self.dist_end = norm(self.goal-self.tip_vec)
         self.info = "GOAL!!! DISPLACEMENT {:.2f}mm COVERED {:5.2f} @step {}".format(1000*norm(self.goal-self.tip_vec), 1000*(self.dist_start-self.dist_end), self.steps)
         return self.state, self.reward, self.done, self.info
      # handling case when max steps are exceeded
      if self.steps >= self.max_steps:
         self.dist_end = norm(self.goal-self.tip_vec)
         self.info = "MAX STEPS {} REACHED, DISTANCE {:5.2f}mm COVERED {:5.2f}mm.".format(self.max_steps, 1000*self.dist_end, 1000*(self.dist_start-self.dist_end))
         self.done = True

      return self.state, self.reward, self.done, self.info

   def get_state(self):
      return np.array([self.l1, self.l2, self.l3, self.tip_vec[0], self.tip_vec[1], self.tip_vec[2],
                       self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec)])

   def update_workspace(self):
      """ updates all necessary variables after changing tendon lengths """
      self.lengths = np.array([self.l1, self.l2, self.l3])
      self.kappa, self.phi, self.seg_len = self.configuration_space(self.l1, self.l2, self.l3)
      self.T01 = self.transformation_matrix(self.kappa, self.phi, self.seg_len, frame="bishop")
      self.normal_vec = self.T01[0:3, 0]
      self.binormal_vec = self.T01[0:3, 1]
      self.tangent_vec = self.T01[0:3, 2]
      self.tip_vec = np.matmul(self.T01, self.base)[0:3]

   def configuration_space(self, l1, l2, l3):
      # useful expressions to shorten formulas below
      lsum = l1+l2+l3
      expr = l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3
      # in rare cases expr ~ +-1e-17 when l1~l2~l3 due to floating point operations
      # in these cases expr has to be set to 0.0 in order to handle the singularity
      if round(abs(expr), self.precision_digits) == 0:
         expr = 0.0
      kappa = 2*sqrt(expr) / (self.d*lsum)
      phi = atan2(sqrt(3)*(l2+l3-2*l1), 3*(l2-l3))
      # calculate total segment length
      if l1 == l2 == l3 or expr == 0.0: # handling the singularity
         seg_len = lsum / 3
      else:
         seg_len = self.n*self.d*lsum / sqrt(expr) * asin(sqrt(expr)/(3*self.n*self.d))
      return kappa, phi, seg_len

   def transformation_matrix(self, kappa, phi, s, frame="bishop"):
      if round(kappa, self.precision_digits) == 0.0: #handling singularity
         T = np.identity(4)
         T[2, 3] = s
      else:
         if frame == "bishop":
            T = np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])
         elif frame == "frenet":
            T = np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-sin(kappa*s), 0, cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])
         else:
            raise NotImplementedError('Use frame="bishop" or frame="frenet"')
      return T

   def points_on_arc(self, kappa, num_points):
      """ returns np.array([num_points, 3]) of arc points [x(s), y(s), z(s)] for plot [m] """
      points = np.zeros((num_points, 3))
      s = np.linspace(0, self.seg_len, num_points)
      for i in range(num_points):
         points[i] = np.matmul(self.transformation_matrix(self.kappa, self.phi, s[i]),
                                np.array([0.0, 0.0, 0.0, 1]))[0:3]
      return points

   def arc_params_to_tendon_lenghts(self, kappa, phi, s):
      """ converts configuration space [kappa, phi, s] to actuator space [l1, l2, l3] of the robot's segment """
      if round(kappa, self.precision_digits) == 0:
         l1 = s; l2 = s; l3 = s
      else:
         l1 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*sin(phi))
         l2 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa+self.d*sin(np.pi/3+phi))
         l3 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*cos(np.pi/6+phi))
      return l1, l2, l3

   def render(self, pause=0.0000001, frame="bishop", save_frames=False):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      if self.fig == None:
         self.init_render()

      points = self.points_on_arc(self.kappa, 100) # points to be plotted from base to robot's tip

      while self.ax.lines:
         self.ax.lines.pop() # delete plots of previous frame
      self.ax.plot(points[:,0], points[:,1], points[:,2], label="segment 1", c="black", linewidth=4)
      self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="goal", c="magenta", marker="*", markersize=15)
      self.ax.legend()

      # delete arrows of previous frame, except base frame
      while len(self.ax.artists) > 3:
         self.ax.artists.pop()
      # add current frenet or bishop coordinate frame in plot
      normal_vec = self.normal_vec
      binormal_vec = self.binormal_vec
      tangent_vec = self.tangent_vec
      if frame == "frenet":
         T = self.transformation_matrix(self.kappa, self.phi, self.seg_len, frame="frenet")
         normal_vec = T[0:3, 0]
         binormal_vec = T[0:3, 1]
         tangent_vec = T[0:3, 2]

      anormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*normal_vec[0]],
                        [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*normal_vec[1]],
                        [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*normal_vec[2]],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      abinormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*binormal_vec[0]],
                          [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*binormal_vec[1]],
                          [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*binormal_vec[2]],
                          arrowstyle="-|>", lw=1.5, mutation_scale=10, color="g")
      atangent = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*tangent_vec[0]],
                         [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*tangent_vec[1]],
                         [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*tangent_vec[2]],
                         arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      self.ax.add_artist(anormal)
      self.ax.add_artist(abinormal)
      self.ax.add_artist(atangent)
      # tangent vector indicating orientation of goal point
      atangent_goal = Arrow3D([self.goal[0], self.goal[0]+self.arrow_len*self.tangent_vec_goal[0]],
                              [self.goal[1], self.goal[1]+self.arrow_len*self.tangent_vec_goal[1]],
                              [self.goal[2], self.goal[2]+self.arrow_len*self.tangent_vec_goal[2]],
                              arrowstyle="fancy", lw=1, mutation_scale=10, color="b")
      self.ax.add_artist(atangent_goal)
      mypause(pause) # updates plot without losing focus
      # save frames of plot
      if save_frames == True:
         frame = 1000
         self.fig.savefig("figures/frame"+str(frame)[1:]+".png")
         self.frame += 1

   def init_render(self):
      """ sets up 3d plot """
      plt.ion() # interactive plot mode, panning, zooming enabled
      self.fig = plt.figure(figsize=(9.5,7.2))
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axe limits and labels
      self.ax.set_xlim([-0.5*self.lmax, 0.5*self.lmax])
      self.ax.set_ylim([-0.5*self.lmax, 0.5*self.lmax])
      self.ax.set_zlim([0.0, self.lmax])
      self.ax.set_xlabel("X")
      self.ax.set_ylabel("Y")
      self.ax.set_zlabel("Z")
      # add coordinate 3 arrows of base frame, have to be defined once!
      self.arrow_len = 0.02
      ax_base = Arrow3D([0.0, self.arrow_len], [0.0, 0.0], [0.0, 0.0],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      ay_base = Arrow3D([0.0, 0.0], [0.0, self.arrow_len], [0.0, 0.0],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="g")
      az_base = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, self.arrow_len],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      self.ax.add_artist(ax_base)
      self.ax.add_artist(ay_base)
      self.ax.add_artist(az_base)
      plt.show() # display figure and bring focus (once) to plotting window
      self.fig.tight_layout() # fits the plot to window size

"""MAIN"""
#env = ForwardKinematicsOneSegment(lmin=0.075, lmax=0.125, d=0.01, n=10)
#state = env.reset(0.1, 0.1, 0.1)
#actions = [-env.delta_l, 0.0, env.delta_l]

"""random move"""
#episode = 0
#max_episodes = 5
#while episode < max_episodes:
#   dl1 = np.random.choice(actions); dl2 = np.random.choice(actions); dl3 = np.random.choice(actions)
#   t = time.time()
#   _, _, done, _ = env.step(dl1, dl2, dl3)
#   print("STEP TIME", time.time() - t)
#   t = time.time()
#   env.render()
#   print("RENDER TIME:", time.time()-t)
#   if done:
#
#      episode += 1
#      env.reset()

"""linear move, printing rewards"""
#rewards = []
#state = env.reset(l1=0.1, l2=0.1, l3=0.1, l1goal=0.105, l2goal=0.105, l3goal=0.105)
#pause = 0.01
#for i in range(25):
#   state, reward, done, info = env.step(-env.delta_l, -env.delta_l, -env.delta_l)
##   dl1 = np.random.choice(actions); dl2 = np.random.choice(actions); dl3 = np.random.choice(actions)
##   state, reward, done, info = env.step(dl1, dl2, dl3)
#   print("Distance covered: {:3.1f}mm, reward: {:6.2f}".format(-1000*(env.new_dist_euclid-env.old_dist_euclid), reward))
#   rewards.append(reward)
#   env.render(pause=pause)
#for i in range(500):
#   state, reward, done, info = env.step(env.delta_l, env.delta_l, env.delta_l)
##   dl1 = np.random.choice(actions); dl2 = np.random.choice(actions); dl3 = np.random.choice(actions)
##   state, reward, done, info = env.step(dl1, dl2, dl3)
#   print("Distance covered: {:3.1f}mm, reward: {:6.2f}".format(-1000*(env.new_dist_euclid-env.old_dist_euclid), reward))
#   rewards.append(reward)
#   env.render(pause=pause)
#   if done:
#      print(reward, info)
#      break
