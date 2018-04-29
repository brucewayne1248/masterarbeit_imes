import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin
# libraries needed to render continuum robot
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch # used to create Arrow3D
from mpl_toolkits.mplot3d import proj3d # used to get 3D arrows to work
# video file
# import cv2 by deleting kinetic python path from sys (else error)
#import sys
#try:
#   sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
#except :
#   print("kinetic path already deleted")
#import cv2

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def mypause(interval):
   """ Pause function to be used in plotting loop to keep plot window in background. """
   backend = plt.rcParams['backend']
   if backend in matplotlib.rcsetup.interactive_bk:
      figManager = matplotlib._pylab_helpers.Gcf.get_active()
      if figManager is not None:
         canvas = figManager.canvas
         if canvas.figure.stale:
            canvas.draw()
         canvas.start_event_loop(interval)
         return

class ForwardKinematicsOneSegment():
   """
   class handling the forward kinematics of a single segment tendon driven continuum robot

   lmin, lmax: min and max tendon length [m]

   d:          pitch distance to cable guides [m]

   n:          number of units (spacer discs) within one segment
   """
   precision_digits = 16 # rounding precision needed for handling the singularity l1=l2=l3

   def __init__(self, lmin, lmax, d, n):
      self.lmin = lmin
      self.lmax = lmax
      self.d = d
      self.n = n
      # define variables to be used after environment is reset
      self.l1 = None; self.l2 = None; self.l3 = None;
      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.kappa = None # curvature kappa [m^(-1)]
      self.phi = None # angle rotating arc out of x-z plane [rad]
      self.seg_len = None # total arc length [m]
      self.T01_frenet = None # frenet transformation matrix from base to tip of segment 1
      self.T01_bishop = None # bishop transformation matrix from base to tip of segment 1
#      self.configuration_space = None # array containing [kappa, phi, seg_len]
      self.normal_vec_frenet = None # Frenet normal vector of robot tip
      self.tangent_vec_frenet = None # Frenet tangent vector of robot tip
      self.binormal_vec_frenet = None # Frenet binormal vector of robot tip
      self.normal_vec_bishop = None # Bishop normal vector of robot tip
      self.tangent_vec_bishop = None # Bishop tangent vector of robot tip
      self.binormal_vec_bishop = None # Bishop binormal vector of robot tip
      self.tip_vec = None # robot's tip vector [m] [x, y, z]
      # define plot variables
      self.fig = None # fig variable used for plotting
      self.ax = None # axis variable used for plotting
      self.scatter = None # used for scatter plotting the goal
      self.arrow_len = 0.02 # arrow length of coordinate frames
      self.frame = 1000 # used for naming frames when saving single pictures
      # variables needed in episodic reinforcement learning
      self.state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.reward = None # current reward
      self.done = None # indicates that episode is progress or over
      self.info = None # additional info returned by stepping the environment
      self.steps = None # actual steps taken per episode
      self.goal = None # goal to be reached by the robot's tip
      self.tangent_vec_goal = None # tangent vector of goal position
      self.n_states = 9
      self.n_actuators = 3
      self.n_actions = 3
      self.delta_l = 0.001
      self.max_steps = 200 # max steps per episode
      self.n_action_values = [-self.delta_l, 0.0, self.delta_l]
      self.r_crash = -self.max_steps
      self.r_goal = self.max_steps
      self.r_approach = -0.5
      self.r_depart = -1.0
      self.gamma = 0.99
      self.eps = 5e-3 # distance tolerance to reach goal

   def reset(self, l1=None, l2=None, l3=None, l1goal=None, l2goal=None, l3goal=None):
      """ Resets the environment and updates other variables accordingly. Returns state of new episode. """
      self.l1 = round(np.random.uniform(self.lmin, self.lmax), 3) if l1 == None else l1
      self.l2 = round(np.random.uniform(self.lmin, self.lmax), 3) if l2 == None else l2
      self.l3 = round(np.random.uniform(self.lmin, self.lmax), 3) if l3 == None else l3
      # after resetting tendon lengths, variables need to be updated
      self.update_variables()
      # create goal far enough away from tip-vetor
      self.goal = self.tip_vec
      while norm(self.goal-self.tip_vec) < 3*self.eps:
         self.set_goal(l1goal, l2goal, l3goal) # set a new goal for the episode
      self.state = self.get_state()
      self.reward = 0
      self.info = "Reset the environment"
      self.done = False
      self.info= ""
      self.steps = 0
      return self.state

   def set_goal(self, l1=None, l2=None, l3=None):
      """ Returns a random point in the workspace of the robot [x, y, z] in [m] and it's tangent vector. """
      l1 = np.random.uniform(self.lmin, self.lmax) if l1 == None else l1
      l2 = np.random.uniform(self.lmin, self.lmax) if l2 == None else l2
      l3 = np.random.uniform(self.lmin, self.lmax) if l3 == None else l3
      kappa, phi, seg_len = self.configuration_space(l1, l2, l3, self.d, self.n)
      T01 = self.transformation_matrix_bishop(kappa, phi, seg_len)
      self.goal = np.matmul(T01, self.base)[0:3]
      self.tangent_vec_goal =  T01[0:3, 2]

   def step(self, delta_l1, delta_l2, delta_l3):
      """ incrementally changes the tendon lengths and updates variables """
      # warn to reset the envinroment in case the episode is done
      if self.done == True:
         self.info = "Environment needs to be reset, episode is done."
#         print(self.info)
         return self.state, self.reward, self.done, self.info
      self.steps += 1
      self.l1 += delta_l1
      self.l2 += delta_l2
      self.l3 += delta_l3
      old_tip_vec = self.tip_vec
      old_dist = norm(self.goal-old_tip_vec)
      self.update_variables()
      # handling regular step
      new_tip_vec = self.tip_vec
      new_dist = norm(self.goal-new_tip_vec)
      self.state = self.get_state()
      if new_dist - old_dist >= 0.0:
         self.reward = self.r_depart
      else:
         self.reward = self.r_approach
      self.info = "EPISODE RUNNING @STEP {} DISTANCE: {:.4f}".format(self.steps, new_dist)
      # handling the case that actuator limits are exceeded
      lengths = [self.l1, self.l2, self.l3]
      for idx, l in enumerate(lengths):
         if round(l, self.precision_digits) < self.lmin or round(l, self.precision_digits) > self.lmax:
            self.state = self.get_state()
            self.reward = self.r_crash
            self.done = True
            self.info = "ACTUATOR: l{} = {:.4f} @step {}".format(idx+1, l, self.steps)
            return self.state, self.reward, self.done, self.info
      # handling goal reaching case
      if norm(new_tip_vec-self.goal) < self.eps:
         self.info = "GOAL!!! DISPLACEMENT {:.4f}m @step {}".format(norm(self.goal-self.tip_vec), self.steps)
         self.done = True
         self.reward += self.r_goal
         return self.state, self.reward, self.done, self.info
      # handling case when max steps are exceeded
      if self.steps >= self.max_steps:
         self.info = "MAX STEPS {} REACHED, DISTANCE {:.4f}.".format(self.max_steps, norm(self.goal-self.tip_vec))
         self.done = True

      return self.state, self.reward, self.done, self.info

   def get_state(self):
      return np.array([self.l1, self.l2, self.l3,
                       self.tip_vec[0], self.tip_vec[1], self.tip_vec[2],
                       self.goal[0]-self.tip_vec[0], self.goal[1]-self.tip_vec[1], self.goal[2]-self.tip_vec[2]])

   def update_variables(self):
      """ updates all necessary variables after changing tendon lengths """
      self.kappa, self.phi, self.seg_len = self.configuration_space(self.l1, self.l2, self.l3, self.d, self.n)
      self.T01_frenet = self.transformation_matrix_frenet(self.kappa, self.phi, self.seg_len)
      self.T01_bishop = self.transformation_matrix_bishop(self.kappa, self.phi, self.seg_len)
      self.tip_vec = np.matmul(self.T01_bishop, self.base)[0:3]
      self.normal_vec_frenet = self.T01_frenet[0:3, 0]
      self.binormal_vec_frenet = self.T01_frenet[0:3, 1]
      self.tangent_vec_frenet = self.T01_frenet[0:3, 2]
      self.normal_vec_bishop = self.T01_bishop[0:3, 0]
      self.binormal_vec_bishop=  self.T01_bishop[0:3, 1]
      self.tangent_vec_bishop = self.T01_bishop[0:3, 2]

   def configuration_space(self, l1, l2, l3, d, n):
      """ returns the configuration parameters kappa, phi, seg_len of one segment """
      # useful expressions
      lsum = l1+l2+l3
      expr = l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3
      # in rare cases expr ~ +-1e-17 when l1~l2~l3 due to floating point operations
      # in these cases expr has to be set to 0.0
      if round(abs(expr), self.precision_digits) == 0:
         expr = 0.0
      kappa = 2*sqrt(expr) / (d*lsum)
      phi = atan2(sqrt(3)*(l2+l3-2*l1), 3*(l2-l3))
      # calculate total segment length
      if l1 == l2 == l3 or expr == 0.0: # handling the singularity
         seg_len = lsum / 3
      else:
         seg_len = n*d*lsum / sqrt(expr) * asin(sqrt(expr)/(3*n*d))
      return kappa, phi, seg_len

   def transformation_matrix_frenet(self, kappa, phi, s):
      """ returns a 4x4 SE3 frenet transformation matrix """
      if round(kappa, self.precision_digits) == 0.0:
         # See Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review
         # the entries (limits) of the 4th column in case kappa = 0 can be calculated by using L'Hopital's rule
#         return np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), 0],
#                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), 0],
#                          [-sin(kappa*s), 0, cos(kappa*s), s],
#                          [0, 0, 0, 1]])
         T = np.identity(4)
         T[2, 3] = s
         return T
      else:
         return np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-sin(kappa*s), 0, cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])

   def transformation_matrix_bishop(self, kappa, phi, s):
      """ returns a 4x4 SE3 frenet transformation matrix """
      if round(kappa, self.precision_digits) == 0.0: # take floating point precision into consideration
         # See Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review
         # the entries (limits) of the 4th column in case kappa = 0 can be calculated by using L'Hopital's rule
#         return np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), 0],
#                          [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), 0],
#                          [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), s],
#                          [0, 0, 0, 1]])
         T = np.identity(4)
         T[2, 3] = s
         return T
      else:
         return np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])

   def get_points_on_arc(self, kappa, num_points):
      """ returns np.array([num_points, 3]) of arc points [x(s), y(s), z(s)] for plot [m] """
      points = np.zeros((num_points, 3))
      s = np.linspace(0, self.seg_len, num_points)
      for i in range(num_points):
         points[i] = np.matmul(self.transformation_matrix_bishop(self.kappa, self.phi, s[i]),
                                np.array([0.0, 0.0, 0.0, 1]))[0:3]
      return points

   def arc_params_to_tendon_lenghts(self, kappa, phi, s):
      """ converts configuration space [kappa, phi, s] to
          actuator space [l1, l2, l3] of the robot's segment """
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

      points = self.get_points_on_arc(self.kappa, 100) # points to be plotted from base to robot's tip

      while self.ax.lines:
         self.ax.lines.pop() # delete plots of previous frame
      self.ax.plot(points[:,0], points[:,1], points[:,2], label="segment 1", c="black", linewidth=4)
      self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="goal", c="magenta", marker="*", markersize=15)
      self.ax.legend()

      # delete arrows of previous frame, except base frame
      while len(self.ax.artists) > 3:
         self.ax.artists.pop()
      # add current frenet or bishop coordinate frame in plot
      if frame == "frenet":
         normal_vec = self.normal_vec_frenet
         tangent_vec = self.tangent_vec_frenet
         binormal_vec = self.binormal_vec_frenet
      elif frame == "bishop":
         normal_vec = self.normal_vec_bishop
         tangent_vec = self.tangent_vec_bishop
         binormal_vec = self.binormal_vec_bishop

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
      # vector pointing from robot's tip to goal point
#      goal_vec = (self.goal-self.tip_vec)/np.linalg.norm(self.goal-self.tip_vec)
#      agoal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*goal_vec[0]],
#                      [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*goal_vec[1]],
#                      [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*goal_vec[2]],
#                      arrowstyle="fancy", lw=0.5, mutation_scale=5, color="magenta")
#      self.ax.add_artist(agoal)
      # tangent vector indicating orientation of goal point
      atangent_goal = Arrow3D([self.goal[0], self.goal[0]+self.arrow_len*self.tangent_vec_goal[0]],
                              [self.goal[1], self.goal[1]+self.arrow_len*self.tangent_vec_goal[1]],
                              [self.goal[2], self.goal[2]+self.arrow_len*self.tangent_vec_goal[2]],
                              arrowstyle="fancy", lw=1, mutation_scale=10, color="b")
      self.ax.add_artist(atangent_goal)
      mypause(pause) # updates plot without losing focus
      # save frames of plot
      if save_frames == True:
         self.fig.savefig("figures/frame"+str(self.frame)[1:]+".png")
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
#state = env.reset(l1=0.1, l2=0.1, l3=0.1, l1goal=0.4, l2goal=0.4, l3goal=0.3)
###
#actions = [-0.0001, 0.0, 0.0001]
#total_episodes = 10000
#episode = 0
#move_dist = []
#while episode < total_episodes:
##   state, reward, done, info = env.step(-0.0001, -0.0001, -0.0001)
##   print(env.configuration_space(env.l1, env.l2, env.l3, env.d, env.n))
##   state, reward, done, info = env.step(0.001, 0, 0)
##   print(reward)
#   old_tip = env.tip_vec
#   state, reward, done, info = env.step(np.random.choice(actions), np.random.choice(actions), np.random.choice(actions))
#   new_tip = env.tip_vec
#   move_dist.append(norm(new_tip-old_tip))
#   if done:
#      episode += 1
#      env.reset(l1=0.1, l2=0.1, l3=0.1, l1goal=0.4, l2goal=0.4, l3goal=0.3)
##   env.render(pause=pause)
##   print(env.T01_bishop)
#
#print(sum(move_dist)/len(move_dist))

#pause = 0.00001
#rewards = []
#infos = []
#ep_reward = 0
#env.gamma = 0.9999999
#for i in range(5):
#   state, reward, done, info = env.step(0.001, 0.001, 0.001)
#   print(state)
#   rewards.append(reward)
#   env.render(pause=pause)
#for i in range(100):
#   state, reward, done, info = env.step(-0.001, -0.001, -0.001)
#   print(state)
#   rewards.append(reward)
#   env.render(pause=pause)
#   if done:
#      print(reward)
#      break

#
#steps = 100
#phi = np.linspace(env.phi, env.phi+2*np.pi, steps)

#l1, l2, l3 = env.arc_params_to_tendon_lenghts(env.kappa, phi[0], env.seg_len)
#print(l1, l2, l3)
#print(l1, l2, l3)
#print(env.l1, env.l2, env.l3)
#for i in range(steps):
#   l1, l2, l3 = env.arc_params_to_tendon_lenghts(env.kappa, phi[i], env.seg_len)
##   env.reset(l1, l2, l3)
#   env.step(l1-env.l1, l2-env.l2, l3-env.l3)
#   env.render(pause=0.05)
#   rewards.append(reward)
#
#
#   ep_reward += reward
#   if done == True:
##      state = env.reset(0.1, 0.1, 0.1)
#      rewards.append(ep_reward)
#
#      infos.append([episode, info])
#      ep_reward = 0
#      episode += 1
#
#
#print(np.mean(np.array(rewards)))

#env.set_goal(l1=0.11, l2=0.11, l3=0.11)
#while True:
#   env.render(pause=0.25, frame="bishop")
#
#   step_size = 0.001
##   env.reset()
##   r = env.step(np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size))
#
#
#   state, reward, done, info = env.step(0.001, 0.001, 0.001)
##   print(env.tip_vec, reward)
#   if done == True:
#      env.reset(0.1, 0.1, 0.1)
#   env.step(0.001, 0.0, 0.0)

#steps = 100
#phi = np.linspace(env.phi, env.phi+2*np.pi, steps)
#
#for i in range(steps):
#   l1, l2, l3 = env.arc_params_to_tendon_lenghts(env.kappa, phi[i], env.seg_len)
#   env.reset(l1, l2, l3)
#   env.render(pause=0.05)