import numpy as np
from math import sqrt, asin, atan2, cos, sin
# libraries needed to render continuum robot
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
   """ pause function to be used in plotting loop to keep plot window in background """
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
   class handling the forward kinematics of a
   single segment tendon driven continuum robot

   lmin, lmax  min and max of tendon length [m]
   d           pitch distance to cable guides [m]
   n           number of units within one segment
   """
   def __init__(self, lmin, lmax, d, n):
      self.d = d
      self.n = n
      self.lmin = lmin
      self.lmax = lmax
      # define variables to be used after environment is reset
      self.l1 = None; self.l2 = None; self.l3 = None;
      self.base = np.array([0.0, 0.0, 0.0, 1]) # base vector used for transformations
      self.lenghts = None # array containing tendon lenghts [l1, l2, l3]
      self.lsum = None # sum of all tendons
      self.expr = None # useful expression for l1²+l2²+l3²-l1*l2-l1*l2-l2*l3
      self.center_len = None # virtual length center tendon
      self.kappa = None # curvature kappa [m^(-1)]
      self.phi = None # angle rotating arc out of x-z plane [rad]
      self.seg_len = None # total arc length [m]
      self.T01_frenet = None # frenet transformation matrix from base to point on segment 1
      self.T01_bishop = None # bishop transformation matrix from base to point on segment 1
      self.configuration_space = None # array containing [kappa, phi, seg_len]
      self.normal_vec_frenet = None # Frenet normal vector of robot tip
      self.tangent_vec_frenet = None # Frenet tangent vector of robot tip
      self.binormal_vec_frenet = None # Frenet binormal vector of robot tip
      self.normal_vec_bishop = None # Bishop normal vector of robot tip
      self.tangent_vec_bishop = None # Bishop tangent vector of robot tip
      self.binormal_vec_bishop = None # Bishop binormal vector of robot tip
      self.tip_vec = None # robot's tip vector [m] [x, y, z]
      # define plot variables
      self.fig = None
      self.ax = None
      self.scatter = None
      self.arrow_len = 0.03 # arrow length of coordinate frames
      self.frame = 1000
      self.fig_name = None

   def reset(self, l1=None, l2=None, l3=None):
      """ resets the tendon lengths and updates other variables accordingly """
      self.l1 = np.random.uniform(self.lmin, self.lmax) if l1 == None else l1
      self.l2 = np.random.uniform(self.lmin, self.lmax) if l2 == None else l2
      self.l3 = np.random.uniform(self.lmin, self.lmax) if l3 == None else l3
      self.lenghts = np.array([self.l1, self.l2, self.l3])
      # after resetting tendon lengths, variables need to be updated
      self.update_variables()

   def step(self, delta_l1, delta_l2, delta_l3):
      """ incrementally changes the tendon lengths and updates variables """
      self.l1 += delta_l1
      self.l2 += delta_l2
      self.l3 += delta_l3
      self.l1 = self.lmin if self.l1 < self.lmin else self.l1
      self.l2 = self.lmin if self.l2 < self.lmin else self.l2
      self.l3 = self.lmin if self.l3 < self.lmin else self.l3
      self.l1 = self.lmax if self.l1 > self.lmax else self.l1
      self.l2 = self.lmax if self.l2 > self.lmax else self.l2
      self.l3 = self.lmax if self.l3 > self.lmax else self.l3
      self.update_variables()

   def update_variables(self):
      """ updates all necessary variables after changing tendon lengths """
      self.lsum = self.l1+self.l2+self.l3
      self.expr = self.l1**2+self.l2**2+self.l3**2-self.l1*self.l2-self.l1*self.l3-self.l2*self.l3
      self.lenghts = np.array([self.l1, self.l2, self.l3])
      # in rare cases self.expr turns out to be infinitely small negative number ~ 0
      # in these cases self.expr has to be set to 0
      if self.expr > -1e-17 and self.expr < 0.0:
         self.expr = 0.0
      self.center_len = self.lsum / 3
      self.kappa = 2*sqrt(self.expr) / (self.d*self.lsum)
      self.phi = atan2(sqrt(3)*(self.l2+self.l3-2*self.l1), 3*(self.l2-self.l3))
      self.seg_len = self._seg_length()
      self.T01_frenet = self.transformation_matrix_frenet(self.kappa, self.phi, self.seg_len)
      self.T01_bishop = self.transformation_matrix_bishop(self.kappa, self.phi, self.seg_len)
      self.configuration_space = np.array([self.kappa, self.phi, self.seg_len])
      self.tip_vec = np.matmul(self.T01_bishop, self.base)[0:3]
      self.normal_vec_frenet = self.T01_frenet[0:3, 0]
      self.binormal_vec_frenet = self.T01_frenet[0:3, 1]
      self.tangent_vec_frenet = self.T01_frenet[0:3, 2]
      self.normal_vec_bishop = self.T01_bishop[0:3, 0]
      self.binormal_vec_bishop=  self.T01_bishop[0:3, 1]
      self.tangent_vec_bishop = self.T01_bishop[0:3, 2]

   def _seg_length(self):
      """ returns the current segment length of primary backbone [m] """
      if self.l1 == self.l2 == self.l3 or self.expr == 0.0:
         return self.lsum / 3
      else:
         return self.n*self.d*self.lsum / (sqrt(self.expr)) * asin(sqrt(self.expr)/(3*self.n*self.d))

   def transformation_matrix_frenet(self, kappa, phi, s):
      """ returns a 4x4 SE3 frenet transformation matrix """
      if kappa == 0.0:
         # See Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review
         # the entries (limits) of the 4th column in case kappa = 0 can be calculated by using L'Hopital's rule
         return np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), 0],
                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), 0],
                          [-sin(kappa*s), 0, cos(kappa*s), s],
                          [0, 0, 0, 1]])
      else:
         return np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-sin(kappa*s), 0, cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])

   def transformation_matrix_bishop(self, kappa, phi, s):
      """ returns a 4x4 SE3 frenet transformation matrix """
      if kappa == 0.0:
         # See Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review
         # the entries (limits) of the 4th column in case kappa = 0 can be calculated by using L'Hopital's rule
         return np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), 0],
                          [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), 0],
                          [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), s],
                          [0, 0, 0, 1]])
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
      """ converts configuration space to actuator space of the robot """
      if kappa == 0:
         l1 = s
         l2 = s
         l3 = s
      else:
         l1 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*sin(phi))
         l2 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa+self.d*sin(np.pi/3+phi))
         l3 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*cos(np.pi/6+phi))
      return l1, l2, l3

   def render(self, pause=0.05, frame="bishop", save_frames=False):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      if self.fig == None:
         self.init_render()

      points = self.get_points_on_arc(self.kappa, 100)
      while self.ax.lines:
         self.ax.lines.pop() # make sure only current arc of robot is plotted
      self.ax.plot(points[:,0], points[:,1], points[:,2], label="segment 1", c="black", linewidth=4)
      self.ax.legend()

      # base frame is always included in plot (first 3 artists)
      # remaining arrows are deleted and current axes are added
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

      atangent = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*tangent_vec[0]],
                         [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*tangent_vec[1]],
                         [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*tangent_vec[2]],
                         arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      anormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*normal_vec[0]],
                        [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*normal_vec[1]],
                        [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*normal_vec[2]],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      abinormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*binormal_vec[0]],
                          [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*binormal_vec[1]],
                          [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*binormal_vec[2]],
                          arrowstyle="-|>", lw=1.5, mutation_scale=10, color="g")
      self.ax.add_artist(atangent)
      self.ax.add_artist(anormal)
      self.ax.add_artist(abinormal)
      mypause(pause)

      if save_frames == True:
         self.fig.savefig("figures/frame"+str(self.frame)[1:]+".png")
         self.frame += 1

   def init_render(self):
      """ sets up 3d plot """
      plt.ion() # interactive plot mode, panning, zooming enabled
      self.fig = plt.figure(figsize=(9.5,7.2))
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axe limits and labels
      self.ax.set_xlim([-self.lmax, self.lmax])
      self.ax.set_ylim([-self.lmax, self.lmax])
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
env = ForwardKinematicsOneSegment(lmin=0.075, lmax=0.125, d=0.01, n=5)
env.reset(l1=0.1, l2=0.1, l3=0.1)

for i  in range(15):
   env.render(pause=0.05, frame="bishop")
   step_size = 0.01
#   env.step(np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size))
   env.step(0.001, 0.0, 0.0)
steps = 100
phi = np.linspace(env.phi, env.phi+2*np.pi, steps)

for i in range(steps):
   l1, l2, l3 = env.arc_params_to_tendon_lenghts(env.kappa, phi[i], env.seg_len)
   env.reset(l1, l2, l3)
   env.render(pause=0.05)





