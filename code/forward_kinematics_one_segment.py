import numpy as np
from math import sqrt, asin, atan2, cos, sin
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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
      self.lenghts = np.array([self.l1, self.l2, self.l3]) # array containing tendon lenghts [l1, l2, l3]
      self.lsum = None # sum of all tendons
      self.expr = None # useful expression for l1²+l2²+l3²-l1*l2-l1*l2-l2*l3
      self.center_length = None # virtual length center tendon
      self.kappa = None # curvature kappa [m^(-1)]
      self.phi = None # angle rotating arc out of x-z plane [rad]
      self.seg_length = None # total arc length [m]
      self.configuration_space = None # array containing [kappa, phi, seg_length]
      self.normal_vector = None # Frenet normal vector of robot tip
      self.tangent_vector = None # Frenet tangent vector of robot tip
      self.binormal_vector = None # Frenet binormal vector of robot tip
      self.tip_vector = None # robot's tip vector [m] [x, y, z]

   def reset(self, l1=None, l2=None, l3=None):
      """ resets the tendon lengths and updates other variables accordingly """
      self.l1 = np.random.uniform(self.lmin, self.lmax) if l1 == None else l1
      self.l2 = np.random.uniform(self.lmin, self.lmax) if l2 == None else l2
      self.l3 = np.random.uniform(self.lmin, self.lmax) if l3 == None else l3
      self.lenghts = np.array([self.l1, self.l2, self.l3])
      # after resetting tendon lengths, variables need to be updated
      self.update_variables()

   def update_variables(self):
      """ updates all necessary variables after changing tendon lengths """
      self.lsum = self.l1+self.l2+self.l3
      self.expr = self.l1**2+self.l2**2+self.l3**2-self.l1*self.l2-self.l1*self.l3-self.l2*self.l3
      self.center_length = self.lsum / 3
      self.kappa = 2*sqrt(self.expr) / (self.d*self.lsum)
      self.phi = atan2(sqrt(3)*(self.l2+self.l3-2*self.l1), 3*(self.l2-self.l3))
      self.seg_length = self.get_seg_length()
      self.configuration_space = np.array([self.kappa, self.phi, self.seg_length])
      self.tip_vector = self.get_tip_position(self.kappa)
      self.normal_vector, self.tangent_vector, self.binormal_vector = self.get_frenet_frame()
      self.frenet_frame = np.array([self.normal_vector, self.tangent_vector, self.binormal_vector])
#      print("l1", self.l1, "l2", self.l2, "l3", self.l3)
      print("kappa", self.kappa, "phi", self.phi*180/math.pi, "seg_length", self.seg_length)
#      print("tip vector", self.tip_vector)
#      print("normal   vector", self.normal_vector)
#      print("tangent  vector", self.tangent_vector)
#      print("binormal vector", self.binormal_vector)

   def get_seg_length(self):
      """ returns the current arc length of primary backbone [m] """
      if self.l1 == self.l2 == self.l3:
         return self.lsum / 3
      else:
         return self.n*self.d*self.lsum / (sqrt(self.expr)) * asin(sqrt(self.expr)/(3*self.n*self.d))

   def get_tip_position(self, kappa):
      """ returns the tip position in cartesian coordiantes [x, y, z] [m] """
      if kappa == 0:
         return np.array([0.0, 0.0, self.seg_length])
      else:
         return np.array([cos(self.phi)*(1-cos(self.kappa*self.seg_length))/self.kappa,
                          sin(self.phi)*(1-cos(self.kappa*self.seg_length))/self.kappa,
                          sin(self.kappa*self.seg_length)/self.kappa])

   def get_point_on_arc(self, kappa, s):
      """ returns the position in cartesian coordiantes along the arc [x, y, z] [m] """
      if kappa == 0:
         return np.array([0.0, 0.0, s])
      else:
         return np.array([cos(self.phi)*(1-cos(self.kappa*s))/self.kappa,
                          sin(self.phi)*(1-cos(self.kappa*s))/self.kappa,
                          sin(self.kappa*s)/self.kappa])

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

   def get_frenet_frame(self):
      """ returns three unit vectors of the robot's tip Local Frenet Frame """
      normal_vector = np.array([cos(self.phi)*cos(self.kappa*self.seg_length),
                                sin(self.phi)*cos(self.kappa*self.seg_length),
                                cos(self.kappa*self.seg_length)])
      tangent_vector = np.array([cos(self.phi)*sin(self.kappa*self.seg_length),
                                 sin(self.phi)*sin(self.kappa*self.seg_length),
                                 cos(self.kappa*self.seg_length)])
      binormal_vector = np.cross(tangent_vector, normal_vector)
      return normal_vector, tangent_vector, binormal_vector

   def render(self):
      fig = plt.figure(1)
      plt.cla()
      ax = fig.gca(projection="3d")
      s = np.linspace(0, self.seg_length, 100)
      points = np.zeros((100,3))
      for idx, _ in enumerate(points):
         points[idx] = self.get_point_on_arc(self.kappa, s[idx])
      ax.plot(points[:,0], points[:,1], points[:,2], label="simple arc")
      ax.set_xlim((-self.lmax, self.lmax))
      ax.set_ylim((-self.lmax, self.lmax))
      ax.set_zlim((0, self.lmax))
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")
      ax.legend()
      plt.show()

myclass = ForwardKinematicsOneSegment(0.075, 0.125, 0.01, 5)
myclass.reset(l1=0.1, l2=0.1, l3=0.1)

for i in range(100):
   myclass.step(np.random.uniform(-0.001, 0.001), np.random.uniform(-0.001, 0.001), np.random.uniform(-0.001, 0.001))
   myclass.render()
   plt.pause(0.001)


#fig = plt.figure(1)
#ax = fig.gca(projection="3d")
#s = np.linspace(0, myclass.seg_length, 100)
#points = np.zeros((100,3))
#for idx, p in enumerate(points):
#   points[idx] = myclass.get_point_on_arc(myclass.kappa, s[idx])
#ax.plot(points[:,0], points[:,1], points[:,2], label="simple arc")
#plt.xlim((-myclass.lmax, myclass.lmax))
#ax.set_xlim((-myclass.lmax, myclass.lmax))
#plt.ylim((-myclass.lmax, myclass.lmax))
#ax.legend()
#ax.set_zlim(0, myclass.lmax)
#plt.show()
#print(points)

