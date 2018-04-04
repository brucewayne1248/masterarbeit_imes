import numpy as np
from math import sqrt
from math import asin
from math import atan2

class ForwardKinematicsOneSegment():
   """
   class handling the forward kinematics of a
   single segment tendon driven continuum robot
   """
   def __init__(self, l1, l2, l3, d, n):
      """
      l1, l2, l3  initial lengths of tendons [m]
      d           distance from the center of a section to the center of an actuator [m]
      n           number of units within one segment
      """
      self.l1 = l1
      self.l2 = l2
      self.l3 = l3
      self.d = d
      # some helpful expressions to shorten calculations
      # l1+l2+l3
      self.lsum = self.get_lsum()
      # l1²+l2²+l3²-l1*l2-l1*l2-l2*l3
      self.expr1 = self.get_expr1()

      # calculate initial configuration parameters
      self.kappa = self.get_kappa()
      self.phi = self.get_phi()
      self.l_seg = self.get_segment_length()

   def get_lsum(self):
      """ returns the sum of all tendon lengths """
      return self.l1+self.l2+self.l3

   def get_expr1(self):
      """ returns the value of the expression l1²+l2²+l3²-l1*l2-l1*l2-l2*l3 """
      return self.l1**2+self.l2**2+self.l3**2-self.l1*self.l2-self.l1*self.l3-self.l2*self.l3

   def get_kappa(self):
      """ returns the current curvature of the continuum robot [m^(-1)] """
      if self.l1 == self.l2 == self.l3:
         kappa = float("Inf")
      else:
         kappa = 2*sqrt(self.expr1) / (self.d*self.lsum)
      return kappa

   def get_phi(self):
      """ returns the current bending angle phi of the continuum robot [rad] """
      return atan2(sqrt(3)*(self.l2+self.l3-2*self.l1), 3*(self.l2-self.l3))

   def get_segment_length(self):
      """ returns the current segment length of the continuum robot [m] """
      if self.l1 == self.l2 == self.l3:
         segment_length = self.lsum / 3
      else:
         segment_length = self.n*self.d*self.lsum / (2*sqrt(self.expr1)) * \
                           asin(sqrt(self.expr1)/(3*self.n*self.d))
      return segment_length

   def robot_independent_mapping(self):
      pass

   def robot_specific_mapping(self):
      pass

myclass = ForwardKinematicsOneSegment(0.1, 0.1, 0.1, 0.01, 5)