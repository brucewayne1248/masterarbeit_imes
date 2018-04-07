import numpy as np
from math import sqrt, asin, atan2, cos, sin

import math

class ForwardKinematicsTwoSegments():
   """
   class handling the forward kinematics of a
   double segment tendon driven continuum robot

   l11, l12, l13  initial tendon lengths of proximal segment [m]
   l21, l22, l23  initial tendon lengths of distal segment (extending proximal segment) [m]
   lmin1, lmax1   min and max of proximal tendon lengths [m]
   lmin2, lmax2   min and max of distal tendon lengths [m]
   d              pitch distance to cable guides [m]
   n              number of units within one segment
   """
   def __init__(self, l11, l12, l13, l21, l22, l23, lmin1, lmax1, lmin2, lmax2, d, n):
      self.l11 = l11; self.l12 = l12; self.l13 = l13;
      self.l21 = l21; self.l22 = l22; self.l23 = l23;
#      self.delta_l1 = self.l21-self.l11;
#      self.delta_l2 = self.l22-self.l12;
#      self.delta_l3 = self.l23-self.l13;
      self.lmin1 = lmin1; self.lmax1 = lmax1;
      self.lmin2 = lmin2; self.lmax2 = lmax2;
      self.d = d
      self.n = n
      # some helpful expressions to shorten calculations
      self.lsum1, self.lsum2 = self.get_lsum1(); #self.lsum2 = self.get_lsum2()   # l1+l2+l3
      self.expr1 = self.get_expr1(); self.expr2 = self.get_expr2() # l1²+l2²+l3²-l1*l2-l1*l2-l2*l3
      # calculate initial configuration parameters
      self.l_center1, self.l_center2 = self.get_center_length()
      self.kappa1, self.kappa2 = self.get_kappa()
      self.phi1, self.phi2 = self.get_phi()
      self.arc_length1, self.arc_length2 = self.get_arc_length()
      self.configuration_space = self.get_configuration_space()
      # calculate the Frenet Frame of the tip position
      self.normal_vector, self.tanget_vector, self.binormal_vector = self.get_frenet_frame()

   def get_lsum1(self):
      """ returns the sums of proximal, distal tendon lengths """
      return self.l11+self.l12+self.l13, self.l21+self.l22+self.l23

   def get_lsum2(self):
      """ returns the sum of distal tendon lengths """
      return self.l21+self.l22+self.l23

   def get_expr1(self):
      """ returns the value of the expression l11²+l12²+l13²-l11*l12-l11*l11-l12*l13 """
      return self.l11**2+self.l12**2+self.l13**2 - \
             self.l11*self.l12-self.l11*self.l13-self.l12*self.l13

   def get_expr2(self):
      """ returns the value of the expression l21²+l22²+l23²-l21*l22-l21*l21-l22*l23 """
      return self.l21**2+self.l22**2+self.l23**2 - \
             self.l21*self.l22-self.l21*self.l23-self.l22*self.l23
#      return self.delta_l1**2+self.delta_l2**2+self.delta_l3**2 - \
#             self.delta_l1*self.delta_l2-self.delta_l1*self.delta_l3-self.delta_l2*self.delta_l3

   def get_kappa(self):
      """ returns the curvatures kappa1, kappa2 of the continuum robot [m^(-1)] """
      return 2*sqrt(self.expr1) / (self.d*self.lsum1), 2*sqrt(self.expr2) / (self.d*self.lsum2)

   def get_phi(self):
      """ returns the current bending angles phi1, phi2 of the continuum robot [rad] """
      return atan2(sqrt(3)*(self.l12+self.l13-2*self.l11), 3*(self.l12-self.l13)), \
             atan2(sqrt(3)*(self.l22+self.l23-2*self.l21), 3*(self.l22-self.l23))

   def get_arc_length(self):
      """ returns the current arc length of primary backbone [m] """
      if self.l11 == self.l12 == self.l13:
         arc_length1 = self.lsum1 / 3
      else:
         arc_length1 = self.n*self.d*self.lsum1 / (sqrt(self.expr1)) * asin(sqrt(self.expr1)/(3*self.n*self.d))
      if self.l21 == self.l22 == self.l23:
         arc_length2 = self.lsum2 / 3
      else:
         arc_length2 = self.n*self.d*self.lsum2 / (sqrt(self.expr2)) * asin(sqrt(self.expr2)/(3*self.n*self.d))
      return arc_length1, arc_length2

   def get_center_length(self):
      """ returns the length of the virtual straight center tendon/cable [m] """
      return self.lsum1 / 3, self.lsum2 /3

   def get_configuration_space(self):
      """ returns an array containing kappa, phi, segment_length of continuum robot"""
      print("kappa: ", self.kappa, " phi: ", self.phi*180/math.pi, " arc length: ", self.arc_length, " center length: ", self.l_center)
      return np.array([self.kappa1, self.phi1, self.arc_length1]), \
             np.array([self.kappa2, self.phi2, self.arc_length2])

   def get_frenet_frame(self):
      """ returns three vectors of the Local Frenet Frame of the robot tip """
      normal_vector = np.array([cos(self.phi)*cos(self.kappa*self.arc_length),
                                sin(self.phi)*cos(self.kappa*self.arc_length),
                                cos(self.kappa*self.arc_length)])
      tangent_vector = np.array([cos(self.phi)*sin(self.kappa*self.arc_length),
                                 sin(self.phi)*sin(self.kappa*self.arc_length),
                                 cos(self.kappa*self.arc_length)])
      binormal_vector = np.cross(tangent_vector, normal_vector)
      return normal_vector, tangent_vector, binormal_vector

   def get_tip_position1(self):
      """ returns the tip position in cartesian coordiantes [x, y, z] [m]"""
      if self.kappa1 == 0:
         return np.array([0.0, 0.0, self.arc_length1])
      else:
         return np.array([cos(self.phi)*(1-cos(self.kappa*self.arc_length))/self.kappa,
                       sin(self.phi)*(1-cos(self.kappa*self.arc_length))/self.kappa,
                       sin(self.kappa*self.arc_length)/self.kappa])



   def render(self):
      pass

myclass = ForwardKinematicsOneSegment(0.10, 0.1, 0.1, 0.01, 5)
print("tip position", myclass.get_tip_position())