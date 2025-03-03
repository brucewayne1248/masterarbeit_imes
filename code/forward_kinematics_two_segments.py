import numpy as np
from math import sqrt, asin, atan2, cos, sin
# libraries needed to render continuum robot
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch # used to create Arrow3D
from mpl_toolkits.mplot3d import proj3d # used to get 3D arrows to work

class Arrow3D(FancyArrowPatch):
    """ used to create arrows in 3d plot """
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

class ForwardKinematicsTwoSegments():
   """
   class handling the forward kinematics of a
   single segment tendon driven continuum robot

   l1min, l1max   min and max lengths of segment 1 tendons [m]
   l2min, l2max   min and max lengths of segment 2 tendons [m]
   d              pitch distance to cable guides [m]
   n              number of units within one segment
   verbose        boolean for displaying values during calculations
   """
   r_static = -1
   r_goal = 10
   r_crash = -10
   gamma = 0.95 # discount factor

   def __init__(self, l1min, l1max, l2min, l2max, d, n, verbose=False):
      self.d = d
      self.n = n
      self.l1min = l1min; self.l1max = l1max
      self.l2min = l2min; self.l2max = l2max
      self.verbose = verbose # used to print further information when stepping or resetting environment
      # general variables
      self.base = np.array([0.0, 0.0, 0.0, 1]) # base vector
      self.l11 = None; self.l12 = None; self.l13 = None; # absolute&effective segment 1 tendon lengths
      self.l21 = None; self.l22 = None; self.l23 = None; # absolute segment 2 tendon lengths
      self.dl21 = None; self.dl22 = None; self.dl23 = None; # effective  segment 2 tendon lengths (needed for kappa2, phi2, seg_len2)
      self.goal = None # goal coordinates [x, y, z] in [m]
      # configuration space variables
      self.kappa1 = None # segment 1 curvature [m^(-1)]
      self.kappa2 = None # segment 2 curvature [m^(-1)]
      self.phi1 = None # segment 1 angle rotating arc out of x-z base plane [rad]
      self.phi2 = None # segment 2 angle [rad]
      self.seg_len1 = None # segment 1 total arc length [m]
      self.seg_len2 = None # segment 2 total arc length [m]
      # transformation matrices
      self.T01_frenet = None # frenet transformation matrix from base to segment 1 tip
      self.T12_frenet = None # frenet transformation matrix from segment 1 tip to segment 2 tip
      self.T02_frenet = None # frenet transformation matrix from base to segment 2 tip
      self.T01_bishop = None # bishop transformation matrix from base to segment 1 tip
      self.T12_bishop = None # bishop transformation matrix from segment 1 tip to segment 2 tip
      self.T02_bishop = None # bishop transformation matrix from base to segment 2 tip
      # Frenet and Bishop coordinate frame vectors
      self.tangent_vec_frenet1 = None # segment 1 tip Frenet tangent vector
      self.tangent_vec_frenet2 = None # segment 2 Frenet tangent vector
      self.normal_vec_frenet1 = None # segment 1 tip Frenet normal vector
      self.normal_vec_frenet2 = None # segment 2 tip Frenet normal vector
      self.binormal_vec_frenet1 = None # segment 1 tip Frenet binormal vector
      self.binormal_vec_frenet2 = None # segment 2 Frenet binormal vector
      self.tangent_vec_bishop1 = None # segment 1 tip bishop tangent vector
      self.tangent_vec_bishop2 = None # segment 2 bishop tangent vector
      self.normal_vec_bishop1 = None # segment 1 tip bishop normal vector
      self.normal_vec_bishop2 = None # segment 2 tip bishop normal vector
      self.binormal_vec_bishop1 = None # segment 1 tip bishop binormal vector
      self.binormal_vec_bishop2 = None # segment 2 bishop binormal vector
      self.tip_vec1 = None # segment 1 tip vector [m] [x, y, z]
      self.tip_vec2 = None # segment 2 tip vector [m] [x, y, z]
      # define plot variables
      self.fig = None
      self.ax = None
      self.frame = 1000 # used to name frames -> frame000, frame001, etc
      self.arrow_len = 0.03
      # variables needed for episodic reinforcement learning
      self.done = False

   def reset(self, l11=None, l12=None, l13=None, l21=None, l22=None, l23=None):
      """ resets the tendon lengths and updates other variables accordingly """
      self.l11 = np.random.uniform(self.l1min, self.l1max) if l11 == None else l11
      self.l12 = np.random.uniform(self.l1min, self.l1max) if l12 == None else l12
      self.l13 = np.random.uniform(self.l1min, self.l1max) if l13 == None else l13
      self.l21 = np.random.uniform(self.l2min, self.l2max) if l21 == None else l21
      self.l22 = np.random.uniform(self.l2min, self.l2max) if l22 == None else l22
      self.l23 = np.random.uniform(self.l2min, self.l2max) if l23 == None else l23
      # after resetting tendon lengths, variables need to be updated
      self.goal = self.get_goal()
      self.done = False
      self.update_variables()

   def get_goal(self):
      """ returns a random point in the workspace of the robot [x, y, z] in [m]"""
      l11 = np.random.uniform(self.l1min, self.l1max)
      l12 = np.random.uniform(self.l1min, self.l1max)
      l13 = np.random.uniform(self.l1min, self.l1max)
      l21 = np.random.uniform(self.l2min, self.l2max)
      l22 = np.random.uniform(self.l2min, self.l2max)
      l23 = np.random.uniform(self.l2min, self.l2max)
      dl21 = l21-l11; dl22 = l22-l12; dl23 = l23-l13
      kappa1, phi1, seg_len1 = self.configuration_space(l11, l12, l13, self.d, self.n)
      kappa2, phi2, seg_len2 = self.configuration_space(dl21, dl22, dl23, self.d, self.n)
      T01 = self.transformation_matrix_bishop(kappa1, phi1, seg_len1)
      T12 = self.transformation_matrix_bishop(kappa2, phi2, seg_len2)
      T02 = np.matmul(T01, T12)
      return np.matmul(T02, self.base)[0:3]

   def step(self, delta_l11, delta_l12, delta_l13, delta_l21, delta_l22, delta_l23):
      """ incrementally changes the tendon lengths and updates variables """
      self.l11 += delta_l11; self.l12 += delta_l12; self.l13 += delta_l13
      self.l21 += delta_l11; self.l22 += delta_l12; self.l23 += delta_l13
      self.l21 += delta_l21; self.l22 += delta_l22; self.l23 += delta_l23
      # check that all tendon lenghts are within limit
      self.l11 = self.l1min if self.l11 < self.l1min else self.l11
      self.l12 = self.l1min if self.l12 < self.l1min else self.l12
      self.l13 = self.l1min if self.l13 < self.l1min else self.l13
      self.l11 = self.l1max if self.l11 > self.l1max else self.l11
      self.l12 = self.l1max if self.l12 > self.l1max else self.l12
      self.l13 = self.l1max if self.l13 > self.l1max else self.l13
      self.l21 = self.l2min if self.l21 < self.l2min else self.l21
      self.l22 = self.l2min if self.l22 < self.l2min else self.l22
      self.l23 = self.l2min if self.l23 < self.l2min else self.l23
      self.l21 = self.l2max if self.l21 > self.l2max else self.l21
      self.l22 = self.l2max if self.l22 > self.l2max else self.l22
      self.l23 = self.l2max if self.l23 > self.l2max else self.l23
      old_tip_vec = self.tip_vec2 # used for potential reward
      self.update_variables()
      new_tip_vec = self.tip_vec2 # used for potential reward
      reward = self.r_static
      return reward

   def update_variables(self):
      """ updates all necessary variables after resetting or changing tendon lengths """
      self.dl21 = self.l21-self.l11; self.dl22 = self.l22-self.l12; self.dl23 = self.l23-self.l13;
      self.kappa1, self.phi1, self.seg_len1 = self.configuration_space(self.l11, self.l12, self.l13, self.d, self.n)
      self.kappa2, self.phi2, self.seg_len2 = self.configuration_space(self.dl21, self.dl22, self.dl23, self.d, self.n)
      # aquire transformation matrices and tips for segment 1 and 2
      self.T01_bishop = self.transformation_matrix_bishop(self.kappa1, self.phi1, self.seg_len1)
      self.T12_bishop = self.transformation_matrix_bishop(self.kappa2, self.phi2, self.seg_len2)
      self.T02_bishop = np.matmul(self.T01_bishop, self.T12_bishop)
      self.T01_frenet = self.transformation_matrix_frenet(self.kappa1, self.phi1, self.seg_len1)
      self.T12_frenet = self.transformation_matrix_frenet(self.kappa2, self.phi2, self.seg_len2)
      self.T02_frenet = np.matmul(self.T01_frenet, self.T12_frenet)
      self.tip_vec1 = np.matmul(self.T01_bishop, self.base)[0:3]
      self.tip_vec2 = np.matmul(self.T02_bishop, self.base)[0:3]
      # Frenet frames
      self.normal_vec_frenet1 = self.T01_frenet[0:3, 0]
      self.binormal_vec_frenet1 = self.T01_frenet[0:3, 1]
      self.tangent_vec_frenet1 = self.T01_frenet[0:3, 2]
      self.normal_vec_frenet2 = self.T02_frenet[0:3, 0]
      self.binormal_vec_frenet2 = self.T02_frenet[0:3, 1]
      self.tangent_vec_frenet2 = self.T02_frenet[0:3, 2]
      # Bishop frames
      self.normal_vec_bishop1 = self.T01_bishop[0:3, 0]
      self.binormal_vec_bishop1 = self.T01_bishop[0:3, 1]
      self.tangent_vec_bishop1 = self.T01_bishop[0:3, 2]
      self.normal_vec_bishop2 = self.T02_bishop[0:3, 0]
      self.binormal_vec_bishop2 = self.T02_bishop[0:3, 1]
      self.tangent_vec_bishop2 = self.T02_bishop[0:3, 2]

   def configuration_space(self, l1, l2, l3, d, n):
      """ returns the configuration parameters kappa, phi, seg_len of one segment """
      # useful expressions
      lsum = l1+l2+l3
      expr = l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3
      # in rare cases expr = -0.00000000002 when l1~l2~l3 due to floating point operations
      # in these cases expr has to be set to 0.0
      if expr > -1e-17 and expr < 0.0:
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

   def get_points_on_arc(self, num_points):
      """ returns np.arrays [num_points, 3] arc points [x(s), y(s), z(s)] [m] """
      points1 = np.zeros((num_points, 3)) # points placeholder for segment 1
      points2 = np.zeros((num_points, 3)) # points placeholder for segment 2
      s1 = np.linspace(0.0, self.seg_len1, num_points) # variable arc length 1
      s2 = np.linspace(0.0, self.seg_len2, num_points) # variable arc length 2
      for i in range(num_points):
         points1[i] = np.matmul(self.transformation_matrix_bishop(self.kappa1, self.phi1, s1[i]), self.base)[0:3]
      for i in range(num_points):
         T02_s = np.matmul(self.T01_bishop, self.transformation_matrix_bishop(self.kappa2, self.phi2, s2[i]))
         points2[i] = np.matmul(T02_s, self.base)[0:3]
      return points1, points2

   def arc_params_to_tendon_lenghts(self, kappa, phi, s):
      if kappa == 0:
         l1 = s; l2 = s; l3 = s
      else:
         l1 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*sin(phi))
         l2 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa+self.d*sin(np.pi/3+phi))
         l3 = 2*self.n*sin(kappa*s/(2*self.n))*(1/kappa-self.d*cos(np.pi/6+phi))
      return l1, l2, l3

   def render(self, pause=0.05, frame="bishop", save_frames=False):
      """ renders the 3D plot of the robot's arc, pause (float) determines plot speed """
      if self.fig == None:
         self.init_render()

      points1, points2 = self.get_points_on_arc(num_points=100)

      while self.ax.lines:
         self.ax.lines.pop() # delete previous plots
      self.ax.plot(points1[:,0], points1[:,1], points1[:,2], label="Segment 1", c="black", linewidth=3)
      self.ax.plot(points2[:,0], points2[:,1], points2[:,2], label="Segment 2", c="grey", linewidth=2)
      self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], label="Goal", c="lime", marker="*", markersize=15)
      self.ax.legend() # display legend

      if frame == "bishop":
         tangent_vec1 = self.tangent_vec_bishop1
         normal_vec1 = self.normal_vec_bishop1
         binormal_vec1 = self.binormal_vec_bishop1
         tangent_vec2 = self.tangent_vec_bishop2
         normal_vec2 = self.normal_vec_bishop2
         binormal_vec2 = self.binormal_vec_bishop2
      elif frame == "frenet":
         tangent_vec1 = self.tangent_vec_frenet1
         normal_vec1 = self.normal_vec_frenet1
         binormal_vec1 = self.binormal_vec_frenet1
         tangent_vec2 = self.tangent_vec_frenet2
         normal_vec2 = self.normal_vec_frenet2
         binormal_vec2 = self.binormal_vec_frenet2

      # add dynamic coordinate frenet frame of segment 1 tip
      while len(self.ax.artists) > 3:
         self.ax.artists.pop() # delete previous arrows, except base frame
      atangent1 = Arrow3D([self.tip_vec1[0], self.tip_vec1[0]+self.arrow_len*tangent_vec1[0]],
                          [self.tip_vec1[1], self.tip_vec1[1]+self.arrow_len*tangent_vec1[1]],
                          [self.tip_vec1[2], self.tip_vec1[2]+self.arrow_len*tangent_vec1[2]],
                          arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      anormal1 = Arrow3D([self.tip_vec1[0], self.tip_vec1[0]+self.arrow_len*normal_vec1[0]],
                         [self.tip_vec1[1], self.tip_vec1[1]+self.arrow_len*normal_vec1[1]],
                         [self.tip_vec1[2], self.tip_vec1[2]+self.arrow_len*normal_vec1[2]],
                         arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      abinormal1 = Arrow3D([self.tip_vec1[0], self.tip_vec1[0]+self.arrow_len*binormal_vec1[0]],
                           [self.tip_vec1[1], self.tip_vec1[1]+self.arrow_len*binormal_vec1[1]],
                           [self.tip_vec1[2], self.tip_vec1[2]+self.arrow_len*binormal_vec1[2]],
                           arrowstyle="-|>", lw=1, mutation_scale=10, color="g")
      self.ax.add_artist(atangent1)
      self.ax.add_artist(anormal1)
      self.ax.add_artist(abinormal1)
      # add dynamic coordinate frenet frame of segment 2 tip
      atangent2 = Arrow3D([self.tip_vec2[0], self.tip_vec2[0]+self.arrow_len*tangent_vec2[0]],
                          [self.tip_vec2[1], self.tip_vec2[1]+self.arrow_len*tangent_vec2[1]],
                          [self.tip_vec2[2], self.tip_vec2[2]+self.arrow_len*tangent_vec2[2]],
                          arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      anormal2 = Arrow3D([self.tip_vec2[0], self.tip_vec2[0]+self.arrow_len*normal_vec2[0]],
                         [self.tip_vec2[1], self.tip_vec2[1]+self.arrow_len*normal_vec2[1]],
                         [self.tip_vec2[2], self.tip_vec2[2]+self.arrow_len*normal_vec2[2]],
                         arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      abinormal2 = Arrow3D([self.tip_vec2[0], self.tip_vec2[0]+self.arrow_len*binormal_vec2[0]],
                           [self.tip_vec2[1], self.tip_vec2[1]+self.arrow_len*binormal_vec2[1]],
                           [self.tip_vec2[2], self.tip_vec2[2]+self.arrow_len*binormal_vec2[2]],
                           arrowstyle="-|>", lw=1, mutation_scale=10, color="g")
      self.ax.add_artist(atangent2)
      self.ax.add_artist(anormal2)
      self.ax.add_artist(abinormal2)
#      goal_vec = (self.goal-self.tip_vec2)/np.linalg.norm(self.goal-self.tip_vec2)
#      agoal = Arrow3D([self.tip_vec2[0], self.tip_vec2[0]+self.arrow_len*goal_vec[0]],
#                      [self.tip_vec2[1], self.tip_vec2[1]+self.arrow_len*goal_vec[1]],
#                      [self.tip_vec2[2], self.tip_vec2[2]+self.arrow_len*goal_vec[2]],
#                      arrowstyle="fancy", lw=0.5, mutation_scale=15, color="magenta")
#      self.ax.add_artist(agoal)
      mypause(pause) # pause video without losing focus of current window
      # save frames of plot if asked
      if save_frames == True:
         filename = "figures/frame"+str(self.frame)[1:]+".png"
         self.fig.savefig(filename)
         self.frame += 1

   def init_render(self):
      """ sets up 3D plot """
      plt.ion() # interactive plot mode, panning, zooming enabled
      self.fig = plt.figure(figsize=(9,7)) # create figure object
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axe limits and labels
      self.ax.set_xlim([-self.l1max, self.l1max])
      self.ax.set_ylim([-self.l1max, self.l1max])
      self.ax.set_zlim([-self.l1max, self.l1max])
      self.ax.set_xlabel("X")
      self.ax.set_ylabel("Y")
      self.ax.set_zlabel("Z")
      # add 3 arrows of coordinate base frame
      ax_base = Arrow3D([0.0, self.arrow_len], [0.0, 0.0], [0.0, 0.0],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="r")
      ay_base = Arrow3D([0.0, 0.0], [0.0, self.arrow_len], [0.0, 0.0],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="g")
      az_base = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, self.arrow_len],
                        arrowstyle="-|>", lw=1, mutation_scale=10, color="b")
      self.ax.add_artist(ax_base)
      self.ax.add_artist(ay_base)
      self.ax.add_artist(az_base)
      plt.show(block=False) # display figure and bring focus (once) to plotting window
      self.fig.tight_layout() # fits the plot to window size

"""MAIN"""
env = ForwardKinematicsTwoSegments(l1min=0.075, l1max=0.125, l2min=0.15, l2max=0.25, d=0.01, n=5)
env.reset(l11=0.1, l12=0.1, l13=0.1, l21=0.2, l22=0.2, l23=0.2)
step_size = 0.001
steps = 15
pause = 0.001
tipold = env.tip_vec2; tipnew = env.tip_vec2
dist = []
action = [-0.001, 0.0, 0.001]
for i in range(10000):
   tipold= tipnew
   tipnew = env.tip_vec2

   #   env.step(np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size),
#            np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size), np.random.uniform(-step_size, step_size))
   env.step(np.random.choice(action), np.random.choice(action), np.random.choice(action),
            np.random.choice(action), np.random.choice(action), np.random.choice(action))
   env.render()
#   dist.append(np.linalg.norm(tipnew-tipold))



#print("{:.5f}".format(np.array(dist).mean()))
"""used for demo"""
#for i  in range(steps):
#   env.step(0.001, 0.0, 0.0, 0.0, 0.0, 0.0)
#   env.render(pause=pause, frame="bishop")
#for i  in range(steps):
#   env.step(0.0, 0.0, 0.0, 0.001, 0.0, 0.0)
#   env.render(pause=pause, frame="bishop")
#for i  in range(steps):
#   env.step(0.0, 0.0, 0.0, -0.001, 0.0, 0.0)
#   env.render(pause=pause, frame="bishop")
##
#steps = 100
#phi1 = np.linspace(env.phi1, env.phi1+2*np.pi, steps)
#
#for i in range(steps):
#   l1new, l2new, l3new = env.arc_params_to_tendon_lenghts(env.kappa1, phi1[i], env.seg_len1)
#   dl11 = l1new-env.l11; dl12 = l2new-env.l12; dl13 = l3new-env.l13;
##   l21, l22, l23 = env.arc_params_to_tendon_lenghts(env.kappa2, env.phi2, env.seg_len2)
#   env.step(dl11, dl12, dl13, 0, 0, 0)
#   env.render(pause=pause, frame="bishop")
"""used for demo"""
