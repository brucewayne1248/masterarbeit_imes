import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch # used to create Arrow3D
from mpl_toolkits.mplot3d import proj3d # used to get 3D arrows to work

class Arrow3D(FancyArrowPatch):
   """class used to render 3d arrows"""
   def __init__(self, xs, ys, zs, *args, **kwargs):
      FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs

   def draw(self, renderer):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      FancyArrowPatch.draw(self, renderer)

def mypause(interval):
   """ Pause function to be used in plotting loop to keep plot window in background and not lose focus at every frame. """
   backend = plt.rcParams['backend']
   if backend in matplotlib.rcsetup.interactive_bk:
      figManager = matplotlib._pylab_helpers.Gcf.get_active()
      if figManager is not None:
         canvas = figManager.canvas
         if canvas.figure.stale:
            canvas.draw()
         canvas.start_event_loop(interval)
         return