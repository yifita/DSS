from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLOR = "g"


class Painter(object):
    def __init__(self, name):
        self.__fig = None
        self.__plot = None
        self.__ax = None
        self.__tmp_plot = []  # type: ignore
        self.picked = []  # type: ignore
        self.nnIdx = None
        self.name = name

    def interactive_3D_plot(self, points, title=None):
        self.data = points
        self.__fig = plt.figure(num=self.name)
        self.__ax = self.__fig.add_subplot(111, projection='3d')
        self.__ax.set_xlabel('X Axis')
        self.__ax.set_ylabel('Y Axis')
        self.__ax.set_zlabel('Z Axis')
        self.__ax.set_title(title)
        self.__plot = self.__ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                        c=DEFAULT_COLOR, picker=True, alpha=0.2)
        # self.__ax.set_aspect('equal')

        bb_max = np.amax(points, axis=0)
        bb_min = np.amin(points, axis=0)
        self.__fig.tight_layout()
        self.__ax.set_xlim(bb_min[0], bb_max[0])
        self.__ax.set_ylim(bb_min[1], bb_max[1])
        self.__ax.set_zlim(bb_min[2], bb_max[2])
        self.__fig.canvas.mpl_connect('pick_event', self.onpick)
        plt.show()

    def onpick(self, event):
        # clear colors
        if self.__tmp_plot is not None:
            for p in self.__tmp_plot:
                p.remove()
        # picked index
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d
        self.picked.append((x[ind], y[ind], z[ind]))
        print(self.picked)
        if self.nnIdx is not None:
            # nnIdx (N, k)
            idx = self.nnIdx[ind, :]
            # print(self.data[idx, :])
            self.__tmp_plot += [self.__ax.scatter(self.data[idx, 0],
                                                  self.data[idx, 1],
                                                  self.data[idx, 2],
                                                  c="r", marker="*", alpha=1.0)]
            self.__tmp_plot += [self.__ax.scatter(x[ind], y[ind], z[ind],
                                                  c="yellow", marker="*",
                                                  s=25, alpha=1.0)]
            self.__fig.canvas.draw()
            self.__fig.canvas.flush_events()

        self.picked[:] = []  # reset list for future pick events
