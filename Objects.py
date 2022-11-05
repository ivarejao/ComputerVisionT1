# Code made in Pycharm by Igor Varejao
import numpy as np
from math import pi, cos, sin
from abc import ABC, abstractmethod

class Transformation(ABC):

    @property
    @abstractmethod
    def transformation(self):
        pass

    def dot(self, obj : np.ndarray) -> np.ndarray:
        return self.transformation.dot(obj)

class Rotatation(Transformation):

    @property
    def transformation(self):
        return self.R

    def __init__(self, angle, axis : str):
        R = np.eye(4, 4)
        if (axis == "x"):
            R[1:3, 1:3] = np.array([[cos(angle), -sin(angle)],
                                    [sin(angle), cos(angle)]])
        elif (axis == "y"):
            R[0] = [cos(angle), 0, sin(angle), 0]
            R[2] = [-sin(angle), 0, cos(angle), 0]
        elif (axis == "z"):
            R[:2, :2] = np.array([[cos(angle), -sin(angle)],
                                  [sin(angle), cos(angle)]])
        else:
            raise NotImplementedError("axis %s not implemented" % axis)

        self.R = R


class Translation(Transformation):

    @property
    def transformation(self):
        return self.T

    def __init__(self, x : int, y : int, z: int):
        T = np.eye(4, 4)
        T[:3, -1] = [x, y, z]
        self.T = T


class Cam():

    def __init__(self, base : np.ndarray):
        self.base = base
        self.cam = base

    def transform_on_own_ref(self, transf : Transformation):
        self.cam = self.cam.dot(transf.dot(self.base))

    def transform_on_world_ref(self, transf : Transformation):
        self.cam = transf.dot(self.cam)
