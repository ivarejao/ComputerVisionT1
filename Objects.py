# Code made in Pycharm by Igor Varejao
import math
from typing import List, Dict

import numpy as np
from math import pi, cos, sin
from abc import ABC, abstractmethod

def create_base():
    e1 = np.array([[1], [0], [0], [0]])  # X
    e2 = np.array([[0], [1], [0], [0]])  # Y
    e3 = np.array([[0], [0], [1], [0]])  # Z
    e4 = np.array([0, 0, 0, 1]).reshape(-1, 1)
    base = np.hstack((e1, e2, e3, e4))
    return base

class Transformation(ABC):

    @property
    @abstractmethod
    def transformation(self):
        pass

    @property
    @abstractmethod
    def ref(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    def dot(self, obj : np.ndarray) -> np.ndarray:
        return self.transformation.dot(obj)


class Rotation(Transformation):

    @property
    def transformation(self):
        return self.R

    @property
    def ref(self):
        return self.cam_ref

    def __init__(self, angle, axis : str, cam_ref=False):
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
        self.angle = angle
        self.axis = axis
        self.cam_ref = cam_ref

    def __str__(self):
        s = f"[Rotação] \n   eixo {self.axis} em {math.degrees(self.angle)}º\n   Ref: "
        s += "Câmera" if self.cam_ref else "Mundo"
        return s

    def to_dict(self):
        return dict(angle=self.angle,
                    axis=self.axis,
                    cam_ref=self.cam_ref)

    @classmethod
    def from_dict(cls, data : Dict):
        return Rotation(data["angle"], data["axis"], cam_ref=data["cam_ref"])


class Translation(Transformation):

    @property
    def transformation(self):
        return self.T

    @property
    def ref(self):
        return self.cam_ref

    def __init__(self, coord : int, axis : str, cam_ref=False):
        idx = {"x":0, "y":1, "z":2}[axis]
        T = np.eye(4, 4)
        coord_3d = np.zeros(3)
        coord_3d[idx] = coord
        T[:3, -1] = coord_3d
        self.T = T
        self.coord = coord
        self.axis = axis
        self.cam_ref = cam_ref

    def __str__(self):
        s = "[Translação] \n"
        s += f"   eixo {self.axis}: {self.coord} \n"
        s += f"   Ref: {'Câmera' if self.cam_ref else 'Mundo'}"
        return s

    def to_dict(self):
        return dict(
            coord=self.coord,
             axis=self.axis,
             cam_ref=self.cam_ref)

    @classmethod
    def from_dict(cls, data : Dict):
        return Translation(data["coord"], data["axis"], cam_ref=data["cam_ref"])

class Camera():

    def __init__(self):
        self.base = create_base()
        self.cam = create_base()
        self.all_transforms = []

    def add_transformation(self, t : Transformation):
        self.all_transforms.append(t)

    def transform(self, tr : Transformation):
        if tr.ref : # A transformação é no eixo da câmera
            self._transform_on_own_ref(tr)
        else:
            self._transform_on_world_ref(tr)
        self.add_transformation(tr)

    def _transform_on_own_ref(self, transf : Transformation):
        self.cam = self.cam.dot(transf.dot(self.base))

    def _transform_on_world_ref(self, transf : Transformation):
        self.cam = transf.dot(self.cam)

    def to_dict(self):
        return dict(all_transforms=[t.to_dict() for t in self.all_transforms])

    @classmethod
    def from_dict(cls, data : Dict):
        cam = Camera()
        for transf_dict in data["all_transforms"]:
            t = Translation.from_dict(transf_dict) if "coord" in transf_dict else Rotation.from_dict(transf_dict)
            cam.transform(t)
        return cam