import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QSpinBox
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import *
from math import pi, cos, sin
from Objects import Cam, Rotatation, Translation
from itertools import product
import mpl_toolkits.mplot3d.art3d as art3d

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3, suppress=True)


def set_plot(ax=None, figure=None, lim=[-2, 2]):
    if figure is None:
        figure = plt.figure(figsize=(8, 8))
    if ax is None:
        ax = plt.axes(projection='3d')

    ax.set_title("camera referecnce")
    ax.set_xlim(lim)
    ax.set_xlabel("x axis")
    ax.set_ylim(lim)
    ax.set_ylabel("y axis")
    ax.set_zlim(lim)
    ax.set_zlabel("z axis")
    return ax


# adding quivers to the plot
def draw_arrows(point, base, axis, length=1.5):
    # The object base is a matrix, where each column represents the vector 
    # of one of the axis, written in homogeneous coordinates (ax,ay,az,0)

    # Plot vector of x-axis
    axis.quiver(point[0], point[1], point[2], base[0, 0], base[1, 0], base[2, 0], color='red', pivot='tail',
                length=length)
    # Plot vector of y-axis
    axis.quiver(point[0], point[1], point[2], base[0, 1], base[1, 1], base[2, 1], color='green', pivot='tail',
                length=length)
    # Plot vector of z-axis
    axis.quiver(point[0], point[1], point[2], base[0, 2], base[1, 2], base[2, 2], color='blue', pivot='tail',
                length=length)
    return axis


# Código auxiliar deve ser ignorado

def translate(x, y, z):
    T = np.eye(4, 4)
    T[:3, -1] = [x, y, z]
    return T


def rotate(ang, axis):
    R = np.eye(4, 4)
    if (axis == "x"):
        R[1:3, 1:3] = np.array([[cos(ang), -sin(ang)],
                                [sin(ang), cos(ang)]])
    elif (axis == "y"):
        R[0] = [cos(ang), 0, sin(ang), 0]
        R[2] = [-sin(ang), 0, cos(ang), 0]
    else:
        R[:2, :2] = np.array([[cos(ang), -sin(ang)],
                              [sin(ang), cos(ang)]])
    return R

# Transforma a câmera no próprio referencial
def transf_cam_axis(cam : np.ndarray, transforms : List[np.ndarray], base : np.ndarray):
    new_cam = base
    for t in reversed(transforms):
        new_cam = t.dot(new_cam)
    new_cam = cam.dot(new_cam)
    return new_cam

# Transforma a camêra no referencial do mundo
def transf_world_axis(cam, transforms : List[np.ndarray]):
    for t in transforms:
        cam = t.dot(cam)
    return cam

def create_cube():
    # cube = np.array(list(product(range(-1, 2, 2), repeat=3)))
    cube = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 0, 1],
                     [0, 0, 1],
                     [0, 1, 1],
                     [1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [1, 1, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0],
                     [0, 1, 1],
                     [0, 1, 0],
                     [0, 0, 0],
                     [0, 0, 1]])

    cube = np.transpose(cube)

    # add a vector of ones to the house matrix to represent the house in homogeneous coordinates
    cube = np.vstack([cube, np.ones(np.size(cube, 1))])
    return cube

def plot_cube(ax, cube):
    bottom = cube[:, 0].min()
    top = cube[:, 0].max()
    width = top - bottom

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, (z, zdir) in enumerate(product([bottom, top], ['x', 'y', 'z'])):
        side = Rectangle((bottom, bottom), width, width, facecolor=colors[i])
        ax.add_patch(side)
        art3d.pathpatch_2d_to_3d(side, z=z, zdir=zdir)

def create_base():
    e1 = np.array([[1], [0], [0], [0]])  # X
    e2 = np.array([[0], [1], [0], [0]])  # Y
    e3 = np.array([[0], [0], [1], [0]])  # Z
    e4 = np.array([0, 0, 0, 1]).reshape(-1, 1)
    base = np.hstack((e1, e2, e3, e4))
    return base

if __name__ == "__main__":
    # Make chanonical base
    base = create_base()
    origin = np.array([0, 0, 0]).reshape(-1, 1)

    # Bota a câmera na orientação padrão, com o eixo z apontando pra frente
    # E translada para diferenciar do objeto
    a90 = -np.pi / 2  # 90º
    R = rotate(a90, "x")
    T = translate(0, -10, 0)
    M1 = T.dot(R)
    # cam = M1.dot(base)

    # Testando
    T1 = translate(0, 0, -10)
    R2 = rotate(-a90, "x")
    R3 = rotate(np.pi/4, "x")

    cam = transf_cam_axis(base, [R, T, T1, R2, R3], base)
    # cam = transf_world_axis(base, [R, T, T1, R2, R3])

    # Plota a câmera e o objeto
    ax0 = set_plot()
    cb = create_cube()
    plot_cube(ax0, cb)
    draw_arrows(cam[:, 3], cam[:, :3], ax0)
    plt.axis('scaled')
    # ax0.set_ylim([-12, 2])
    plt.title("Cam 1")
    plt.show()
