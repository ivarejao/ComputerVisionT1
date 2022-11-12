import math
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
from math import pi, cos, sin, radians
from itertools import product
import mpl_toolkits.mplot3d.art3d as art3d
import streamlit as st
from Objects import Rotation, Translation, Transformation, Camera

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

def set_picture():
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(720*px, 480*px))
    ax.set_xlim(0, 720)
    ax.set_ylim(480, 0)
    ax.xaxis.tick_top()
    return fig, ax

def get_mesh():
    your_mesh = mesh.Mesh.from_file('dice.stl')

    # Get the x, y, z coordinates contained in the mesh structure that are the 
    # vertices of the triangular faces of the object
    x = your_mesh.x.flatten()
    y = your_mesh.y.flatten()
    z = your_mesh.z.flatten()

    # Get the vectors that define the triangular faces that form the 3D object
    kong_vectors = your_mesh.vectors

    # Create the 3D object from the x,y,z coordinates and add the additional array of ones to 
    # represent the object using homogeneous coordinates
    
    kong = np.dot(translate(40, 40, -20), np.array([x.T,y.T,z.T,np.ones(x.size)]))
    return kong


# adding quivers to the plot
def draw_arrows(point, base, axis, length=5):
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

def image_projection(cam, cb, f, sx, sy, ox, oy, stheta = 0):
    g = np.linalg.inv(cam.cam)
    p0 = np.eye(3,4)
    k = np.array([
        [f*sx, f*stheta, ox],
        [0, f*sy, oy],
        [0, 0, 1]
    ])

    new_cb = np.dot(g, cb)
    l, c = new_cb.shape
    index_delete = []
    print(new_cb)
    for i in range(c):
        if new_cb[2, i] < 0:
            index_delete.insert(0, i)

    new_cb = np.delete(new_cb, index_delete, axis=1)
    print(new_cb)
            


    projection = np.dot(k, np.dot(p0, new_cb))
    l, c = projection.shape
    for i in range(l):
        for j in range(c):
            if(projection[2][j] != 0):
                projection[i][j] = projection[i][j]/projection[2][j]

    return projection

# Transforma a câmera no próprio referencial
def transf_cam_axis(cam : np.ndarray, transforms : List[Transformation], base : np.ndarray):
    new_cam = base
    for t in reversed(transforms):
        new_cam = t.dot(new_cam)
    new_cam = cam.dot(new_cam)
    return new_cam

# Transforma a camêra no referencial do mundo
def transf_world_axis(cam, transforms : List[Transformation]):
    for t in transforms:
        cam = t.dot(cam)
    return cam

def create_cube() -> np.array:
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
    ax.scatter(cube[0,:],cube[1,:],cube[2,:],'c', s = 0.5)
    # bottom = cube[:, 0].min()
    # top = cube[:, 0].max()
    # width = top - bottom

    # colors = ['b', 'g', 'r', 'c', 'm', 'y']
    # for i, (z, zdir) in enumerate(product([bottom, top], ['x', 'y', 'z'])):
    #     side = Rectangle((bottom, bottom), width, width, facecolor=colors[i])
    #     ax.add_patch(side)
    #     art3d.pathpatch_2d_to_3d(side, z=z, zdir=zdir)

def create_base():
    e1 = np.array([[1], [0], [0], [0]])  # X
    e2 = np.array([[0], [1], [0], [0]])  # Y
    e3 = np.array([[0], [0], [1], [0]])  # Z
    e4 = np.array([0, 0, 0, 1]).reshape(-1, 1)
    base = np.hstack((e1, e2, e3, e4))
    return base

def set_axes_equal(ax):
    #Make axes of 3D plot have equal scale so that spheres appear as spheres,
    #cubes as cubes, etc..  This is one possible solution to Matplotlib's
    #ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    #Input
    #  ax: a matplotlib axis, e.g., as output from plt.gca().
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_cam(cam : np.ndarray, cb: np.array) -> plt.Figure:
    ax0 = set_plot()
    set_axes_equal(ax0)
    plot_cube(ax0, cb)
    draw_arrows(cam[:, 3], cam[:, :3], ax0)
    ax0.text(cam[0, 3] + .15, cam[1, 3] + .15, cam[2, 3] + .15, "Cam")
    plt.axis('scaled')
    # ax0.set_ylim([-12, 2])
    return ax0.get_figure()

if __name__ == "__main__":
    # Make chanonical base
    base = create_base()
    origin = np.array([0, 0, 0]).reshape(-1, 1)

    # Bota a câmera na orientação padrão, com o eixo z apontando pra frente
    # E translada para diferenciar do objeto
    Rx = Rotation(-math.radians(90), "x", cam_ref=True)
    Ry = Rotation(-math.radians(90), "y", cam_ref=True)
    Tx = Translation(-5, "x", cam_ref=True)
    Ty = Translation(10, "y", cam_ref=True)
    Tz = Translation(-30, "z", cam_ref=True)

    if "cam" in st.session_state:
        dict_cam = st.session_state["cam"]
        cam = Camera.from_dict(dict_cam)
    else:
        cam = Camera()
        cam.transform(Rx)
        cam.transform(Tz)
        cam.transform(Tx)
        cam.transform(Ty)

        
    cb = get_mesh()



    # Começa a setar a interface gráfica
    st.set_page_config(layout="wide", page_icon="📷", page_title="Computer Vision T1")
    st.title("Computer Vision T1")

    # Cria os tabs
    world_view_tab, cam_view_tab = st.tabs(["Visão do mundo", "Visão da câmera"])

    with world_view_tab:
        row1_1, row1_2 = st.columns((2, 4))
        # Seção onde é inserido as transformações que devem ser feitas
        with row1_1:
            row1_1.subheader("Transformando a câmera")
            form1 = row1_1.form("Transformando")
            transform_type = form1.selectbox("Tipo de transformação", ["Rotação", "Translação"])
            if transform_type == "Translação":
                # Parâmetros Translação
                axis = form1.radio("Eixo", ["x", "y", "z"])
                c = form1.number_input("Valor", step=1)
            else:
                # Parâmetros Rotação
                axis = form1.radio("Eixo", ["x", "y", "z"])
                theta = form1.number_input("Ângulo(º)", step=1, format="%d")
            ref = form1.checkbox("Eixo da câmera")
            submit2 = form1.form_submit_button("Transformar câmera")
            if submit2:
                if transform_type == "Rotação":
                    trans = Rotation(math.radians(theta), axis, cam_ref=ref)
                else:
                    trans = Translation(c, axis, cam_ref=ref)
                # Aplica a transformação na câmera
                cam.transform(trans)

            # Reseta a câmera
            if st.button("Resetar câmera", type="primary"):
                del st.session_state["cam"]
                st.experimental_rerun()

        # Onde ocorre a inserção dos parâmetros intrísecos da câmera
        with row1_2:
            fig = plot_cam(cam.cam, cb)
            fig.set_dpi(300)
            st.write(fig)

    row2_1, row2_2 = st.columns((1, 4))
    # Parte que mostra o gráfico da visão do mundo da cena
    with row2_1:
        # st.header("Out side vision")
        # fig = plot_cam(cam)
        # st.write(fig)
        row2_1.subheader("Parâm. intrísecos da câmera")
        form2 = row2_1.form("Parâm. intrísecos da câmera")
        f = form2.number_input("f", step=0.1, value=0.5)
        sx = form2.number_input("sx", step=1, value=600)
        sy = form2.number_input("sy", step=1, value=600)
        ox = form2.number_input("ox", step=1, value=360)
        oy = form2.number_input("oy", step=1, value=240)
        # alphax = form2.number_input("Horizontal field of view", step=1)
        # alphay = form2.number_input("Vertical field of view", step=1)
        submit3 = form2.form_submit_button("Renderizar foto")

        # Seção que mostra o gráfico da visão da câmera da cena
        with row2_2:

        image = image_projection(cam, cb, f, sx, sy, ox, oy)

            # O gráfico da visão projetada da câmera tem que vir aqui
            projection_fig, projection_ax = set_picture()
            projection_ax.scatter(image[0,:],image[1,:], color ='c', s = 0.5)
        # projection_ax.plot(image[0, :], image[1, :])
        projection_ax.grid()
            # projection_fig.set_dpi(300)
            st.write(projection_fig)

        # Salva as informações da câmera
        st.session_state["cam"] = cam.to_dict()

        # Mostra as transformações aplicas à câmera
        with st.sidebar:
            st.subheader("Transformações aplicadas")
            for i, trf in enumerate(cam.all_transforms, 1):
                st.code(f"{i}. " + str(trf))