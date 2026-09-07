# 3d plot of the grids of celestial sphere
#Original Code from Tomoki Terasaki 2024
import numpy as np
import so3g.proj.quat as quat
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def plot_sphere_grid(ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x,y,z, rstride=5, cstride=5, color='gray', alpha=0.2)
    ax.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='black', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def plot_from_thetaphi(ax, theta, phi, label, **kwargs):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    ax.plot(x, y, z, label=label,**kwargs)
    return

def plot_from_q(ax, q, label, **kwargs):
    theta, phi, _ = quat.decompose_iso(q)
    #label += f'(theta={theta:.2f}, phi={phi:.2f})'
    plot_from_thetaphi(ax, theta, phi, label, **kwargs)
    return