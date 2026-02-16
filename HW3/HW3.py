import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import interact, FloatSlider, IntSlider, Dropdown

def loss_and_grad(w, loss_name):
    # Returns (C(w), dC/dw) for 1D losses
    if loss_name == "Quadratic: (w-3)^2":
        C = (w-3)**2
        dC = 2*(w-3)
        return C, dC

    if loss_name == "Nonconvex: (w^2-1)^2":
        C = (w*w - 1)**2
        dC = 4*w*(w*w - 1)
        return C, dC

    if loss_name == "Flat near 0: w^4":
        C = w**4
        dC = 4*w**3
        return C, dC

    raise ValueError("Unknown loss")

def run_gd_1d(w0, eta, steps, loss_name):
    ws = [w0]
    Cs = []
    w = w0
    for k in range(steps):
        C, dC = loss_and_grad(w, loss_name)
        Cs.append(C)
        w = w - eta*dC
        ws.append(w)
    Cs.append(loss_and_grad(w, loss_name)[0])
    return np.array(ws), np.array(Cs)

def plot_1d(loss_name, w0, eta, steps):
    wmin, wmax = -4, 6
    W = np.linspace(wmin, wmax, 800)
    Cvals = np.array([loss_and_grad(w, loss_name)[0] for w in W])

    ws, Cs = run_gd_1d(w0, eta, steps, loss_name)

    plt.figure(figsize=(7,4))
    plt.plot(W, Cvals)
    plt.plot(ws, Cs, marker='o')
    plt.title(f"{loss_name} | w0={w0}, eta={eta}, steps={steps}")
    plt.xlabel("w")
    plt.ylabel("C(w)")
    plt.grid(True)
    plt.show()

# 2D "saddle-like" loss from class:
# C(u,v)=u^2+v^4-v^2
def saddle_like_C(u, v):
    return u**2 + v**4 - v**2

def saddle_like_grad(u, v):
    du = 2*u
    dv = 4*v**3 - 2*v
    return du, dv

def run_gd_2d(u0, v0, eta, steps):
    us = [u0]; vs = [v0]
    u, v = u0, v0
    for k in range(steps):
        du, dv = saddle_like_grad(u, v)
        u = u - eta*du
        v = v - eta*dv
        us.append(u); vs.append(v)
    return np.array(us), np.array(vs)

def plot_2d(u0, v0, eta, steps):
    U = np.linspace(-1.5, 1.5, 250)
    V = np.linspace(-1.5, 1.5, 250)
    UU, VV = np.meshgrid(U, V)
    CC = saddle_like_C(UU, VV)

    us, vs = run_gd_2d(u0, v0, eta, steps)

    plt.figure(figsize=(6,5))
    plt.contour(UU, VV, CC, levels=25)
    plt.plot(us, vs, marker='o')
    plt.title(f"2D: u^2+v^4-v^2 | (u0,v0)=({u0},{v0}), eta={eta}")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.grid(True)
    plt.show()

interact(
    plot_1d,
    loss_name=Dropdown(options=[
        "Quadratic: (w-3)^2",
        "Nonconvex: (w^2-1)^2",
        "Flat near 0: w^4"
    ], value="Quadratic: (w-3)^2"),
    w0=FloatSlider(min=-3, max=5, step=0.1, value=4.0),
    eta=FloatSlider(min=0.0, max=2.0, step=0.01, value=0.2),
    steps=IntSlider(min=1, max=30, step=1, value=12)
);

interact(
    plot_2d,
    u0=FloatSlider(min=-1.0, max=1.0, step=0.01, value=0.2),
    v0=FloatSlider(min=-1.0, max=1.0, step=0.01, value=0.2),
    eta=FloatSlider(min=0.0, max=1.5, step=0.01, value=0.2),
    steps=IntSlider(min=1, max=40, step=1, value=20)
);
