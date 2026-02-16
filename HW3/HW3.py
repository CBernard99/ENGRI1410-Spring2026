import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from ipywidgets import HBox, VBox

# =====================================================
# DEFAULT VALUES
# =====================================================

defaults = {
    "w0": 4.0,
    "eta1": 0.20,
    "steps1": 12,
    "u0": 0.20,
    "v0": 0.20,
    "eta2": 0.20,
    "steps2": 20
}

# =====================================================
# LOSS FUNCTIONS
# =====================================================

def loss_and_grad(w, loss_name):
    if loss_name == "Quadratic: (w-3)^2":
        return (w-3)**2, 2*(w-3)

    if loss_name == "Nonconvex: (w^2-1)^2":
        return (w*w-1)**2, 4*w*(w*w-1)

    if loss_name == "Flat near 0: w^4":
        return w**4, 4*w**3


def run_gd_1d(w0, eta, steps, loss_name):
    ws = [w0]
    Cs = []
    w = w0
    for _ in range(steps):
        C, dC = loss_and_grad(w, loss_name)
        Cs.append(C)
        w -= eta*dC
        ws.append(w)
    Cs.append(loss_and_grad(w, loss_name)[0])
    return np.array(ws), np.array(Cs)

# =====================================================
# 2D LOSS
# =====================================================

def saddle_C(u,v):
    return u**2 + v**4 - v**2

def saddle_grad(u,v):
    du = 2*u
    dv = 4*v**3 - 2*v
    return du,dv

def run_gd_2d(u0,v0,eta,steps):
    us=[u0]
    vs=[v0]
    u,v=u0,v0
    for _ in range(steps):
        du,dv=saddle_grad(u,v)
        u -= eta*du
        v -= eta*dv
        us.append(u)
        vs.append(v)
    return np.array(us),np.array(vs)

# =====================================================
# PLOTTING FUNCTIONS
# =====================================================

def update_plots(*args):

    # ---------- FIGURE 1 ----------
    W=np.linspace(-4,6,800)
    Cvals=np.array([loss_and_grad(w,loss_dd.value)[0] for w in W])

    ws,Cs=run_gd_1d(w0_slider.value,
                    eta1_slider.value,
                    steps1_slider.value,
                    loss_dd.value)

    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(W,Cvals)
    ax.plot(ws,Cs,'o-')
    ax.plot(ws[0],Cs[0],'*',markersize=15,color='gold')  # start marker
    ax.set_title("1D Gradient Descent")
    ax.set_xlabel("w")
    ax.set_ylabel("C(w)")
    ax.grid(True)
    plt.show()

    # ---------- FIGURE 2 ----------
    U=np.linspace(-1.5,1.5,120)
    V=np.linspace(-1.5,1.5,120)
    UU,VV=np.meshgrid(U,V)
    CC=saddle_C(UU,VV)

    us,vs=run_gd_2d(
        u0_slider.value,
        v0_slider.value,
        eta2_slider.value,
        steps2_slider.value
    )

    fig=plt.figure(figsize=(12,5))

    # ---- 3D surface (LEFT) ----
    ax1=fig.add_subplot(1,2,1,projection='3d')
    ax1.plot_surface(UU,VV,CC,alpha=0.85)
    ax1.set_title("3D Loss Landscape")

    # ---- contour + trajectory (RIGHT) ----
    ax2=fig.add_subplot(1,2,2)
    ax2.contour(UU,VV,CC,levels=25)
    ax2.plot(us,vs,'o-')
    ax2.plot(us[0],vs[0],'*',markersize=15,color='gold')
    ax2.set_title("2D Gradient Descent Path")
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")
    ax2.grid(True)

    plt.show()

# =====================================================
# SLIDERS + RESET BUTTONS
# =====================================================

def make_slider(label,minv,maxv,step,default):
    s=widgets.FloatSlider(
        description=f"{label} (def: {default:.2f})",
        min=minv,max=maxv,step=step,value=default,
        continuous_update=False
    )
    b=widgets.Button(description="reset",layout=widgets.Layout(width="70px"))
    b.on_click(lambda x: setattr(s,'value',default))
    return HBox([s,b]),s

def make_int_slider(label,minv,maxv,step,default):
    s=widgets.IntSlider(
        description=f"{label} (def: {default})",
        min=minv,max=maxv,step=step,value=default,
        continuous_update=False
    )
    b=widgets.Button(description="reset",layout=widgets.Layout(width="70px"))
    b.on_click(lambda x: setattr(s,'value',default))
    return HBox([s,b]),s

# 1D controls
loss_dd = widgets.Dropdown(
    options=[
        "Quadratic: (w-3)^2",
        "Nonconvex: (w^2-1)^2",
        "Flat near 0: w^4"
    ],
    description="loss"
)

w0_box,w0_slider=make_slider("w0",-3,5,0.1,defaults["w0"])
eta1_box,eta1_slider=make_slider("eta",0,2,0.01,defaults["eta1"])
steps1_box,steps1_slider=make_int_slider("steps",1,30,1,defaults["steps1"])

# 2D controls
u0_box,u0_slider=make_slider("u0",-1,1,0.01,defaults["u0"])
v0_box,v0_slider=make_slider("v0",-1,1,0.01,defaults["v0"])
eta2_box,eta2_slider=make_slider("eta",0,1.5,0.01,defaults["eta2"])
steps2_box,steps2_slider=make_int_slider("steps",1,40,1,defaults["steps2"])

# GLOBAL RESET
global_reset=widgets.Button(description="GLOBAL RESET",button_style='danger')
def do_global_reset(x):
    for k,v in defaults.items():
        if k=="w0": w0_slider.value=v
        if k=="eta1": eta1_slider.value=v
        if k=="steps1": steps1_slider.value=v
        if k=="u0": u0_slider.value=v
        if k=="v0": v0_slider.value=v
        if k=="eta2": eta2_slider.value=v
        if k=="steps2": steps2_slider.value=v
global_reset.on_click(do_global_reset)

# observe changes
for s in [loss_dd,w0_slider,eta1_slider,steps1_slider,
          u0_slider,v0_slider,eta2_slider,steps2_slider]:
    s.observe(update_plots,names='value')

display(VBox([
    widgets.HTML("<h3>1D Gradient Descent</h3>"),
    loss_dd,w0_box,eta1_box,steps1_box,
    widgets.HTML("<hr><h3>2D Gradient Descent</h3>"),
    u0_box,v0_box,eta2_box,steps2_box,
    global_reset
]))

update_plots()
