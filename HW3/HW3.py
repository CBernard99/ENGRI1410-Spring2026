# ================================================================
# INTERACTIVE GRADIENT DESCENT LAB (Final UI)
# - 1D sliders directly above 1D graph
# - 2D sliders directly above 3D+contour graphs
# - per-slider: ▲ ▼ reset
# - global reset button shown above controls for BOTH sections
# - rotatable 3D landscape (Plotly) with start point + trajectory on surface
# - starting point marked with a star on BOTH plots
# - stable display windows so nonconvex/w^4 are readable
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import VBox, HBox

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "colab"

# ================================================================
# DEFAULTS
# ================================================================

defaults = {
    "w0": 4.00,
    "eta1": 0.20,
    "steps1": 12,
    "u0": 0.20,
    "v0": 0.20,
    "eta2": 0.20,
    "steps2": 20
}

# ================================================================
# NUMERICAL SAFETY (prevents overflow)
# ================================================================

W_LIMIT_1D = 1e3
UV_LIMIT_2D = 1e3

# ================================================================
# LOSS FUNCTIONS (1D)
# ================================================================

def loss_and_grad(w, loss_name):
    # Safety first: never compute powers if w is already huge
    if (not np.isfinite(w)) or abs(w) > W_LIMIT_1D:
        return np.inf, 0.0

    if loss_name == "Quadratic: (w-3)^2":
        C = (w-3)**2
        dC = 2*(w-3)
        return C, dC

    if loss_name == "Nonconvex: (w^2-1)^2":
        w2 = w*w
        C  = (w2 - 1)*(w2 - 1)
        dC = 4*w*(w2 - 1)
        return C, dC

    if loss_name == "Flat near 0: w^4":
        w2 = w*w
        C  = w2*w2
        dC = 4*w*w2
        return C, dC

    raise ValueError("Unknown loss")


def run_gd_1d(w0, eta, steps, loss_name):
    ws = [w0]
    Cs = []
    w = w0

    for _ in range(steps):

        if (not np.isfinite(w)) or abs(w) > W_LIMIT_1D:
            break

        C, dC = loss_and_grad(w, loss_name)
        if not np.isfinite(C):
            break

        Cs.append(C)
        w = w - eta*dC
        ws.append(w)

    C_end, _ = loss_and_grad(w, loss_name)
    Cs.append(C_end)

    return np.array(ws), np.array(Cs)

# ================================================================
# LOSS FUNCTION (2D)
# ================================================================

def saddle_C(u, v):
    return u**2 + v**4 - v**2

def saddle_grad(u, v):
    du = 2*u
    dv = 4*v**3 - 2*v
    return du, dv

def run_gd_2d(u0, v0, eta, steps):
    us = [u0]
    vs = [v0]
    u, v = u0, v0

    for _ in range(steps):

        if (not np.isfinite(u)) or (not np.isfinite(v)) or abs(u) > UV_LIMIT_2D or abs(v) > UV_LIMIT_2D:
            break

        du, dv = saddle_grad(u, v)

        # optional gentle guard against explosive jumps (still illustrates divergence)
        if (not np.isfinite(du)) or (not np.isfinite(dv)):
            break

        u = u - eta*du
        v = v - eta*dv
        us.append(u)
        vs.append(v)

    return np.array(us), np.array(vs)

# ================================================================
# OUTPUT AREAS
# ================================================================

out1 = widgets.Output()
out2 = widgets.Output()

# ================================================================
# PLOTTING (1D)
# ================================================================

def update_plot_1d(*args):

    with out1:
        out1.clear_output(wait=True)

        loss = loss_dd.value

        # Teaching windows + stable y-limits (prevents trajectory blow-up from destroying the view)
        if loss == "Quadratic: (w-3)^2":
            wmin, wmax = -3, 9
            y_max = 60
        elif loss == "Nonconvex: (w^2-1)^2":
            wmin, wmax = -2.0, 2.0
            y_max = 10
        else:  # "Flat near 0: w^4"
            wmin, wmax = -2.0, 2.0
            y_max = 10

        W = np.linspace(wmin, wmax, 800)
        Cvals = np.array([loss_and_grad(w, loss)[0] for w in W])

        ws, Cs = run_gd_1d(w0_slider.value, eta1_slider.value, steps1_slider.value, loss)

        # clip for display
        Cvals_plot = np.clip(Cvals, -2, y_max)
        Cs_plot    = np.clip(Cs,    -2, y_max)

        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(W, Cvals_plot)          # blue loss curve
        ax.plot(ws, Cs_plot, "o-")      # orange trajectory

        # start marker (star)
        ax.plot(ws[0], Cs_plot[0], "*", markersize=15, color="gold")

        # show divergence if it stopped early
        if len(ws) < steps1_slider.value + 1:
            ax.text(0.02, 0.95, "DIVERGED (η too large)", transform=ax.transAxes,
                    color="red", fontsize=12, va="top")

        ax.set_xlim([wmin, wmax])
        ax.set_ylim([-2, y_max])
        ax.set_title(f"1D Gradient Descent: {loss}")
        ax.set_xlabel("w")
        ax.set_ylabel("C(w)")
        ax.grid(True)
        plt.show()

# ================================================================
# PLOTTING (2D) — Plotly rotatable 3D + contour
# ================================================================

def update_plot_2d(*args):

    with out2:
        out2.clear_output(wait=True)

        # grid for surface/contours
        U = np.linspace(-1.5, 1.5, 140)
        V = np.linspace(-1.5, 1.5, 140)
        UU, VV = np.meshgrid(U, V)
        CC = saddle_C(UU, VV)

        # trajectory
        us, vs = run_gd_2d(u0_slider.value, v0_slider.value, eta2_slider.value, steps2_slider.value)
        zs = saddle_C(us, vs)

        # start point
        u0 = us[0]
        v0 = vs[0]
        z0 = zs[0]

        # If diverged early, note it in title
        diverged = (len(us) < steps2_slider.value + 1)

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "xy"}]],
            column_widths=[0.52, 0.48],
            subplot_titles=("3D Loss Landscape (rotatable)", "Contour + Gradient Descent Path")
        )

        # --- left: 3D surface ---
        fig.add_trace(
            go.Surface(x=UU, y=VV, z=CC, showscale=False, opacity=0.90),
            row=1, col=1
        )

        # trajectory on the surface (line+markers)
        fig.add_trace(
            go.Scatter3d(x=us, y=vs, z=zs, mode="lines+markers",
                         marker=dict(size=3),
                         line=dict(width=6)),
            row=1, col=1
        )

        # start point as a star on the surface
        fig.add_trace(
            go.Scatter3d(x=[u0], y=[v0], z=[z0], mode="markers",
                         marker=dict(size=8, symbol="star")),
            row=1, col=1
        )

        # --- right: contour plot ---
        fig.add_trace(
            go.Contour(x=U, y=V, z=CC, contours=dict(showlabels=False), showscale=False),
            row=1, col=2
        )

        # trajectory on contour
        fig.add_trace(
            go.Scatter(x=us, y=vs, mode="lines+markers"),
            row=1, col=2
        )

        # start point star on contour
        fig.add_trace(
            go.Scatter(x=[u0], y=[v0], mode="markers",
                       marker=dict(size=14, symbol="star")),
            row=1, col=2
        )

        title = "2D Gradient Descent on  C(u,v)=u^2+v^4-v^2"
        if diverged:
            title += "   —   DIVERGED (η too large)"

        fig.update_layout(
            height=520,
            title=title,
            margin=dict(l=10, r=10, t=60, b=10)
        )

        # improve 3D view defaults
        fig.update_scenes(
            xaxis_title="u", yaxis_title="v", zaxis_title="C(u,v)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9))
        )

        fig.update_xaxes(title_text="u", row=1, col=2)
        fig.update_yaxes(title_text="v", row=1, col=2)

        fig.show()

# ================================================================
# SLIDER FACTORIES (with ▲ ▼ reset)
# ================================================================

def make_float_control(label, minv, maxv, slider_step, button_step, default):

    s = widgets.FloatSlider(
        description=f"{label} (def: {default:.2f})",
        min=minv, max=maxv, step=slider_step, value=default,
        continuous_update=False,
        style={'description_width': '220px'},
        layout=widgets.Layout(width='650px')
    )

    up_btn = widgets.Button(description="▲", layout=widgets.Layout(width="35px"))
    dn_btn = widgets.Button(description="▼", layout=widgets.Layout(width="35px"))
    rs_btn = widgets.Button(description="reset", layout=widgets.Layout(width="70px"))

    def do_up(_):
        s.value = min(s.max, round(s.value + button_step, 10))

    def do_dn(_):
        s.value = max(s.min, round(s.value - button_step, 10))

    def do_rs(_):
        s.value = default

    up_btn.on_click(do_up)
    dn_btn.on_click(do_dn)
    rs_btn.on_click(do_rs)

    return HBox([s, up_btn, dn_btn, rs_btn]), s


def make_int_control(label, minv, maxv, step, default):

    s = widgets.IntSlider(
        description=f"{label} (def: {default})",
        min=minv, max=maxv, step=step, value=default,
        continuous_update=False,
        style={'description_width': '220px'},
        layout=widgets.Layout(width='650px')
    )

    up_btn = widgets.Button(description="▲", layout=widgets.Layout(width="35px"))
    dn_btn = widgets.Button(description="▼", layout=widgets.Layout(width="35px"))
    rs_btn = widgets.Button(description="reset", layout=widgets.Layout(width="70px"))

    def do_up(_):
        s.value = min(s.max, s.value + 1)

    def do_dn(_):
        s.value = max(s.min, s.value - 1)

    def do_rs(_):
        s.value = default

    up_btn.on_click(do_up)
    dn_btn.on_click(do_dn)
    rs_btn.on_click(do_rs)

    return HBox([s, up_btn, dn_btn, rs_btn]), s

# ================================================================
# CONTROLS
# ================================================================

loss_dd = widgets.Dropdown(
    options=[
        "Quadratic: (w-3)^2",
        "Nonconvex: (w^2-1)^2",
        "Flat near 0: w^4"
    ],
    description="loss"
)

# 1D: float controls use arrow step = 0.01
w0_box, w0_slider       = make_float_control("w0",  -3,  5, slider_step=0.1,  button_step=0.01, default=defaults["w0"])
eta1_box, eta1_slider   = make_float_control("eta",  0,  2, slider_step=0.01, button_step=0.01, default=defaults["eta1"])
steps1_box, steps1_slider = make_int_control("steps", 1, 30, step=1, default=defaults["steps1"])

# 2D: float controls use arrow step = 0.01
u0_box, u0_slider       = make_float_control("u0",  -1,  1, slider_step=0.01, button_step=0.01, default=defaults["u0"])
v0_box, v0_slider       = make_float_control("v0",  -1,  1, slider_step=0.01, button_step=0.01, default=defaults["v0"])
eta2_box, eta2_slider   = make_float_control("eta",  0,  1.5, slider_step=0.01, button_step=0.01, default=defaults["eta2"])
steps2_box, steps2_slider = make_int_control("steps", 1, 40, step=1, default=defaults["steps2"])

# ================================================================
# GLOBAL RESET (two buttons, same behavior)
# ================================================================

global_reset_1 = widgets.Button(description="GLOBAL RESET", button_style="danger")
global_reset_2 = widgets.Button(description="GLOBAL RESET", button_style="danger")

def do_global_reset(_):
    w0_slider.value = defaults["w0"]
    eta1_slider.value = defaults["eta1"]
    steps1_slider.value = defaults["steps1"]
    u0_slider.value = defaults["u0"]
    v0_slider.value = defaults["v0"]
    eta2_slider.value = defaults["eta2"]
    steps2_slider.value = defaults["steps2"]

global_reset_1.on_click(do_global_reset)
global_reset_2.on_click(do_global_reset)

# ================================================================
# OBSERVERS
# ================================================================

for s in [loss_dd, w0_slider, eta1_slider, steps1_slider]:
    s.observe(update_plot_1d, names='value')

for s in [u0_slider, v0_slider, eta2_slider, steps2_slider]:
    s.observe(update_plot_2d, names='value')

# ================================================================
# DISPLAY LAYOUT (Global reset above each section)
# ================================================================

display(VBox([
    widgets.HTML("<h3>1D Gradient Descent</h3>"),
    global_reset_1,
    loss_dd,
    w0_box,
    eta1_box,
    steps1_box,
    out1,

    widgets.HTML("<hr><h3>2D Gradient Descent</h3>"),
    global_reset_2,
    u0_box,
    v0_box,
    eta2_box,
    steps2_box,
    out2
]))

# initial draw
update_plot_1d()
update_plot_2d()
