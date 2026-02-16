# ENGRI 1410 - Spring 2025
# Copyright Carl Bernard
#
# HW3.py
# Run this cell in Google Colab (no need to modify code).
# If widgets don't appear: Runtime -> Restart runtime, then run again.

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import display, clear_output

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "colab"

# ================================================================
# DEFAULTS + SAFETY
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

W_LIMIT_1D = 1e3
UV_LIMIT_2D = 1e3

UV_MIN, UV_MAX = -1.2, 1.2

# ================================================================
# 1D LOSSES
# ================================================================

def loss_and_grad(w, loss_name):
    if (not np.isfinite(w)) or abs(w) > W_LIMIT_1D:
        return np.inf, 0.0

    if loss_name == "Quadratic: (w-3)^2":
        return (w-3)**2, 2*(w-3)

    if loss_name == "Nonconvex: (w^2-1)^2":
        w2 = w*w
        return (w2-1)*(w2-1), 4*w*(w2-1)

    if loss_name == "Flat near 0: w^4":
        w2 = w*w
        return w2*w2, 4*w*w2

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
# 2D LOSS (saddle-like)
# ================================================================

def saddle_C(u, v):
    return u**2 + v**4 - v**2

def saddle_grad(u, v):
    return 2*u, (4*v**3 - 2*v)

def run_gd_2d(u0, v0, eta, steps):
    us = [u0]
    vs = [v0]
    u, v = u0, v0

    for _ in range(steps):
        if (not np.isfinite(u)) or (not np.isfinite(v)) or abs(u) > UV_LIMIT_2D or abs(v) > UV_LIMIT_2D:
            break
        du, dv = saddle_grad(u, v)
        if (not np.isfinite(du)) or (not np.isfinite(dv)):
            break
        u = u - eta*du
        v = v - eta*dv
        us.append(u)
        vs.append(v)

    return np.array(us), np.array(vs)

# ================================================================
# UI CONTROL FACTORIES: slider + ▲ ▼ + reset
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

def make_int_control(label, minv, maxv, default):
    s = widgets.IntSlider(
        description=f"{label} (def: {default})",
        min=minv, max=maxv, step=1, value=default,
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
# WIDGETS
# ================================================================

loss_dd = widgets.Dropdown(
    options=[
        "Quadratic: (w-3)^2",
        "Nonconvex: (w^2-1)^2",
        "Flat near 0: w^4"
    ],
    description="loss"
)

# 1D controls
w0_box, w0_slider           = make_float_control("w0",  -3,  5,  slider_step=0.1,  button_step=0.01, default=defaults["w0"])
eta1_box, eta1_slider       = make_float_control("eta",  0,  2,  slider_step=0.01, button_step=0.01, default=defaults["eta1"])
steps1_box, steps1_slider   = make_int_control("steps",  1, 30, default=defaults["steps1"])

# 2D controls
u0_box, u0_slider           = make_float_control("u0",  UV_MIN, UV_MAX, slider_step=0.01, button_step=0.01, default=defaults["u0"])
v0_box, v0_slider           = make_float_control("v0",  UV_MIN, UV_MAX, slider_step=0.01, button_step=0.01, default=defaults["v0"])
eta2_box, eta2_slider       = make_float_control("eta",  0.0,  1.5,    slider_step=0.01, button_step=0.01, default=defaults["eta2"])
steps2_box, steps2_slider   = make_int_control("steps",  1,  40, default=defaults["steps2"])

# GLOBAL RESET buttons
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
# RENDER FUNCTION (exact layout order requested)
# ================================================================

def render(*args):
    clear_output(wait=True)

    # -------------------------
    # 1D controls
    # -------------------------
    display(VBox([
        widgets.HTML("<h3>1D Gradient Descent</h3>"),
        global_reset_1,
        loss_dd,
        w0_box,
        eta1_box,
        steps1_box
    ]))

    # -------------------------
    # 1D plot
    # -------------------------
    loss = loss_dd.value
    if loss == "Quadratic: (w-3)^2":
        wmin, wmax = -3, 9
        y_max = 60
    elif loss == "Nonconvex: (w^2-1)^2":
        wmin, wmax = -2.0, 2.0
        y_max = 10
    else:  # w^4
        wmin, wmax = -2.0, 2.0
        y_max = 10

    W = np.linspace(wmin, wmax, 800)
    Cvals = np.array([loss_and_grad(w, loss)[0] for w in W])

    ws, Cs = run_gd_1d(w0_slider.value, eta1_slider.value, steps1_slider.value, loss)

    # clip for display
    Cvals_plot = np.clip(Cvals, -2, y_max)
    Cs_plot    = np.clip(Cs,    -2, y_max)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(W, Cvals_plot, label="Loss function")
    ax1.plot(ws, Cs_plot, "o-", color="orange", label="Iterations")
    ax1.plot(ws[0], Cs_plot[0], "*", markersize=15, color="gold", label="Starting value")

    if len(ws) < steps1_slider.value + 1:
        ax1.text(0.02, 0.95, "DIVERGED (η too large)", transform=ax1.transAxes,
                 color="red", fontsize=12, va="top")

    ax1.set_xlim([wmin, wmax])
    ax1.set_ylim([-2, y_max])
    ax1.set_title(f"1D Gradient Descent: {loss}")
    ax1.set_xlabel("w")
    ax1.set_ylabel("C(w)")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    plt.show()

    # -------------------------
    # Divider
    # -------------------------
    display(widgets.HTML("<hr>"))

    # -------------------------
    # 2D controls
    # -------------------------
    display(VBox([
        widgets.HTML("<h3>2D Gradient Descent</h3>"),
        global_reset_2,
        u0_box,
        v0_box,
        eta2_box,
        steps2_box
    ]))

    # -------------------------
    # 2D plots (Plotly)
    # -------------------------
    U = np.linspace(UV_MIN, UV_MAX, 170)
    V = np.linspace(UV_MIN, UV_MAX, 170)
    UU, VV = np.meshgrid(U, V)
    CC = saddle_C(UU, VV)

    us, vs = run_gd_2d(u0_slider.value, v0_slider.value, eta2_slider.value, steps2_slider.value)
    zs = saddle_C(us, vs)

    diverged2 = (len(us) < steps2_slider.value + 1)
    u_start, v_start, z_start = us[0], vs[0], zs[0]

    fig2 = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        subplot_titles=("3D Loss Landscape (drag to rotate)", "Contour + GD Path")
    )

    # --- 3D surface with mesh ---
    fig2.add_trace(
        go.Surface(
            x=UU, y=VV, z=CC,
            showscale=False,
            opacity=0.95,
            colorscale="Greys",
            showlegend=False,
            contours={
                "x": {"show": True, "color": "black", "width": 1},
                "y": {"show": True, "color": "black", "width": 1}
            },
            lighting=dict(ambient=0.8, diffuse=0.6)
        ),
        row=1, col=1
    )

    # --- 3D GD path: red, thicker line for visibility ---
    fig2.add_trace(
        go.Scatter3d(
            x=us, y=vs, z=zs,
            mode="lines+markers",
            marker=dict(size=3, color="red"),
            line=dict(width=7, color="red"),
            showlegend=False
        ),
        row=1, col=1
    )

    # --- 3D start marker: green diamond ---
    fig2.add_trace(
        go.Scatter3d(
            x=[u_start], y=[v_start], z=[z_start],
            mode="markers",
            marker=dict(size=10, symbol="diamond", color="green"),
            showlegend=False
        ),
        row=1, col=1
    )

    # --- contour: lines only, neutral ---
    fig2.add_trace(
        go.Contour(
            x=U, y=V, z=CC,
            showscale=False,
            colorscale="Greys",
            contours=dict(showlabels=False, coloring="lines"),
            line=dict(width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    # --- contour GD path: red, slightly thicker (small improvement) ---
    fig2.add_trace(
        go.Scatter(
            x=us, y=vs,
            mode="lines+markers",
            line=dict(color="red", width=4),
            marker=dict(color="red", size=7),
            showlegend=False
        ),
        row=1, col=2
    )

    # --- contour start marker: green diamond ---
    fig2.add_trace(
        go.Scatter(
            x=[u_start], y=[v_start],
            mode="markers",
            marker=dict(size=14, symbol="diamond", color="green"),
            showlegend=False
        ),
        row=1, col=2
    )

    title = "2D Gradient Descent on  C(u,v)=u^2 + v^4 - v^2"
    if diverged2:
        title += "  —  DIVERGED (η too large)"

    fig2.update_layout(
        height=580,
        title=title,
        margin=dict(l=10, r=10, t=90, b=10),
        showlegend=False
    )

    fig2.update_scenes(
        xaxis_title="u",
        yaxis_title="v",
        zaxis_title="C(u,v)",
        camera=dict(eye=dict(x=1.35, y=1.35, z=0.9))
    )
    fig2.update_xaxes(title_text="u", row=1, col=2)
    fig2.update_yaxes(title_text="v", row=1, col=2)

    # --- custom legends as annotations ---
    fig2.add_annotation(
        text="<b>3D Legend</b><br>"
             "<span style='color:green'>◆</span>: Start value<br>"
             "<span style='color:red'>━</span>: Gradient descent path",
        x=0.18, y=1.02,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12)
    )

    fig2.add_annotation(
        text="<b>Contour Legend</b><br>"
             "<span style='color:green'>◆</span>: Starting point<br>"
             "<span style='color:red'>━</span>: GD path",
        x=0.78, y=1.02,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12)
    )

    fig2.show()

# ================================================================
# OBSERVERS
# ================================================================

for s in [
    loss_dd, w0_slider, eta1_slider, steps1_slider,
    u0_slider, v0_slider, eta2_slider, steps2_slider
]:
    s.observe(render, names="value")

# initial draw
render()
