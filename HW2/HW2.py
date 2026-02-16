# ENGRI 1410 - Spring 2026
# Copyright Carl Bernard
#
# HW2_Problem4_PhoneOverheating_NNExplorer_2x2.py
# Run this cell in Google Colab (no need to modify code).
# If widgets don't appear: Runtime -> Restart runtime, then run again.

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Checkbox, Dropdown, IntSlider
import matplotlib.patches as patches
from ipywidgets import VBox, HBox, Button, Layout, interactive_output
from IPython.display import display
from matplotlib.lines import Line2D

# --------------------------------------------------------
# Helper so that each slider has a reset to default button
# --------------------------------------------------------
def slider_with_reset(slider, default):
    btn = Button(
        description="↺",
        tooltip="Reset to default",
        layout={'width': '40px'}
    )

    def _reset(b):
        slider.value = default

    btn.on_click(_reset)
    return HBox([slider, btn])


# -----------------------------
# Math utilities
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------
# Synthetic but relatable dataset
# -----------------------------
def make_dataset(seed=0, n_per_corner=35, noise=0.18, pattern="xor_overheat"):
    """
    Synthetic phone-usage dataset in standardized coordinates:

      x1 ~ screen brightness deviation (standardized)
      x2 ~ CPU/GPU load deviation (standardized)

    Interpretation:
      x1 = -1  means "low brightness"      ;  x1 = +1 means "very high brightness"
      x2 = -1  means "light workload"      ;  x2 = +1 means "heavy workload"

    pattern:
      - "xor_overheat": overheat if exactly one of (brightness, load) is high (XOR)
      - "and_overheat": overheat if both brightness and load are high (AND)
      - "or_overheat":  overheat if brightness or load is high (OR)
    """
    rng = np.random.default_rng(seed)

    # Four cluster centers in standardized coordinates
    corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)

    X = []
    for c in corners:
        pts = c + noise * rng.standard_normal((n_per_corner, 2))
        X.append(pts)
    X = np.vstack(X)

    # Define "high" by sign of standardized coordinate
    bright_high = X[:, 0] > 0
    load_high   = X[:, 1] > 0

    if pattern == "xor_overheat":
        y = (bright_high ^ load_high).astype(int)
    elif pattern == "and_overheat":
        y = (bright_high & load_high).astype(int)
    elif pattern == "or_overheat":
        y = (bright_high | load_high).astype(int)
    else:
        raise ValueError("Unknown pattern")

    return X, y


def standardized_to_real_ticks():
    """
    Map standardized ticks [-1,0,1] to plausible phone-use labels.

      x1: screen brightness (%) ~ 20, 50, 80
      x2: CPU/GPU load (%)      ~ 20, 50, 80
    """
    ticks_std = np.array([-1.0, 0.0, 1.0])

    brightness_pct = 50 + 30 * ticks_std   # 20, 50, 80
    load_pct       = 50 + 30 * ticks_std   # 20, 50, 80

    return ticks_std, brightness_pct, ticks_std, load_pct


def decision_boundary_line(w, b, xlim):
    """
    For w=(w1,w2), line w1*x1 + w2*x2 + b = 0 in standardized coordinates.
    Returns (xs, ys) for plotting, or None if nearly vertical.
    """
    w1, w2 = w
    xs = np.linspace(xlim[0], xlim[1], 200)
    if abs(w2) < 1e-9:
        return None
    ys = -(w1 * xs + b) / w2
    return xs, ys


# -----------------------------
# Network forward pass
# -----------------------------
def forward(X, w11, w12, b1, w21, w22, b2, v1, v2, bout, out_activation):
    """
    Hidden layer: 2 sigmoid neurons
      z = W x + b
      h = sigmoid(z)

    Output:
      s = v1*h1 + v2*h2 + bout
      y = s (linear) OR y = sigmoid(s) (sigmoid)

    Classification uses s>0 in both cases (since sigmoid threshold 0.5 <-> s>0).
    """
    W1 = np.array([[w11, w12],
                   [w21, w22]], dtype=float)
    b = np.array([b1, b2], dtype=float)

    z = X @ W1.T + b          # (N,2)
    h = sigmoid(z)            # (N,2)
    s = v1 * h[:, 0] + v2 * h[:, 1] + bout  # (N,)

    if out_activation == "linear":
        yhat = s
        pred = (s > 0).astype(int)
    else:
        yhat = sigmoid(s)
        pred = (yhat > 0.5).astype(int)  # equivalent to s>0

    return z, h, s, yhat, pred


# -----------------------------
# Schematic drawing
# -----------------------------
def draw_nn_schematic(ax, out_activation, w11, w12, b1, w21, w22, b2, v1, v2, bout):
    ax.set_axis_off()
    ax.set_title("A) Network schematic (parameters)", pad=10)

    # Node positions (axes coordinates 0..1)
    x_in, x_h, x_out = 0.10, 0.55, 0.92
    y_x1, y_x2 = 0.72, 0.28
    y_h1, y_h2 = 0.72, 0.28
    y_out = 0.50

    def node(x, y, label, fc="white", fontsize=11):
        circ = patches.Circle((x, y), 0.09, facecolor=fc, edgecolor="black", linewidth=1.5)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=11)

    def arrow(x0, y0, x1, y1, label=None, dx=0.0, dy=0.02):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.4)
        )
        if label is not None:
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(xm + dx, ym + dy, label, ha="center", va="center", fontsize=10)

    # Nodes (still x1, x2 mathematically, but conceptually: brightness and load)
    node(x_in, y_x1, r"$x_1$")
    node(x_in, y_x2, r"$x_2$")
    node(x_h,  y_h1, r"$h_1=\sigma(\cdot)$", fc="#f6f6f6", fontsize=6)
    node(x_h,  y_h2, r"$h_2=\sigma(\cdot)$", fc="#f6f6f6", fontsize=6)
    node(x_out-0.05, y_out, r"$y$",  fc="#f6f6f6")

    # Input -> hidden (label components)
    arrow(x_in + 0.07, y_x1, x_h - 0.07, y_h1, label=rf"$w_{{11}}={w11:.2g}$")
    arrow(x_in + 0.07, y_x2, x_h - 0.07, y_h1, label=rf"$w_{{12}}={w12:.2g}$", dx=-0.03, dy=-0.15)

    arrow(x_in + 0.07, y_x1, x_h - 0.07, y_h2, label=rf"$w_{{21}}={w21:.2g}$", dx=-0.03, dy=0.15)
    arrow(x_in + 0.07, y_x2, x_h - 0.07, y_h2, label=rf"$w_{{22}}={w22:.2g}$", dy=-0.03)

    # Hidden biases
    ax.text(x_h, y_h1 + 0.13, rf"$b_1={b1:.2g}$", ha="center", va="center", fontsize=10)
    ax.text(x_h, y_h2 - 0.13, rf"$b_2={b2:.2g}$", ha="center", va="center", fontsize=10)

    # Hidden -> output
    arrow(x_h + 0.07, y_h1, x_out - 0.07, y_out + 0.10, label=rf"$v_1={v1:.2g}$", dy=0.04)
    arrow(x_h + 0.07, y_h2, x_out - 0.07, y_out - 0.10, label=rf"$v_2={v2:.2g}$", dy=0.04)

    # Output bias + formulas
    ax.text(x_out, y_out + 0.15, rf"$b_{{out}}={bout:.2g}$", ha="center", va="center", fontsize=10)

    ax.text(
        0.02, -0.12,
        r"Inputs: $x_1=$ brightness, $x_2=$ CPU/GPU load" "\n"
        r"Hidden: $h_j=\sigma(w_j\cdot x+b_j)$" "\n"
        r"Output pre-act: $s=v_1h_1+v_2h_2+b_{out}$" "\n"
        + (r"Output: $y=s$ (linear), classify by $s>0$"
           if out_activation == "linear"
           else r"Output: $y=\sigma(s)$, classify by $y>0.5\Leftrightarrow s>0$"),
        ha="left", va="bottom", fontsize=10
    )


# -----------------------------
# Main plotting function (interactive target)
# -----------------------------
def plot_all(
    seed=0,
    pattern="xor_overheat",
    out_activation="linear",
    w11=3.0, w12=3.0, b1=0.0,
    w21=3.0, w22=-3.0, b2=0.0,
    v1=6.0, v2=6.0, bout=-6.0,
    show_hidden_lines=True,
    show_activation_maps=True,
    legend_alpha=0.5
):
    # Data + model
    X, y = make_dataset(seed=seed, pattern=pattern)
    z, h, s, yhat, pred = forward(X, w11, w12, b1, w21, w22, b2, v1, v2, bout, out_activation)
    acc = (pred == y).mean()

    # Grid for contours in input space
    gx = np.linspace(-1.6, 1.6, 240)
    gy = np.linspace(-1.6, 1.6, 240)
    XX, YY = np.meshgrid(gx, gy)
    G = np.c_[XX.ravel(), YY.ravel()]
    zG, hG, sG, yG, pG = forward(G, w11, w12, b1, w21, w22, b2, v1, v2, bout, out_activation)

    # Tick relabeling (now brightness and load)
    x_ticks_std, bright_ticks_real, y_ticks_std, load_ticks_real = standardized_to_real_ticks()

    # -----------------------------
    # 2x2 Figure layout
    # -----------------------------
    fig = plt.figure(figsize=(12.5, 9))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.3],
        height_ratios=[1.0, 1.0],
        wspace=0.25,
        hspace=0.25
    )

    # A) Upper-left: schematic
    ax_schem = fig.add_subplot(gs[0, 0])
    draw_nn_schematic(ax_schem, out_activation, w11, w12, b1, w21, w22, b2, v1, v2, bout)

    # B) Upper-right: input space
    ax_in = fig.add_subplot(gs[0, 1])

    # Background predicted regions based on s>0
    region = (sG.reshape(YY.shape) > 0).astype(float)
    ax_in.contourf(XX, YY, region, levels=[-0.5, 0.5, 1.5], alpha=0.25)

    # Data points
    ax_in.scatter(X[y == 0, 0], X[y == 0, 1], marker="o", label="No overheat (0)")
    ax_in.scatter(X[y == 1, 0], X[y == 1, 1], marker="^", label="Overheat (1)")

    # Final decision boundary: s=0
    ax_in.contour(XX, YY, sG.reshape(YY.shape), levels=[0], linewidths=5)

    # Hidden z=0 lines
    if show_hidden_lines:
        ln1 = decision_boundary_line((w11, w12), b1, (-1.6, 1.6))
        ln2 = decision_boundary_line((w21, w22), b2, (-1.6, 1.6))

        if ln1 is not None:
            ax_in.plot(
                ln1[0], ln1[1],
                linestyle="--", linewidth=1.5,
                label=r"$z_1=0$ (hidden neuron $h_1$)"
            )

        if ln2 is not None:
            ax_in.plot(
                ln2[0], ln2[1],
                linestyle="--", linewidth=1.5,
                label=r"$z_2=0$ (hidden neuron $h_2$)"
            )

    ax_in.set_title(f"B) Input space (brightness vs load)   accuracy = {acc*100:.1f}%")
    ax_in.set_xlim(-1.6, 1.6)
    ax_in.set_ylim(-1.6, 1.6)
    ax_in.set_xticks(x_ticks_std, [f"{int(t)}%" for t in bright_ticks_real])
    ax_in.set_yticks(y_ticks_std, [f"{int(t)}%" for t in load_ticks_real])
    ax_in.set_xlabel("Screen brightness (approx %)")
    ax_in.set_ylabel("CPU/GPU load (approx %)")
    ax_in.legend(
        loc="upper left",
        fontsize=9,
        framealpha=float(legend_alpha)   # 1.0 = fully opaque
    )

    # C) Lower-left: hidden activation contours in input space
    ax_hid = fig.add_subplot(gs[1, 0])

    if show_activation_maps:
        # One neuron = one color; levels are distinguished by linestyle
        levels = [0.25, 0.50, 0.75]
        level_styles = {
            0.25: (0, (1, 2)),  # dotted
            0.50: "solid",      # solid
            0.75: "dashed"      # dashed
        }

        # Matplotlib default tab colors are stable and high-contrast
        neuron_colors = ["tab:blue", "tab:orange"]

        # h1 contours (all same color, different linestyles by level)
        H1 = hG[:, 0].reshape(YY.shape)
        for lev in levels:
            ax_hid.contour(
                XX, YY, H1,
                levels=[lev],
                linewidths=1.6 if lev == 0.50 else 1.2,
                colors=[neuron_colors[0]],
                linestyles=[level_styles[lev]]
            )

        # h2 contours (all same color, different linestyles by level)
        H2 = hG[:, 1].reshape(YY.shape)
        for lev in levels:
            ax_hid.contour(
                XX, YY, H2,
                levels=[lev],
                linewidths=1.6 if lev == 0.50 else 1.2,
                colors=[neuron_colors[1]],
                linestyles=[level_styles[lev]]
            )

        # Legend proxies (MATCH contour colors AND linestyles exactly)
        handles = [
            Line2D([0], [0], color=neuron_colors[0], lw=1.6,
                  linestyle=level_styles[0.25], label=r"$h_1 = 0.25$"),
            Line2D([0], [0], color=neuron_colors[0], lw=2.0,
                  linestyle=level_styles[0.50], label=r"$h_1 = 0.50$ ($z_1=0$)"),
            Line2D([0], [0], color=neuron_colors[0], lw=1.6,
                  linestyle=level_styles[0.75], label=r"$h_1 = 0.75$"),

            Line2D([0], [0], color=neuron_colors[1], lw=1.6,
                  linestyle=level_styles[0.25], label=r"$h_2 = 0.25$"),
            Line2D([0], [0], color=neuron_colors[1], lw=2.0,
                  linestyle=level_styles[0.50], label=r"$h_2 = 0.50$ ($z_2=0$)"),
            Line2D([0], [0], color=neuron_colors[1], lw=1.6,
                  linestyle=level_styles[0.75], label=r"$h_2 = 0.75$")
        ]

        ax_hid.legend(
            handles=handles,
            loc="upper left",
            fontsize=9,
            framealpha=float(legend_alpha)
        )


    # Data points on top
    ax_hid.scatter(X[y == 0, 0], X[y == 0, 1], marker="o")
    ax_hid.scatter(X[y == 1, 0], X[y == 1, 1], marker="^")

    ax_hid.set_title("C) Hidden neuron activation contours in input space")
    ax_hid.set_xlim(-1.6, 1.6)
    ax_hid.set_ylim(-1.6, 1.6)
    ax_hid.set_xticks(x_ticks_std, [f"{int(t)}%" for t in bright_ticks_real])
    ax_hid.set_yticks(y_ticks_std, [f"{int(t)}%" for t in load_ticks_real])
    ax_hid.set_xlabel("Screen brightness (approx %)")
    ax_hid.set_ylabel("CPU/GPU load (approx %)")

    # D) Lower-right: feature space (h1,h2)
    ax_feat = fig.add_subplot(gs[1, 1])

    ax_feat.scatter(h[y == 0, 0], h[y == 0, 1], marker="o", label="No overheat (0)")
    ax_feat.scatter(h[y == 1, 0], h[y == 1, 1], marker="^", label="Overheat (1)")

    # Output boundary in feature space: v1*h1 + v2*h2 + bout = 0
    h1s = np.linspace(0, 1, 200)
    if abs(v2) > 1e-9:
        h2s = -(v1 * h1s + bout) / v2
        ax_feat.plot(h1s, h2s, linewidth=2)
    elif abs(v1) > 1e-9:
        h1v = -bout / v1
        ax_feat.plot([h1v, h1v], [0, 1], linewidth=2)

    ax_feat.set_xlim(-0.1, 1.1)
    ax_feat.set_ylim(-0.1, 1.1)
    ax_feat.set_title("D) Feature space (h1,h2) with linear separator")
    ax_feat.set_xlabel("h1")
    ax_feat.set_ylabel("h2")
    ax_feat.legend(loc="upper left", fontsize=9, framealpha=float(legend_alpha))

    plt.show()


# -----------------------------
# Interactive controls
# -----------------------------
STYLE  = {'description_width': '150px'}   # label area
LAYOUT = {'width': '520px'}              # total slider width

# ---- Defaults (single source of truth) ----
DEFAULTS = dict(
    seed=0,
    pattern="xor_overheat",
    out_activation="linear",
    w11=3.0, w12=3.0, b1=0.0,
    w21=3.0, w22=-3.0, b2=0.0,
    v1=6.0, v2=6.0, bout=-6.0,
    show_hidden_lines=True,
    show_activation_maps=True,
    legend_alpha=0.5
)

# ---- Helper: slider with a reset button ----
def slider_with_reset(slider, default_value, button_width="40px"):
    btn = Button(
        description="↺",
        tooltip=f"Reset to default ({default_value:.2f})",
        layout=Layout(width=button_width)
    )
    def _reset(_):
        slider.value = default_value
    btn.on_click(_reset)
    return HBox([slider, btn])

# ---- Create widgets explicitly ----
seed = IntSlider(min=0, max=10, step=1, value=DEFAULTS["seed"],
                 description="data seed (set to 0)", style=STYLE, layout=LAYOUT)

pattern = Dropdown(
    options=[
        ("XOR overheating (exactly one of brightness/load is high)", "xor_overheat"),
        ("AND overheating (both brightness and load are high)", "and_overheat"),
        ("OR overheating (brightness or load is high)", "or_overheat"),
    ],
    value=DEFAULTS["pattern"],
    description="dataset",
    layout=Layout(width="720px")
)

out_activation = Dropdown(
    options=["linear", "sigmoid"],
    value=DEFAULTS["out_activation"],
    description="output act",
    layout=Layout(width="320px")
)

legend_alpha = FloatSlider(
    min=0.0, max=1.0, step=0.05, value=DEFAULTS["legend_alpha"],
    description="legend transparency", style=STYLE, layout=LAYOUT
)

# Sliders (note: NO trailing commas)
w11 = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["w11"],
                  description=f"w11 (def: {DEFAULTS['w11']:.2f})", style=STYLE, layout=LAYOUT)
w12 = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["w12"],
                  description=f"w12 (def: {DEFAULTS['w12']:.2f})", style=STYLE, layout=LAYOUT)
b1  = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["b1"],
                  description=f"b1 (def: {DEFAULTS['b1']:.2f})", style=STYLE, layout=LAYOUT)

w21 = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["w21"],
                  description=f"w21 (def: {DEFAULTS['w21']:.2f})", style=STYLE, layout=LAYOUT)
w22 = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["w22"],
                  description=f"w22 (def: {DEFAULTS['w22']:.2f})", style=STYLE, layout=LAYOUT)
b2  = FloatSlider(min=-10, max=10, step=0.25, value=DEFAULTS["b2"],
                  description=f"b2 (def: {DEFAULTS['b2']:.2f})", style=STYLE, layout=LAYOUT)

v1   = FloatSlider(min=-12, max=12, step=0.5, value=DEFAULTS["v1"],
                   description=f"v1 (def: {DEFAULTS['v1']:.2f})", style=STYLE, layout=LAYOUT)
v2   = FloatSlider(min=-12, max=12, step=0.5, value=DEFAULTS["v2"],
                   description=f"v2 (def: {DEFAULTS['v2']:.2f})", style=STYLE, layout=LAYOUT)
bout = FloatSlider(min=-12, max=12, step=0.5, value=DEFAULTS["bout"],
                   description=f"b_out (def: {DEFAULTS['bout']:.2f})", style=STYLE, layout=LAYOUT)

show_hidden_lines = Checkbox(value=DEFAULTS["show_hidden_lines"], description="show hidden z=0 lines")
show_activation_maps = Checkbox(value=DEFAULTS["show_activation_maps"], description="show hidden contours")

# ---- Wrap sliders with per-slider reset buttons ----
legend_alpha_box = slider_with_reset(legend_alpha, DEFAULTS["legend_alpha"])

w11_box  = slider_with_reset(w11,  DEFAULTS["w11"])
w12_box  = slider_with_reset(w12,  DEFAULTS["w12"])
b1_box   = slider_with_reset(b1,   DEFAULTS["b1"])
w21_box  = slider_with_reset(w21,  DEFAULTS["w21"])
w22_box  = slider_with_reset(w22,  DEFAULTS["w22"])
b2_box   = slider_with_reset(b2,   DEFAULTS["b2"])
v1_box   = slider_with_reset(v1,   DEFAULTS["v1"])
v2_box   = slider_with_reset(v2,   DEFAULTS["v2"])
bout_box = slider_with_reset(bout, DEFAULTS["bout"])

# ---- Optional: one-click reset all ----
reset_all = Button(description="Reset ALL to defaults", tooltip="Reset every parameter to its default value",
                   layout=Layout(width="220px"))

def _reset_all(_):
    seed.value = DEFAULTS["seed"]
    pattern.value = DEFAULTS["pattern"]
    out_activation.value = DEFAULTS["out_activation"]
    legend_alpha.value = DEFAULTS["legend_alpha"]
    w11.value = DEFAULTS["w11"]; w12.value = DEFAULTS["w12"]; b1.value = DEFAULTS["b1"]
    w21.value = DEFAULTS["w21"]; w22.value = DEFAULTS["w22"]; b2.value = DEFAULTS["b2"]
    v1.value = DEFAULTS["v1"]; v2.value = DEFAULTS["v2"]; bout.value = DEFAULTS["bout"]
    show_hidden_lines.value = DEFAULTS["show_hidden_lines"]
    show_activation_maps.value = DEFAULTS["show_activation_maps"]

reset_all.on_click(_reset_all)

# ---- Build the UI layout ----
controls = VBox([
    HBox([seed, reset_all]),
    pattern,
    HBox([out_activation, show_hidden_lines, show_activation_maps]),
    legend_alpha_box,
    w11_box, w12_box, b1_box,
    w21_box, w22_box, b2_box,
    v1_box, v2_box, bout_box
])

# ---- Connect widgets to plotting function ----
out = interactive_output(
    plot_all,
    dict(
        seed=seed,
        pattern=pattern,
        out_activation=out_activation,
        w11=w11, w12=w12, b1=b1,
        w21=w21, w22=w22, b2=b2,
        v1=v1, v2=v2, bout=bout,
        show_hidden_lines=show_hidden_lines,
        show_activation_maps=show_activation_maps,
        legend_alpha=legend_alpha
    )
)

display(controls, out)
