# ================================================================
# Copyright Carl Bernard
# ENGRI Spring 2026
# HW4 INTERACTIVE MODULATION CLASSIFIER (Colab single-cell)
# Engineering-flavored demo: feature-based "signal type" classification
# using an MLP + Softmax + Cross-Entropy.
# Controls:
# - Per slider: ▲ ▼ reset
# - Global reset button
# - RUN / RE-RUN button
# ================================================================

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import VBox, HBox
from IPython.display import display


# ================================================================
# DEFAULTS (tuned to "work out of the box")
# ================================================================
defaults = {
    # data
    "n_samples": 1200,
    "train_ratio": 0.70,
    "seed": 0,

    # signal / feature model
    "n_classes": 5,
    "feat_dim": 20,
    "snr_db_train": 12.0,
    "snr_db_test": 12.0,
    "activation": "relu",
    "standardize": True,

    # NN / training
    "depth": 6,
    "width": 64,
    "lr": 0.10,
    "epochs": 60,
    "batch_size": 128,

    # regularization
    "l2": 0.0,
    "l1": 0.0,
    "dropout_p": 0.0,
    "augment_noise": 0.0,
}


# ================================================================
# ROBUST SLIDER SETTER (fixes ipywidgets "reset sometimes does nothing")
# ================================================================
def force_set_value(widget, target):
    if isinstance(widget, widgets.FloatSlider):
        target = float(target)
        target = min(widget.max, max(widget.min, target))
        eps = widget.step if widget.step else 1e-6
        if abs(widget.value - target) < 1e-12:
            nudge = target + eps if (target + eps) <= widget.max else target - eps
            widget.value = nudge
        widget.value = target

    elif isinstance(widget, widgets.IntSlider):
        target = int(target)
        target = min(widget.max, max(widget.min, target))
        if widget.value == target:
            nudge = target + 1 if (target + 1) <= widget.max else target - 1
            widget.value = nudge
        widget.value = target

    else:
        widget.value = target


# ================================================================
# UI CONTROL FACTORIES: slider + ▲ ▼ + reset
# ================================================================
def make_float_control(label, minv, maxv, slider_step, button_step, default,
                       width_px=720, desc_w=260):
    s = widgets.FloatSlider(
        description=f"{label}",
        min=minv, max=maxv, step=slider_step, value=default,
        continuous_update=False,
        style={'description_width': f'{desc_w}px'},
        layout=widgets.Layout(width=f'{width_px}px')
    )
    up_btn = widgets.Button(description="▲", layout=widgets.Layout(width="35px"))
    dn_btn = widgets.Button(description="▼", layout=widgets.Layout(width="35px"))
    rs_btn = widgets.Button(description="reset", layout=widgets.Layout(width="70px"))

    def do_up(_): force_set_value(s, round(s.value + button_step, 10))
    def do_dn(_): force_set_value(s, round(s.value - button_step, 10))
    def do_rs(_): force_set_value(s, default)

    up_btn.on_click(do_up)
    dn_btn.on_click(do_dn)
    rs_btn.on_click(do_rs)

    return HBox([s, up_btn, dn_btn, rs_btn]), s


def make_int_control(label, minv, maxv, default, button_step=1,
                     width_px=720, desc_w=260):
    s = widgets.IntSlider(
        description=f"{label}",
        min=minv, max=maxv, step=1, value=default,
        continuous_update=False,
        style={'description_width': f'{desc_w}px'},
        layout=widgets.Layout(width=f'{width_px}px')
    )
    up_btn = widgets.Button(description="▲", layout=widgets.Layout(width="35px"))
    dn_btn = widgets.Button(description="▼", layout=widgets.Layout(width="35px"))
    rs_btn = widgets.Button(description="reset", layout=widgets.Layout(width="70px"))

    def do_up(_): force_set_value(s, s.value + button_step)
    def do_dn(_): force_set_value(s, s.value - button_step)
    def do_rs(_): force_set_value(s, default)

    up_btn.on_click(do_up)
    dn_btn.on_click(do_dn)
    rs_btn.on_click(do_rs)

    return HBox([s, up_btn, dn_btn, rs_btn]), s


# ================================================================
# ACTIVATIONS + LOSSES
# ================================================================
def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def dsigmoid_from_a(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def drelu(z):
    return (z > 0.0).astype(z.dtype)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    denom = np.sum(ez, axis=1, keepdims=True)
    return ez / (denom + 1e-12)

def one_hot(y, K):
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1.0
    return Y

def ce_loss(p, y_onehot):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.sum(y_onehot * np.log(p), axis=1)

def accuracy_multiclass(p, y):
    return float(np.mean(np.argmax(p, axis=1) == y))


# ================================================================
# STANDARDIZATION
# ================================================================
def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd


# ================================================================
# MODULATION FEATURE DATASET (synthetic but engineering-plausible)
# ================================================================
def snr_db_to_sigma(snr_db):
    return 10.0 ** (-snr_db / 20.0)

def make_modulation_feature_dataset(n, K, d, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.zeros((K, d), dtype=float)

    gA = np.arange(0, d//4)
    gB = np.arange(d//4, d//2)
    gC = np.arange(d//2, (3*d)//4)
    gD = np.arange((3*d)//4, d)

    for k in range(K):
        archetype = k % 3
        base = rng.normal(0, 0.5, size=d)

        if archetype == 0:
            base[gA] += rng.normal(-1.0, 0.35, size=gA.size)
            base[gB] += rng.normal(+1.2, 0.35, size=gB.size)
            base[gC] += rng.normal(0.0, 0.25, size=gC.size)
            base[gD] += rng.normal(+0.6, 0.30, size=gD.size)
        elif archetype == 1:
            base[gA] += rng.normal(+1.3, 0.45, size=gA.size)
            base[gB] += rng.normal(+0.5, 0.30, size=gB.size)
            base[gC] += rng.normal(0.0, 0.25, size=gC.size)
            base[gD] += rng.normal(+0.3, 0.30, size=gD.size)
        else:
            base[gA] += rng.normal(0.0, 0.25, size=gA.size)
            base[gB] += rng.normal(0.0, 0.25, size=gB.size)
            base[gC] += rng.normal(+1.4, 0.45, size=gC.size)
            base[gD] += rng.normal(+0.5, 0.30, size=gD.size)

        base += (k - (K - 1)/2) * 0.10
        mu[k] = base

    A = rng.normal(0, 1.0, size=(d, d))
    Cov = (A @ A.T) / d
    Cov = Cov / (np.max(np.diag(Cov)) + 1e-12)
    Cov = 0.30 * Cov + 0.70 * np.eye(d)

    sigma = snr_db_to_sigma(snr_db)
    L = np.linalg.cholesky(Cov)

    y = rng.integers(0, K, size=n)
    X = np.zeros((n, d), dtype=float)

    for i in range(n):
        k = y[i]
        z = rng.normal(0, 1.0, size=d)
        noise = (L @ z) * sigma
        X[i] = mu[k] + noise

    return X, y


def train_test_split(X, y, train_ratio=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    ntr = int(train_ratio * X.shape[0])
    tr, te = idx[:ntr], idx[ntr:]
    return X[tr], y[tr], X[te], y[te]


# ================================================================
# MLP (NUMPY)
# ================================================================
def init_params(layer_sizes, init_scale="he", seed=0):
    rng = np.random.default_rng(seed)
    W, b = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        scale = np.sqrt((2.0 if init_scale == "he" else 1.0) / fan_in)
        W.append(rng.normal(0, scale, size=(fan_in, layer_sizes[i + 1])))
        b.append(np.zeros((1, layer_sizes[i + 1])))
    return W, b


def forward(X, W, b, activation="relu", dropout_p=0.0, train=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    A = X
    cache = []
    L = len(W)

    for ell in range(L - 1):
        Z = A @ W[ell] + b[ell]
        A_next = sigmoid(Z) if activation == "sigmoid" else relu(Z)

        mask = None
        if train and dropout_p > 0.0:
            keep = 1.0 - dropout_p
            mask = (rng.random(A_next.shape) < keep).astype(A_next.dtype) / keep
            A_next *= mask

        cache.append({"A_prev": A, "Z": Z, "A": A_next, "mask": mask})
        A = A_next

    ZL = A @ W[-1] + b[-1]
    cache.append({"A_prev": A, "Z": ZL})
    return ZL, cache


def backward_multiclass(X, y, W, b, cache, K, activation="relu", l2=0.0, l1=0.0):
    grads_W = [np.zeros_like(Wi) for Wi in W]
    grads_b = [np.zeros_like(bi) for bi in b]

    ZL = cache[-1]["Z"]
    p = softmax(ZL)
    Y = one_hot(y, K)

    # delta_out for softmax + cross-entropy
    dZ = (p - Y)

    A_prev = cache[-1]["A_prev"]
    grads_W[-1] = (A_prev.T @ dZ) / X.shape[0]
    grads_b[-1] = np.mean(dZ, axis=0, keepdims=True)

    dA = dZ @ W[-1].T

    for ell in reversed(range(len(W) - 1)):
        layer = cache[ell]
        if layer["mask"] is not None:
            dA *= layer["mask"]

        if activation == "sigmoid":
            dZ = dA * dsigmoid_from_a(layer["A"])
        else:
            dZ = dA * drelu(layer["Z"])

        A_prev = layer["A_prev"]
        grads_W[ell] = (A_prev.T @ dZ) / X.shape[0]
        grads_b[ell] = np.mean(dZ, axis=0, keepdims=True)
        dA = dZ @ W[ell].T

    for i in range(len(W)):
        if l2 > 0.0:
            grads_W[i] += l2 * W[i]
        if l1 > 0.0:
            grads_W[i] += l1 * np.sign(W[i])

    return grads_W, grads_b


def grad_norms(grads_W):
    return np.array([np.linalg.norm(g) for g in grads_W], dtype=float)


# ================================================================
# TRAIN + PLOT
# ================================================================
def train_demo(cfg):
    rng = np.random.default_rng(cfg["seed"])

    X_all, y_all = make_modulation_feature_dataset(
        n=cfg["n_samples"], K=cfg["n_classes"], d=cfg["feat_dim"],
        snr_db=cfg["snr_db_train"], seed=cfg["seed"]
    )

    Xtr, ytr, Xte, yte = train_test_split(
        X_all, y_all, train_ratio=cfg["train_ratio"], seed=cfg["seed"] + 1
    )

    # Optional test SNR mismatch: if test is noisier than train, add extra noise to test
    if abs(cfg["snr_db_test"] - cfg["snr_db_train"]) > 1e-9:
        sigma_train = snr_db_to_sigma(cfg["snr_db_train"])
        sigma_test = snr_db_to_sigma(cfg["snr_db_test"])
        if sigma_test > sigma_train:
            extra = np.sqrt(max(0.0, sigma_test**2 - sigma_train**2))
            Xte = Xte + rng.normal(0, extra, size=Xte.shape)

    if cfg["standardize"]:
        mu, sd = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, mu, sd)
        Xte = standardize_apply(Xte, mu, sd)

    D = Xtr.shape[1]
    K = cfg["n_classes"]

    layer_sizes = [D] + [cfg["width"]] * cfg["depth"] + [K]
    init_scale = "he" if cfg["activation"] == "relu" else "xavier"
    W, b = init_params(layer_sizes, init_scale=init_scale, seed=cfg["seed"] + 2)

    tr_loss_hist, te_loss_hist = [], []
    tr_acc_hist, te_acc_hist = [], []
    grad_hist = []

    diverged = False
    reason = ""

    for ep in range(cfg["epochs"]):
        idx = rng.permutation(Xtr.shape[0])
        Xtr_sh, ytr_sh = Xtr[idx], ytr[idx]

        for start in range(0, Xtr_sh.shape[0], cfg["batch_size"]):
            xb = Xtr_sh[start:start + cfg["batch_size"]]
            yb = ytr_sh[start:start + cfg["batch_size"]]

            if cfg["augment_noise"] > 0.0:
                xb = xb + rng.normal(0, cfg["augment_noise"], size=xb.shape)

            ZL, cache = forward(
                xb, W, b,
                activation=cfg["activation"],
                dropout_p=cfg["dropout_p"],
                train=True,
                rng=rng
            )

            grads_W, grads_b = backward_multiclass(
                xb, yb, W, b, cache, K,
                activation=cfg["activation"],
                l2=cfg["l2"], l1=cfg["l1"]
            )

            # NaN checks
            if any((~np.isfinite(g)).any() for g in grads_W):
                diverged = True
                reason = "NaNs in gradients (try smaller learning rate)."
                break

            for i in range(len(W)):
                W[i] -= cfg["lr"] * grads_W[i]
                b[i] -= cfg["lr"] * grads_b[i]

            if any((~np.isfinite(Wi)).any() for Wi in W):
                diverged = True
                reason = "NaNs in weights (try smaller learning rate)."
                break

        if diverged:
            break

        # evaluation
        Ztr, _ = forward(Xtr, W, b, activation=cfg["activation"], dropout_p=0.0, train=False, rng=rng)
        Zte, _ = forward(Xte, W, b, activation=cfg["activation"], dropout_p=0.0, train=False, rng=rng)
        ptr = softmax(Ztr)
        pte = softmax(Zte)

        Ytr = one_hot(ytr, K)
        Yte = one_hot(yte, K)

        tr_loss = float(np.mean(ce_loss(ptr, Ytr)))
        te_loss = float(np.mean(ce_loss(pte, Yte)))
        tr_acc = accuracy_multiclass(ptr, ytr)
        te_acc = accuracy_multiclass(pte, yte)

        if not (np.isfinite(tr_loss) and np.isfinite(te_loss) and np.isfinite(tr_acc) and np.isfinite(te_acc)):
            diverged = True
            reason = "Non-finite loss/accuracy (try smaller learning rate)."
            break

        # gradient snapshot
        xb = Xtr[:min(256, Xtr.shape[0])]
        yb = ytr[:min(256, Xtr.shape[0])]
        ZL, cache = forward(xb, W, b, activation=cfg["activation"], dropout_p=0.0, train=True, rng=rng)
        grads_W, grads_b = backward_multiclass(xb, yb, W, b, cache, K, activation=cfg["activation"], l2=0.0, l1=0.0)
        grad_hist.append(grad_norms(grads_W))

        tr_loss_hist.append(tr_loss)
        te_loss_hist.append(te_loss)
        tr_acc_hist.append(tr_acc)
        te_acc_hist.append(te_acc)

    if len(tr_loss_hist) == 0:
        tr_loss_hist = [np.nan]
        te_loss_hist = [np.nan]
        tr_acc_hist = [np.nan]
        te_acc_hist = [np.nan]
        grad_hist = [np.full(len(W), np.nan)]

    return {
        "cfg": cfg,
        "tr_loss": np.array(tr_loss_hist),
        "te_loss": np.array(te_loss_hist),
        "tr_acc": np.array(tr_acc_hist),
        "te_acc": np.array(te_acc_hist),
        "grad_hist": np.vstack(grad_hist),
        "layer_sizes": layer_sizes,
        "diverged": diverged,
        "diverged_reason": reason
    }


def _cfg_bullets(cfg, layer_sizes=None):
    # Compact bullet list for students; keep it readable.
    lines = []
    lines.append("Current settings")
    lines.append("")
    lines.append(f"• Training dataset size: {cfg['n_samples']}")
    lines.append(f"• Training fraction: {cfg['train_ratio']:.2f}")
    lines.append(f"• Random seed: {cfg['seed']}")
    lines.append("")
    lines.append(f"• Signal types (classes): {cfg['n_classes']}")
    lines.append(f"• Input features: {cfg['feat_dim']}")
    lines.append(f"• Train SNR (dB): {cfg['snr_db_train']:.1f}")
    lines.append(f"• Test SNR (dB): {cfg['snr_db_test']:.1f}")
    lines.append(f"• Activation: {cfg['activation']}")
    lines.append(f"• Normalize features: {cfg['standardize']}")
    lines.append("")
    lines.append(f"• Hidden layers: {cfg['depth']}")
    lines.append(f"• Neurons/layer: {cfg['width']}")
    lines.append(f"• Learning rate: {cfg['lr']:.3f}")
    lines.append(f"• Epochs: {cfg['epochs']}")
    lines.append(f"• Mini-batch size: {cfg['batch_size']}")
    lines.append("")
    lines.append(f"• L2 penalty: {cfg['l2']:.3f}")
    lines.append(f"• L1 penalty: {cfg['l1']:.3f}")
    lines.append(f"• Dropout prob: {cfg['dropout_p']:.2f}")
    lines.append(f"• Train feature noise: {cfg['augment_noise']:.2f}")
    if layer_sizes is not None:
        lines.append(f"• Layer sizes: {list(layer_sizes)}")
    return "\n".join(lines)


def plot_results(res):
    cfg = res["cfg"]
    tr_loss, te_loss = res["tr_loss"], res["te_loss"]
    tr_acc, te_acc = res["tr_acc"], res["te_acc"]
    G = res["grad_hist"]
    layer_sizes = res.get("layer_sizes", None)

    epochs = len(tr_loss)
    x_ep = np.arange(1, epochs + 1)
    layers = G.shape[1]

    # ---- Loss figure with right-side settings panel ----
    fig = plt.figure(figsize=(11.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.8], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    axinfo = fig.add_subplot(gs[0, 1])
    axinfo.axis("off")

    ax.plot(x_ep, tr_loss, linewidth=2, marker="o", markersize=3, label="train loss")
    ax.plot(x_ep, te_loss, linewidth=2, marker="o", markersize=3, label="test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss vs Epoch")
    ax.grid(True)
    ax.legend()

    # Put the settings text on the right (now includes Layer sizes bullet)
    axinfo.text(
        0.0, 1.0,
        _cfg_bullets(cfg, layer_sizes=layer_sizes),
        va="top", ha="left",
        fontsize=10,
        family="monospace"
    )

    if res.get("diverged", False):
        ax.text(
            0.02, 0.95,
            "DIVERGED (try smaller learning rate)",
            transform=ax.transAxes,
            color="red",
            fontsize=12,
            va="top"
        )

    plt.show()

    # ---- Accuracy ----
    plt.figure(figsize=(7, 4))
    plt.plot(x_ep, tr_acc, linewidth=2, marker="o", markersize=3, label="train accuracy")
    plt.plot(x_ep, te_acc, linewidth=2, marker="o", markersize=3, label="test accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs Epoch")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---- Gradient norm vs layer ----
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, layers + 1), G[-1], marker="o", linewidth=2)
    plt.xlabel("layer index (1 = first weight layer)")
    plt.ylabel("||dC/dW||")
    plt.title("Gradient norm vs Layer (last epoch)")
    plt.grid(True)
    plt.show()

    # ---- Gradient norm over epochs (early vs last) ----
    plt.figure(figsize=(7, 4))
    plt.plot(x_ep, G[:, 0], linewidth=2, label="layer 1 grad norm")
    plt.plot(x_ep, G[:, -1], linewidth=2, label=f"last layer grad norm (layer {layers})")
    plt.xlabel("epoch")
    plt.ylabel("||dC/dW||")
    plt.title("Gradient norm over Epochs")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Removed: separate printout of Layer sizes (now shown in right-side panel)
    if res.get("diverged", False):
        print("\nWARNING: Training stopped early.")
        if res.get("diverged_reason"):
            print("Reason:", res["diverged_reason"])


# ================================================================
# WIDGETS
# ================================================================
act_dd = widgets.Dropdown(
    options=[("ReLU", "relu"), ("Sigmoid", "sigmoid")],
    value=defaults["activation"],
    description="Neuron activation function"
)
std_cb = widgets.Checkbox(value=defaults["standardize"], description="Standardize input features")

# Data controls
n_box, n_sl = make_int_control("Training dataset size (examples)", 300, 5000, default=defaults["n_samples"], button_step=100)
tr_box, tr_sl = make_float_control("Fraction used for training", 0.30, 0.90, slider_step=0.05, button_step=0.05, default=defaults["train_ratio"])
seed_box, seed_sl = make_int_control("Random seed (dataset)", 0, 50, default=defaults["seed"], button_step=1)

# Signal / feature controls
K_box, K_sl = make_int_control("Number of signal types (classes)", 3, 10, default=defaults["n_classes"], button_step=1)
d_box, d_sl = make_int_control("Number of input features", 10, 30, default=defaults["feat_dim"], button_step=1)

snrtr_box, snrtr_sl = make_float_control("Training signal SNR (dB)", -5.0, 30.0, slider_step=1.0, button_step=1.0, default=defaults["snr_db_train"])
snrte_box, snrte_sl = make_float_control("Test signal SNR (dB)", -5.0, 30.0, slider_step=1.0, button_step=1.0, default=defaults["snr_db_test"])

# NN + training
depth_box, depth_sl = make_int_control("Number of hidden layers", 1, 30, default=defaults["depth"], button_step=1)
width_box, width_sl = make_int_control("Neurons per hidden layer", 4, 256, default=defaults["width"], button_step=4)
lr_box, lr_sl = make_float_control("Learning rate", 0.001, 1.0, slider_step=0.01, button_step=0.01, default=defaults["lr"])
epochs_box, epochs_sl = make_int_control("Training epochs", 10, 300, default=defaults["epochs"], button_step=10)
batch_box, batch_sl = make_int_control("Mini-batch size", 16, 512, default=defaults["batch_size"], button_step=16)

# Regularization
l2_box, l2_sl = make_float_control("L2 weight penalty (weight decay)", 0.0, 0.05, slider_step=0.001, button_step=0.001, default=defaults["l2"])
l1_box, l1_sl = make_float_control("L1 weight penalty (sparsity)", 0.0, 0.05, slider_step=0.001, button_step=0.001, default=defaults["l1"])
drop_box, drop_sl = make_float_control("Dropout probability", 0.0, 0.8, slider_step=0.05, button_step=0.05, default=defaults["dropout_p"])
aug_box, aug_sl = make_float_control("Noise added to training features", 0.0, 0.8, slider_step=0.05, button_step=0.05, default=defaults["augment_noise"])

run_btn = widgets.Button(description="RUN / RE-RUN", button_style="success")
global_reset_btn = widgets.Button(description="GLOBAL RESET", button_style="danger")
out = widgets.Output()


def get_cfg_from_widgets():
    return {
        "n_samples": int(n_sl.value),
        "train_ratio": float(tr_sl.value),
        "seed": int(seed_sl.value),

        "n_classes": int(K_sl.value),
        "feat_dim": int(d_sl.value),

        "snr_db_train": float(snrtr_sl.value),
        "snr_db_test": float(snrte_sl.value),

        "activation": act_dd.value,
        "standardize": bool(std_cb.value),

        "depth": int(depth_sl.value),
        "width": int(width_sl.value),
        "lr": float(lr_sl.value),
        "epochs": int(epochs_sl.value),
        "batch_size": int(batch_sl.value),

        "l2": float(l2_sl.value),
        "l1": float(l1_sl.value),
        "dropout_p": float(drop_sl.value),
        "augment_noise": float(aug_sl.value),
    }


def do_run(_=None):
    with out:
        out.clear_output(wait=True)
        cfg = get_cfg_from_widgets()
        res = train_demo(cfg)
        plot_results(res)


def do_global_reset(_=None):
    force_set_value(act_dd, defaults["activation"])
    force_set_value(std_cb, defaults["standardize"])

    force_set_value(n_sl, defaults["n_samples"])
    force_set_value(tr_sl, defaults["train_ratio"])
    force_set_value(seed_sl, defaults["seed"])

    force_set_value(K_sl, defaults["n_classes"])
    force_set_value(d_sl, defaults["feat_dim"])

    force_set_value(snrtr_sl, defaults["snr_db_train"])
    force_set_value(snrte_sl, defaults["snr_db_test"])

    force_set_value(depth_sl, defaults["depth"])
    force_set_value(width_sl, defaults["width"])
    force_set_value(lr_sl, defaults["lr"])
    force_set_value(epochs_sl, defaults["epochs"])
    force_set_value(batch_sl, defaults["batch_size"])

    force_set_value(l2_sl, defaults["l2"])
    force_set_value(l1_sl, defaults["l1"])
    force_set_value(drop_sl, defaults["dropout_p"])
    force_set_value(aug_sl, defaults["augment_noise"])


run_btn.on_click(do_run)
global_reset_btn.on_click(do_global_reset)

# ================================================================
# DISPLAY UI
# ================================================================
display(VBox([
    widgets.HTML("<h3>HW4 Modulation Classification Lab (Softmax + Cross-Entropy)</h3>"),
    HBox([global_reset_btn, run_btn]),

    widgets.HTML("<hr><b>Data</b>"),
    n_box,
    tr_box,
    seed_box,

    widgets.HTML("<hr><b>Signal / Feature Model</b>"),
    K_box,
    d_box,
    snrtr_box,
    snrte_box,
    HBox([act_dd, std_cb]),

    widgets.HTML("<hr><b>Neural Network + Training</b>"),
    depth_box,
    width_box,
    lr_box,
    epochs_box,
    batch_box,

    widgets.HTML("<hr><b>Regularization</b>"),
    l2_box,
    l1_box,
    drop_box,
    aug_box,

    widgets.HTML("<hr>"),
    out
]))

# Initial run
do_run()
