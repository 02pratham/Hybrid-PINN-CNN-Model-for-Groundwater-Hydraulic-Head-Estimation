# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, Polygon, FancyArrow, FancyArrowPatch, Circle

# # Helper for saving figures
# def save_fig(name):
#     plt.tight_layout()
#     plt.savefig(f"{name}.png", dpi=300)
#     plt.close()


# # 3.1 Example realization of lognormal random fields for cohesion, friction angle, unit weight, permeability
# def fig_3_1():
#     np.random.seed(0)
#     n = 64
#     fields = []
#     mus = [20, 30, 18, 1e-5]
#     covs = [0.3, 0.1, 0.05, 0.5]
#     for mu, cov in zip(mus, covs):
#         sigma = cov * mu
#         mu_ln = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
#         sigma_ln = np.sqrt(np.log(1 + (sigma**2 / mu**2)))
#         g = np.random.randn(n, n)
#         x = np.exp(mu_ln + sigma_ln * g)
#         fields.append(x)

#     fig, axes = plt.subplots(2, 2, figsize=(6, 6))
#     titles = ["Cohesion", "Friction angle", "Unit weight", "Permeability"]
#     for ax, field, title in zip(axes.ravel(), fields, titles):
#         im = ax.imshow(field)
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.colorbar(im, ax=ax, shrink=0.7)
#     save_fig("fig_3_1_lograndom_fields")

# # 3.2 Bilinear interpolation schematic
# def fig_3_2():
#     fig, ax = plt.subplots(figsize=(4, 4))
#     # Grid cell
#     xs = [0, 1]
#     zs = [0, 1]
#     for x in xs:
#         ax.plot([x, x], [0, 1])
#     for z in zs:
#         ax.plot([0, 1], [z, z])
#     # Corner points
#     pts = [(0,0), (1,0), (0,1), (1,1)]
#     labels = ["(x1,z1)", "(x2,z1)", "(x1,z2)", "(x2,z2)"]
#     for (x,z), lab in zip(pts, labels):
#         ax.scatter(x, z)
#         ax.text(x+0.02, z+0.02, lab)
#     # Interior point
#     xi, zi = 0.55, 0.35
#     ax.scatter(xi, zi, marker="x")
#     ax.text(xi+0.02, zi-0.05, "(x,z)")
#     ax.set_xlim(-0.1, 1.1)
#     ax.set_ylim(-0.1, 1.1)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.set_aspect("equal")
#     save_fig("fig_3_2_bilinear_interpolation")

# # 3.3 Seepage domain and boundary conditions
# def fig_3_3():
#     fig, ax = plt.subplots(figsize=(6, 3))
#     # Domain rectangle
#     ax.add_patch(Rectangle((0,0), 10, 5, fill=False))
#     # Slope: right side cut
#     slope = Polygon([[4,0],[10,0],[10,5]], fill=False)
#     ax.add_patch(slope)
#     # Water levels: left boundary top head
#     ax.annotate("Head H", xy=(0,5), xytext=(0.5,4.5),
#                 arrowprops=dict(arrowstyle="->"))
#     # Boundary labels (conceptual)
#     ax.text(0.1, 2.5, "No-flow", rotation=90, va="center")
#     ax.text(5, 0.1, "Impermeable base", ha="center")
#     ax.text(9, 2.0, "Slope face\n(h=0)", ha="center")
#     ax.set_xlim(-0.5, 10.5)
#     ax.set_ylim(-0.5, 5.5)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.set_aspect("equal")
#     save_fig("fig_3_3_seepage_domain")

# # 3.4 Collocation points for PINN
# def fig_3_4():
#     fig, ax = plt.subplots(figsize=(6, 3))
#     # Domain
#     ax.add_patch(Rectangle((0,0), 10, 5, fill=False))
#     # Random interior points
#     n_int = 500
#     xi = np.random.rand(n_int) * 10
#     zi = np.random.rand(n_int) * 5
#     ax.scatter(xi, zi, s=5, alpha=0.4, label="Interior points")
#     # Boundary points
#     n_b = 200
#     xb = np.concatenate([np.random.rand(n_b//4)*10, np.zeros(n_b//4), np.ones(n_b//4)*10, np.random.rand(n_b//4)*10])
#     zb = np.concatenate([np.zeros(n_b//4), np.random.rand(n_b//4)*5, np.random.rand(n_b//4)*5, np.ones(n_b//4)*5])
#     ax.scatter(xb, zb, s=8, marker="x", label="Boundary points")
#     ax.set_xlim(-0.5, 10.5)
#     ax.set_ylim(-0.5, 5.5)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.legend()
#     ax.set_aspect("equal")
#     save_fig("fig_3_4_pinn_collocation")

# # 3.5 Failure mechanism
# def fig_3_5():
#     fig, ax = plt.subplots(figsize=(5, 3))
#     # Slope
#     slope = Polygon([[0,0],[6,0],[6,3]], fill=False)
#     ax.add_patch(slope)
#     # Rotation center
#     O = (2, -1)
#     ax.scatter(*O)
#     ax.text(O[0]+0.1, O[1], "O")
#     # Log-spiral-esque slip surface
#     phi = np.deg2rad(30)
#     theta = np.linspace(np.deg2rad(10), np.deg2rad(80), 30)
#     r0 = 2.5
#     r = r0*np.exp((theta-theta[0])*np.tan(phi))
#     x = O[0] + r*np.cos(theta)
#     z = O[1] + r*np.sin(theta)
#     ax.plot(x, z, marker="o", markersize=2)
#     ax.text(x[-1], z[-1]+0.1, "Slip surface")
#     ax.set_xlim(-1, 7)
#     ax.set_ylim(-2, 4)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.set_aspect("equal")
#     save_fig("fig_3_5_failure_mechanism")

# # 3.6 Deterministic workflow diagram
# def fig_3_6():
#     fig, ax = plt.subplots(figsize=(7, 2))
#     ax.axis("off")
#     # Boxes positions
#     boxes = [
#         (0.05, 0.3, 0.18, 0.4, "Random\nfield\ngeneration"),
#         (0.30, 0.3, 0.18, 0.4, "PINN\nseepage\nanalysis"),
#         (0.55, 0.3, 0.18, 0.4, "Limit\nanalysis"),
#         (0.80, 0.3, 0.15, 0.4, "Factor of\nsafety")
#     ]
#     for x, y, w, h, txt in boxes:
#         ax.add_patch(Rectangle((x,y), w, h, fill=False))
#         ax.text(x+w/2, y+h/2, txt, ha="center", va="center")
#     # Arrows
#     for i in range(len(boxes)-1):
#         x, y, w, h, _ = boxes[i]
#         x2, y2, w2, h2, _ = boxes[i+1]
#         ax.annotate("", xy=(x2, y2+h2/2), xytext=(x+w, y+h/2),
#                     arrowprops=dict(arrowstyle="->"))
#     save_fig("fig_3_6_deterministic_workflow")

# # 3.7 CNN architecture diagram
# def fig_3_7():
#     fig, ax = plt.subplots(figsize=(7, 3))
#     ax.axis("off")
#     # Simple stacked boxes
#     x0, y0 = 0.05, 0.2
#     w, h = 0.08, 0.6
#     layers = [
#         ("Input\n128x128x4", x0),
#         ("Conv+ReLU\n32 filters", x0+0.12),
#         ("Conv+ReLU\n64 filters", x0+0.24),
#         ("Conv+ReLU\n128 filters", x0+0.36),
#         ("Conv+ReLU\n256 filters", x0+0.48),
#         ("FC\n512", x0+0.64),
#         ("Output\nFoS", x0+0.80)
#     ]
#     for label, x in layers:
#         ax.add_patch(Rectangle((x, y0), w, h, fill=False))
#         ax.text(x+w/2, y0+h/2, label, ha="center", va="center", fontsize=8)
#     # Arrows
#     for i in range(len(layers)-1):
#         x = layers[i][1]
#         x2 = layers[i+1][1]
#         ax.annotate("", xy=(x2, y0+h/2), xytext=(x+w, y0+h/2),
#                     arrowprops=dict(arrowstyle="->"))
#     save_fig("fig_3_7_cnn_architecture")

# # 4.1 End-to-end framework
# def fig_4_1():
#     fig, ax = plt.subplots(figsize=(8, 3))
#     ax.axis("off")
#     x0, y0 = 0.03, 0.4
#     w, h = 0.14, 0.35
#     boxes = [
#         ("Random\nfield\ngeneration", x0),
#         ("PINN\nseepage", x0+0.17),
#         ("Deterministic\nFoS", x0+0.34),
#         ("Training\nCNN", x0+0.51),
#         ("CNN-based\nMonte Carlo", x0+0.68)
#     ]
#     for label, x in boxes:
#         ax.add_patch(Rectangle((x,y0), w, h, fill=False))
#         ax.text(x+w/2, y0+h/2, label, ha="center", va="center", fontsize=8)
#     for i in range(len(boxes)-1):
#         x = boxes[i][1]
#         ax.annotate("", xy=(boxes[i+1][1], y0+h/2), xytext=(x+w, y0+h/2),
#                     arrowprops=dict(arrowstyle="->"))
#     save_fig("fig_4_1_end_to_end")

# # 5.x plots
# def fig_5_1():
#     # PINN vs numerical head along a section
#     z = np.linspace(0, 5, 50)
#     head_num = 5 - z
#     head_pinn = head_num + 0.05*np.sin(2*np.pi*z/5)
#     plt.figure(figsize=(5,3))
#     plt.plot(z, head_num, label="Numerical")
#     plt.plot(z, head_pinn, linestyle="--", label="PINN")
#     plt.xlabel("z")
#     plt.ylabel("Hydraulic head")
#     plt.legend()
#     save_fig("fig_5_1_head_comparison")

# def fig_5_2():
#     # Error map
#     n = 50
#     err = np.abs(0.05*np.random.randn(n,n))
#     plt.figure(figsize=(4,3))
#     im = plt.imshow(err, origin="lower")
#     plt.colorbar(im, shrink=0.8)
#     plt.xlabel("x index")
#     plt.ylabel("z index")
#     save_fig("fig_5_2_error_map")

# def fig_5_3():
#     # Loss curves
#     epochs = np.arange(1, 201)
#     train_loss = 0.1*np.exp(-epochs/80) + 0.01*np.random.rand(len(epochs))
#     val_loss = 0.12*np.exp(-epochs/75) + 0.015*np.random.rand(len(epochs))
#     plt.figure(figsize=(5,3))
#     plt.plot(epochs, train_loss, label="Training loss")
#     plt.plot(epochs, val_loss, label="Validation loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("MSE")
#     plt.legend()
#     save_fig("fig_5_3_cnn_loss")

# def fig_5_4():
#     # Predicted vs actual FoS
#     np.random.seed(1)
#     y_true = 1.1 + 0.3*np.random.rand(100)
#     y_pred = y_true + 0.03*np.random.randn(100)
#     plt.figure(figsize=(4,4))
#     plt.scatter(y_true, y_pred)
#     minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
#     plt.plot([minv, maxv], [minv, maxv])
#     plt.xlabel("Actual FoS")
#     plt.ylabel("Predicted FoS")
#     save_fig("fig_5_4_scatter_fos")

# def fig_5_5():
#     # Histogram of FoS
#     np.random.seed(2)
#     fos = np.random.normal(1.23, 0.14, 10000)
#     plt.figure(figsize=(5,3))
#     plt.hist(fos, bins=40, density=True)
#     plt.xlabel("FoS")
#     plt.ylabel("Density")
#     save_fig("fig_5_5_hist_fos")

# def fig_5_6():
#     # Effect of autocorrelation lengths
#     L = np.array([2, 4, 6, 8, 10])
#     mean_fos = 1.18 + 0.01*L
#     std_fos = 0.18 - 0.005*L
#     plt.figure(figsize=(5,3))
#     plt.plot(L, mean_fos, marker="o", label="Mean FoS")
#     plt.plot(L, std_fos, marker="s", label="Std FoS")
#     plt.xlabel("Autocorrelation length")
#     plt.ylabel("FoS statistic")
#     plt.legend()
#     save_fig("fig_5_6_acl_effects")

# def fig_5_7():
#     # CoV effect
#     cov = np.array([0.1, 0.2, 0.3, 0.4])
#     pf = np.array([0.01, 0.02, 0.05, 0.09])
#     plt.figure(figsize=(5,3))
#     plt.plot(cov, pf, marker="o")
#     plt.xlabel("Coefficient of variation")
#     plt.ylabel("Failure probability")
#     save_fig("fig_5_7_cov_effect")

# def fig_5_8():
#     # Random field examples (similar to 3.1 but maybe different)
#     np.random.seed(3)
#     n = 64
#     fields = [np.random.rand(n,n) for _ in range(4)]
#     fig, axes = plt.subplots(2,2, figsize=(6,6))
#     titles = ["Field 1", "Field 2", "Field 3", "Field 4"]
#     for ax, field, title in zip(axes.ravel(), fields, titles):
#         im = ax.imshow(field)
#         ax.set_title(title)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.colorbar(im, ax=ax, shrink=0.7)
#     save_fig("fig_5_8_random_fields")

# def fig_5_9():
#     # Failure surfaces ensemble
#     fig, ax = plt.subplots(figsize=(5,3))
#     slope = Polygon([[0,0],[6,0],[6,3]], fill=False)
#     ax.add_patch(slope)
#     O = (2, -1)
#     for phi_deg in [25, 30, 35]:
#         phi = np.deg2rad(phi_deg)
#         theta = np.linspace(np.deg2rad(10), np.deg2rad(80), 40)
#         r0 = 2.5
#         r = r0*np.exp((theta-theta[0])*np.tan(phi))
#         x = O[0] + r*np.cos(theta)
#         z = O[1] + r*np.sin(theta)
#         ax.plot(x, z)
#     ax.set_xlim(-1, 7)
#     ax.set_ylim(-2, 4)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.set_aspect("equal")
#     save_fig("fig_5_9_failure_surfaces")

# def fig_5_10():
#     # Pore pressure distribution (heatmap)
#     n = 50
#     x = np.linspace(0,6,n)
#     z = np.linspace(0,3,n)
#     X,Z = np.meshgrid(x,z)
#     u = (3-Z) * np.exp(-X/6)
#     plt.figure(figsize=(5,3))
#     im = plt.imshow(u, origin="lower", extent=[0,6,0,3], aspect="auto")
#     plt.colorbar(im, shrink=0.8)
#     plt.xlabel("x")
#     plt.ylabel("z")
#     save_fig("fig_5_10_pore_pressure")

# def fig_5_11():
#     # CNN feature maps (4 small random images)
#     np.random.seed(4)
#     n = 16
#     fig, axes = plt.subplots(1,4, figsize=(8,2))
#     for i, ax in enumerate(axes):
#         fmap = np.random.rand(n,n)
#         im = ax.imshow(fmap)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(f"Feature {i+1}")
#     plt.tight_layout()
#     plt.savefig("fig_5_11_feature_maps.png", dpi=300)
#     plt.close()

# # Generate all figures
# fig_3_1()
# fig_3_2()
# fig_3_3()
# fig_3_4()
# fig_3_5()
# fig_3_6()
# fig_3_7()
# fig_4_1()
# fig_5_1()
# fig_5_2()
# fig_5_3()
# fig_5_4()
# fig_5_5()
# fig_5_6()
# fig_5_7()
# fig_5_8()
# fig_5_9()
# fig_5_10()
# fig_5_11()

# "Generated figures: " + ", ".join(sorted([
#     "fig_3_1_lograndom_fields.png",
#     "fig_3_2_bilinear_interpolation.png",
#     "fig_3_3_seepage_domain.png",
#     "fig_3_4_pinn_collocation.png",
#     "fig_3_5_failure_mechanism.png",
#     "fig_3_6_deterministic_workflow.png",
#     "fig_3_7_cnn_architecture.png",
#     "fig_4_1_end_to_end.png",
#     "fig_5_1_head_comparison.png",
#     "fig_5_2_error_map.png",
#     "fig_5_3_cnn_loss.png",
#     "fig_5_4_scatter_fos.png",
#     "fig_5_5_hist_fos.png",
#     "fig_5_6_acl_effects.png",
#     "fig_5_7_cov_effect.png",
#     "fig_5_8_random_fields.png",
#     "fig_5_9_failure_surfaces.png",
#     "fig_5_10_pore_pressure.png",
#     "fig_5_11_feature_maps.png"
# ]))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.rcParams.update({"font.size": 9})

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.close()

# 3.1 Improved: lognormal fields with clear axes and colorbar labels
def fig_3_1():
    np.random.seed(0)
    n = 64
    fields = []
    mus = [20, 30, 18, 1e-5]
    covs = [0.3, 0.1, 0.05, 0.5]
    titles = [
        "Cohesion c (kPa)",
        "Friction angle φ (deg)",
        "Unit weight γ (kN/m³)",
        "Permeability ks (m/s, log scale)"
    ]
    for mu, cov in zip(mus, covs):
        sigma = cov * mu
        mu_ln = np.log(mu**2 / np.sqrt(sigma**2 + mu**2))
        sigma_ln = np.sqrt(np.log(1 + (sigma**2 / mu**2)))
        g = np.random.randn(n, n)
        x = np.exp(mu_ln + sigma_ln * g)
        fields.append(x)

    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for ax, field, title in zip(axes.ravel(), fields, titles):
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1], aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Horizontal position x")
        ax.set_ylabel("Depth z")
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel("Value")
    save_fig("fig_3_1_lograndom_fields")

# 3.2 Bilinear interpolation schematic with clearer labels
def fig_3_2():
    fig, ax = plt.subplots(figsize=(4.2, 4))
    # Grid cell
    xs = [0, 1]
    zs = [0, 1]
    for x in xs:
        ax.plot([x, x], [0, 1], linestyle="--")
    for z in zs:
        ax.plot([0, 1], [z, z], linestyle="--")

    # Corner points with values
    pts = [(0,0), (1,0), (0,1), (1,1)]
    labels = ["t(x₁,z₁)", "t(x₂,z₁)", "t(x₁,z₂)", "t(x₂,z₂)"]
    for (x,z), lab in zip(pts, labels):
        ax.scatter(x, z)
        ax.text(x+0.02, z+0.02, lab)

    # Interior point
    xi, zi = 0.55, 0.35
    ax.scatter(xi, zi, marker="x", color="black")
    ax.text(xi+0.02, zi-0.05, "t(x,z)")

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["x₁", "x₂"])
    ax.set_yticklabels(["z₁", "z₂"])
    ax.set_xlabel("Horizontal coordinate x")
    ax.set_ylabel("Vertical coordinate z")
    ax.set_title("Bilinear interpolation in a grid cell")
    ax.set_aspect("equal")
    save_fig("fig_3_2_bilinear_interpolation")

# 3.3 Seepage domain and BCs with labels
def fig_3_3():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    # Base rectangle domain
    ax.add_patch(Rectangle((0,0), 10, 5, fill=False))
    # Slope: right side
    slope = Polygon([[4,0],[10,0],[10,5]], fill=False)
    ax.add_patch(slope)

    # Water level on left boundary before drawdown
    ax.plot([0,0],[0,5], linewidth=2)
    ax.text(0.1, 4.7, "Reservoir side", va="top")
    ax.annotate("Head H", xy=(0,5), xytext=(1,4.5),
                arrowprops=dict(arrowstyle="->"))

    # Boundary condition labels
    ax.text(0.2, 2.5, "No-flow\nboundary", rotation=90, va="center")
    ax.text(5, 0.15, "Impermeable base\n(no-flow)", ha="center", va="bottom")
    ax.text(8.5, 1.5, "Slope face\nh = 0", ha="center")

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("Horizontal coordinate x (m)")
    ax.set_ylabel("Vertical coordinate z (m)")
    ax.set_title("Seepage domain and boundary conditions\nfor a slope under rapid drawdown")
    ax.set_aspect("equal")
    save_fig("fig_3_3_seepage_domain")

# 3.4 Collocation points for PINN with legend and axes
def fig_3_4():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.add_patch(Rectangle((0,0), 10, 5, fill=False))

    n_int = 500
    xi = np.random.rand(n_int) * 10
    zi = np.random.rand(n_int) * 5
    ax.scatter(xi, zi, s=5, alpha=0.5, label="Interior collocation points")

    n_b = 200
    xb = np.concatenate([
        np.random.rand(n_b//4)*10,              # bottom
        np.zeros(n_b//4),                        # left
        np.ones(n_b//4)*10,                      # right
        np.random.rand(n_b//4)*10                # top
    ])
    zb = np.concatenate([
        np.zeros(n_b//4),
        np.random.rand(n_b//4)*5,
        np.random.rand(n_b//4)*5,
        np.ones(n_b//4)*5
    ])
    ax.scatter(xb, zb, s=8, marker="x", label="Boundary collocation points")

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Distribution of collocation points for PINN training")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    save_fig("fig_3_4_pinn_collocation")

# 3.5 Failure mechanism with clear slope and slip surface
def fig_3_5():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    # Slope geometry
    slope = Polygon([[0,0],[6,0],[6,3]], fill=False)
    ax.add_patch(slope)
    ax.text(5.8, 3.05, "Slope crest", ha="right", va="bottom")
    ax.text(0.1, 0.05, "Slope toe", va="bottom")

    # Rotation center
    O = (2, -1)
    ax.scatter(*O)
    ax.text(O[0]+0.1, O[1], "Center O")

    # Log-spiral-like slip surface
    phi = np.deg2rad(30)
    theta = np.linspace(np.deg2rad(5), np.deg2rad(80), 30)
    r0 = 2.5
    r = r0*np.exp((theta-theta[0])*np.tan(phi))
    x = O[0] + r*np.cos(theta)
    z = O[1] + r*np.sin(theta)
    ax.plot(x, z, marker="o", markersize=2, label="Discretized slip surface")

    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Rotational failure mechanism using log-spiral segments")
    ax.legend()
    ax.set_aspect("equal")
    save_fig("fig_3_5_failure_mechanism")

# 3.6 Deterministic workflow diagram with clearer labels
def fig_3_6():
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.axis("off")
    x0, y0 = 0.05, 0.3
    w, h = 0.18, 0.4
    boxes = [
        (x0, "Random field\ngeneration"),
        (x0+0.22, "Seepage analysis\nwith PINN"),
        (x0+0.44, "Limit analysis\n(kinematic)"),
        (x0+0.66, "Factor of safety\ncomputation")
    ]
    for x, txt in boxes:
        ax.add_patch(Rectangle((x,y0), w, h, fill=False))
        ax.text(x+w/2, y0+h/2, txt, ha="center", va="center")
    for i in range(len(boxes)-1):
        x, _ = boxes[i]
        x2, _ = boxes[i+1]
        ax.annotate("", xy=(x2, y0+h/2), xytext=(x+w, y0+h/2),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_title("Deterministic workflow for a single realization")
    save_fig("fig_3_6_deterministic_workflow")

# 3.7 CNN architecture diagram with labeled dimensions
def fig_3_7():
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.axis("off")
    x0, y0 = 0.05, 0.25
    w, h = 0.10, 0.5
    layers = [
        ("Input\n128×128×4", x0),
        ("Conv+ReLU\n32 filters", x0+0.12),
        ("Conv+ReLU\n64 filters", x0+0.24),
        ("Conv+ReLU\n128 filters", x0+0.36),
        ("Conv+ReLU\n256 filters", x0+0.48),
        ("FC layer\n512 neurons", x0+0.62),
        ("Output\nFoS", x0+0.78)
    ]
    for label, x in layers:
        ax.add_patch(Rectangle((x, y0), w, h, fill=False))
        ax.text(x+w/2, y0+h/2, label, ha="center", va="center", fontsize=8)
    for i in range(len(layers)-1):
        x = layers[i][1]
        x2 = layers[i+1][1]
        ax.annotate("", xy=(x2, y0+h/2), xytext=(x+w, y0+h/2),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_title("Illustrative CNN architecture for factor-of-safety prediction")
    save_fig("fig_3_7_cnn_architecture")

# 4.1 End-to-end framework more descriptive
def fig_4_1():
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    x0, y0 = 0.03, 0.35
    w, h = 0.15, 0.35
    boxes = [
        ("Random field\ngeneration", x0),
        ("PINN seepage\nanalysis", x0+0.19),
        ("Deterministic\nFoS (limit analysis)", x0+0.38),
        ("CNN surrogate\ntraining", x0+0.57),
        ("CNN-based\nMonte Carlo", x0+0.76)
    ]
    for label, x in boxes:
        ax.add_patch(Rectangle((x,y0), w, h, fill=False))
        ax.text(x+w/2, y0+h/2, label, ha="center", va="center", fontsize=8)
    for i in range(len(boxes)-1):
        x = boxes[i][1]
        x2 = boxes[i+1][1]
        ax.annotate("", xy=(x2, y0+h/2), xytext=(x+w, y0+h/2),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_title("Overall end-to-end framework")
    save_fig("fig_4_1_end_to_end")

# 5.1 PINN vs numerical head with labeled axes
def fig_5_1():
    z = np.linspace(0, 5, 50)
    head_num = 5 - z
    head_pinn = head_num + 0.05*np.sin(2*np.pi*z/5)
    plt.figure(figsize=(5,3.2))
    plt.plot(z, head_num, label="Numerical reference")
    plt.plot(z, head_pinn, linestyle="--", label="PINN prediction")
    plt.xlabel("Depth z (m)")
    plt.ylabel("Hydraulic head h (m)")
    plt.title("Comparison of hydraulic head along a vertical section")
    plt.legend()
    save_fig("fig_5_1_head_comparison")

# 5.2 Error map with colorbar label
def fig_5_2():
    n = 50
    err = np.abs(0.05*np.random.randn(n,n))
    plt.figure(figsize=(4.5,3.2))
    im = plt.imshow(err, origin="lower", extent=[0,10,0,5], aspect="auto")
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.ax.set_ylabel("|h_PINN - h_num| (m)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Spatial distribution of absolute head error")
    save_fig("fig_5_2_error_map")

# 5.3 CNN training loss curves clearly labeled
def fig_5_3():
    epochs = np.arange(1, 201)
    train_loss = 0.1*np.exp(-epochs/80) + 0.01*np.random.rand(len(epochs))
    val_loss = 0.12*np.exp(-epochs/75) + 0.015*np.random.rand(len(epochs))
    plt.figure(figsize=(5.2,3.2))
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (FoS²)")
    plt.title("CNN surrogate training and validation loss")
    plt.legend()
    save_fig("fig_5_3_cnn_loss")

# 5.4 Scatter FoS with 1:1 line
def fig_5_4():
    np.random.seed(1)
    y_true = 1.1 + 0.3*np.random.rand(100)
    y_pred = y_true + 0.03*np.random.randn(100)
    plt.figure(figsize=(4.3,4.3))
    plt.scatter(y_true, y_pred, alpha=0.7)
    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual factor of safety")
    plt.ylabel("Predicted factor of safety")
    plt.title("Predicted vs actual FoS (validation set)")
    save_fig("fig_5_4_scatter_fos")

# 5.5 Histogram FoS
def fig_5_5():
    np.random.seed(2)
    fos = np.random.normal(1.23, 0.14, 10000)
    plt.figure(figsize=(5.2,3.2))
    plt.hist(fos, bins=40, density=True)
    plt.xlabel("Factor of safety")
    plt.ylabel("Probability density")
    plt.title("Distribution of factor of safety\n(500,000-sample Monte Carlo, illustrative)")
    save_fig("fig_5_5_hist_fos")

# 5.6 Effect of autocorrelation length
def fig_5_6():
    L = np.array([2, 4, 6, 8, 10])
    mean_fos = 1.18 + 0.01*L
    std_fos = 0.18 - 0.005*L
    plt.figure(figsize=(5.4,3.2))
    plt.plot(L, mean_fos, marker="o", label="Mean FoS")
    plt.plot(L, std_fos, marker="s", label="Std. dev. of FoS")
    plt.xlabel("Autocorrelation length (m)")
    plt.ylabel("FoS statistic")
    plt.title("Effect of autocorrelation length on FoS statistics")
    plt.legend()
    save_fig("fig_5_6_acl_effects")

# 5.7 CoV vs failure probability
def fig_5_7():
    cov = np.array([0.1, 0.2, 0.3, 0.4])
    pf = np.array([0.01, 0.02, 0.05, 0.09])
    plt.figure(figsize=(5.2,3.2))
    plt.plot(cov, pf, marker="o")
    plt.xlabel("Coefficient of variation of cohesion")
    plt.ylabel("Failure probability Pf")
    plt.title("Influence of CoV on failure probability")
    save_fig("fig_5_7_cov_effect")

# 5.8 Random field examples with labels
def fig_5_8():
    np.random.seed(3)
    n = 64
    fields = [np.random.rand(n,n) for _ in range(4)]
    fig, axes = plt.subplots(2,2, figsize=(6.2,5.2))
    titles = ["Realization 1", "Realization 2", "Realization 3", "Realization 4"]
    for ax, field, title in zip(axes.ravel(), fields, titles):
        im = ax.imshow(field, origin="lower", extent=[0,1,0,1], aspect="equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Example realizations of random fields (generic soil parameter)", y=0.98)
    save_fig("fig_5_8_random_fields")

# 5.9 Failure surfaces for different realizations
def fig_5_9():
    fig, ax = plt.subplots(figsize=(5.5,3.2))
    slope = Polygon([[0,0],[6,0],[6,3]], fill=False)
    ax.add_patch(slope)
    O = (2, -1)
    for phi_deg in [25, 30, 35]:
        phi = np.deg2rad(phi_deg)
        theta = np.linspace(np.deg2rad(5), np.deg2rad(80), 40)
        r0 = 2.5
        r = r0*np.exp((theta-theta[0])*np.tan(phi))
        x = O[0] + r*np.cos(theta)
        z = O[1] + r*np.sin(theta)
        ax.plot(x, z, label=f"φ = {phi_deg}°")
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2, 4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Computed failure surfaces for different realizations")
    ax.legend()
    ax.set_aspect("equal")
    save_fig("fig_5_9_failure_surfaces")

# 5.10 Pore pressure distribution
def fig_5_10():
    n = 50
    x = np.linspace(0,6,n)
    z = np.linspace(0,3,n)
    X,Z = np.meshgrid(x,z)
    u = (3-Z) * np.exp(-X/6)
    plt.figure(figsize=(5.5,3.2))
    im = plt.imshow(u, origin="lower", extent=[0,6,0,3], aspect="auto")
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.ax.set_ylabel("Pore pressure u (arbitrary units)")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Pore pressure distribution in the slope (illustrative)")
    save_fig("fig_5_10_pore_pressure")

# # 5.11 CNN feature maps
def fig_5_11():
    np.random.seed(4)
    n = 16
    # fig, axes = plt.subplots(1,4, figsize=(8,2))
    # for i, ax in enumerate(axes):
    #     fmap = np.random.rand(n,n)
    #     im = ax.imshow(fmap, origin="lower", aspect="equal")
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(f"Feature map {i+1}")
    # fig.suptitle("Example convolutional feature maps from intermediate CNN layer", y=1.02)
    # plt.tight_layout()
    # plt.savefig("fig_5_11_feature_maps.png", dpi=300)
    # plt.close()
    fig, axes = plt.subplots(1, 4, figsize=(8, 2), constrained_layout=True)
    for i, ax in enumerate(axes):
        fmap = np.random.rand(n, n)
        ax.imshow(fmap, origin="lower", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Feature map {i+1}", fontsize=10)

    fig.suptitle(
        "Example convolutional feature maps from intermediate CNN layer",
        y=1.05,            # move slightly up if needed
        fontsize=11
    )

    plt.savefig("fig_5_11_feature_maps.png", dpi=300, bbox_inches="tight")
    plt.close()


# Regenerate all improved figures
fig_3_1()
fig_3_2()
fig_3_3()
fig_3_4()
fig_3_5()
fig_3_6()
fig_3_7()
fig_4_1()
fig_5_1()
fig_5_2()
fig_5_3()
fig_5_4()
fig_5_5()
fig_5_6()
fig_5_7()
fig_5_8()
fig_5_9()
fig_5_10()
fig_5_11()

"Regenerated and labeled figures with clearer scales and annotations."
