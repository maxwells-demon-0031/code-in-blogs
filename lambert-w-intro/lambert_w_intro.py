# An introduction to the Lambert W function: all figures, tables and numbers
#
# Reproduces Figures 1 to 5, Tables 1 and 2, and every number quoted in the article.
# Result numbers in the comments refer to the numbered equations in the article.
#
# Run:  python lambert_w_intro.py
# Needs: numpy, scipy, matplotlib. SciPy is used only for the reference values of W.
# Output: the numbers are printed; the figures and tables are saved as PNG files.


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import lambertw   # used only as the reference value W0(x)

np.seterr(all='ignore')   # let exp overflow to inf and give nan, as in the article

n_iter = 2000   # at most 2000 iterations
tol = 1e-14     # stop when one iteration changes the guess by at most 1e-14


# Each method returns the iterates w = [w_0, w_1, ...], the step sizes
# E[n] = |w_(n+1) - w_n| and the errors E_act[n] = |w_n - W0(x)|.

def fixed_point_iterations(x, w_init, n_iter):
    w = [w_init]
    E = []
    E_act = []
    w_ref = lambertw(x)                      # W0(x), principal branch
    for k in range(n_iter):
        w.append(x*np.exp(-w[k]))            # Result 15, first rule: w = x exp(-w) (Result 14)
        E.append(abs(w[k+1] - w[k]))
        E_act.append(abs(w[k] - w_ref))
        if E[k] <= tol:
            break
    E_act.append(abs(w[-1] - w_ref))
    return np.array(w), np.array(E), np.array(E_act)


def log_fixed_point_iterations(x, w_init, n_iter):
    w = [w_init]
    E = []
    E_act = []
    w_ref = lambertw(x)
    for k in range(n_iter):
        w.append(np.log(x/w[k]))             # Result 15, second rule: w = ln(x/w) (Result 14), needs x/w > 0
        E.append(abs(w[k+1] - w[k]))
        E_act.append(abs(w[k] - w_ref))
        if E[k] <= tol:
            break
    E_act.append(abs(w[-1] - w_ref))
    return np.array(w), np.array(E), np.array(E_act)


def Newton_iterations(x, w_init, n_iter):
    w = [w_init]
    E = []
    E_act = []
    w_ref = lambertw(x)
    for k in range(n_iter):
        f  = w[k]*np.exp(w[k])               # f(w) = w e^w, Result 6
        fp = np.exp(w[k])*(w[k] + 1)         # f'(w), Result 7
        w.append(w[k] - (f - x)/fp)          # Newton's method, Result 16
        E.append(abs(w[k+1] - w[k]))
        E_act.append(abs(w[k] - w_ref))
        if E[k] <= tol:
            break
    E_act.append(abs(w[-1] - w_ref))
    return np.array(w), np.array(E), np.array(E_act)


def Halley_iterations(x, w_init, n_iter):
    w = [w_init]
    E = []
    E_act = []
    w_ref = lambertw(x)
    for k in range(n_iter):
        f   = w[k]*np.exp(w[k])              # f(w), Result 6
        fp  = np.exp(w[k])*(w[k] + 1)        # f'(w), Result 7
        fpp = np.exp(w[k])*(w[k] + 2)        # f''(w), Result 8
        w.append(w[k] - 2*(f - x)*fp/(2*fp**2 - (f - x)*fpp))   # Halley's method, Result 17
        E.append(abs(w[k+1] - w[k]))
        E_act.append(abs(w[k] - w_ref))
        if E[k] <= tol:
            break
    E_act.append(abs(w[-1] - w_ref))
    return np.array(w), np.array(E), np.array(E_act)


# How fast? (Figure 5): x = 1.5, all methods from w_0 = 0
x = 1.5
w_init = 0.0
w1, E1, E1_act = fixed_point_iterations(x, w_init, n_iter)
w2, E2, E2_act = Newton_iterations(x, w_init, n_iter)
w3, E3, E3_act = Halley_iterations(x, w_init, n_iter)
print(f'How fast? x = 1.5, w_0 = 0, W0(1.5) = {lambertw(x).real:.4f}')
print(f'  fixed point: {len(E1):3d} iterations, final error {E1_act[-1]:.1e}')
print(f'  Newton:      {len(E2):3d} iterations, error exactly 0 from n = {np.argmax(E2_act == 0)}')
print(f'  Halley:      {len(E3):3d} iterations, error exactly 0 from n = {np.argmax(E3_act == 0)}')

# Which answer do you get? (Table 1): x = -2 e^-2, Result 12
x = -2*np.exp(-2)
print(f'\nWhich answer? x = -2e^-2; W0 = -0.406 and W-1 = -2 (Result 13)')
names = ['fixed point, w = x e^-w', 'fixed point, w = ln(x/w)', 'Newton', 'Halley']
methods = [fixed_point_iterations, log_fixed_point_iterations, Newton_iterations, Halley_iterations]
for w_init in [-0.5, -1.5]:
    for i in range(4):
        w, E, E_act = methods[i](x, w_init, n_iter)
        branch = 'W0' if w[-1] >= -1 else 'W-1'   # W0 has w >= -1, W-1 has w <= -1
        print(f'  w_0 = {w_init:4.1f}, {names[i]:24s}: {branch:3s} = {w[-1]:.15f}, {len(E):2d} iterations')

# When does it go wrong? (Table 2): x = 1.5, whose only real solution is W0(1.5),
# from two starting guesses at or to the left of the minimum of f at w = -1
x = 1.5
print('\nWhen does it go wrong? x = 1.5 (Table 2)')
for w_init in [-0.999, -1.3]:
    w2, E2, E2_act = Newton_iterations(x, w_init, n_iter)
    w3, E3, E3_act = Halley_iterations(x, w_init, n_iter)
    w1, E1, E1_act = fixed_point_iterations(x, w_init, n_iter)   # does not use f' (Result 7)
    print(f'  w_0 = {w_init}:')
    print(f'    Newton:      w_1, w_2, w_3, w_4 = {w2[1:5]}, stopping rule not met in {len(E2)} iterations')
    print(f'    Halley:      reaches {w3[-1]:.4f} in {len(E3)} iterations')
    print(f'    fixed point: reaches {w1[-1]:.4f} in {len(E1)} iterations')

# No real solution: x = -0.5 < -1/e, all methods from w_0 = 0
x = -0.5
w1, E1, E1_act = fixed_point_iterations(x, 0.0, n_iter)
w2, E2, E2_act = Newton_iterations(x, 0.0, n_iter)
w3, E3, E3_act = Halley_iterations(x, 0.0, n_iter)
print('\nNo real solution: x = -0.5, w_0 = 0')
print(f'  fixed point: w_1, w_2, w_3 = {w1[1:4]}, ends at {w1[-1]}')
i_bad = np.argmax(~np.isfinite(w2))   # first iterate that is not a finite number
print(f'  Newton:      last finite iterate w_{i_bad-1} = {w2[i_bad-1]:.2g}, then {w2[i_bad]}')
print(f'  Halley:      stops after {len(E3)} iterations at w = {w3[-1]}, '
      f'residual |w e^w - x| = {abs(w3[-1]*np.exp(w3[-1]) - x):.3f}')   # residual of Result 5


# ============================================================================
# Figures and tables
# ============================================================================

plt.rcParams.update({'font.size': 13, 'mathtext.fontset': 'dejavusans'})
c_fp   = (0.90, 0.55, 0.10)   # fixed point: orange
c_nt   = (0.47, 0.27, 0.62)   # Newton: purple
c_hl   = (0.10, 0.60, 0.45)   # Halley: green
c_grid = (0.92, 0.92, 0.92)
eps = np.finfo(float).eps     # errors that are exactly zero are plotted at eps


# ============================================================================
# Figures 1 to 4
# ============================================================================

c_blue  = (0.12, 0.37, 0.66)
c_red   = (0.84, 0.29, 0.35)
c_teal  = (0.16, 0.62, 0.56)
c_lblue = (0.38, 0.55, 0.78)
c_lred  = (0.88, 0.50, 0.55)
c_axis  = '0.45'


def style(ax):
    ax.grid(True, color=c_grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# Figures 1 to 3 use a LaTeX-like serif style: Computer Modern fonts, ticks pointing inwards,
# plain tick labels, only the left and bottom axes, and a legend without a frame.
serif = {'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
         'axes.formatter.use_mathtext': True, 'font.size': 14,
         'xtick.direction': 'in', 'ytick.direction': 'in'}
plain = matplotlib.ticker.FuncFormatter(lambda v, pos: '$%g$' % v)   # tick labels as maths, so minus signs render

with plt.rc_context(serif):

    # Figure 1: both sides of Result 1, 1 - z = exp(-R0 z), for R0 = 2
    R0 = 2
    z = np.linspace(0, 1, 400)
    z_star = 1 + lambertw(-R0*np.exp(-R0)).real/R0   # second crossing, from W0 (Result 13)
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    style(ax)
    ax.plot(z, 1 - z, color=c_blue, lw=2, label=r'$1 - z$')
    ax.plot(z, np.exp(-R0*z), color=c_red, lw=2, label=r'$e^{-2z}$')
    ax.plot([0, z_star], [1, 1 - z_star], 'ko', ms=7, zorder=5)
    ax.text(0.03, 1.0, r'$z = 0$', va='bottom')
    ax.text(z_star + 0.01, 1 - z_star + 0.02, r'$z \approx %.3f$' % z_star, va='bottom')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('value')
    ax.legend(frameon=False, loc='upper right', handlelength=2.5)
    fig.tight_layout()
    fig.savefig('fig1_epidemic.png', dpi=200)

    # Figure 2: f(w) = w exp(w) (Result 6) with its minimum at (-1, -1/e) (Result 10)
    w = np.linspace(-6, 1.3, 800)
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    style(ax)
    ax.axhline(0, color=c_axis, lw=0.8)
    ax.axvline(0, color=c_axis, lw=0.8)
    ax.axhline(-np.exp(-1), color=c_red, ls=':', lw=1.2)
    ax.plot(w, w*np.exp(w), 'k', lw=2)
    ax.plot(-1, -np.exp(-1), 'o', color=c_red, ms=9, zorder=5)
    ax.plot([-0.95, -0.55], [-0.39, -0.45], '-', color=c_red, lw=1)
    ax.text(-1.4, -0.62, r'minimum $(-1,\, -1/e)$', color=c_red)
    ax.text(-5.8, 0.42, 'decreasing,\napproaches 0 from below')
    ax.text(-0.05, 2.15, 'increasing', ha='right')
    ax.set_xlim(-6, 1.3)
    ax.set_ylim(-0.7, 2.5)
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$f(w) = w\,e^{w}$')
    fig.tight_layout()
    fig.savefig('fig2_f_sketch.png', dpi=200)

    # Figure 3: f(w) with horizontal lines at three values of x
    x1, x2, x3 = 1.0, -2*np.exp(-2), -0.5             # one, two and no solutions
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    style(ax)
    ax.axhline(0, color=c_axis, lw=0.8)
    ax.axvline(0, color=c_axis, lw=0.8)
    ax.axhline(x1, color=c_teal, ls='--', lw=1.6, label=r'$x = 1$: one solution')
    ax.axhline(x2, color=c_blue, ls='--', lw=1.6, label=r'$x = -2e^{-2}$: two solutions')
    ax.axhline(x3, color=c_red, ls='--', lw=1.6, label=r'$x = -0.5$: no solution')
    ax.plot(w, w*np.exp(w), 'k', lw=2)
    ax.plot(lambertw(x1).real, x1, 'o', color=c_teal, ms=8, zorder=5)   # reference values, only to place the markers
    ax.plot([lambertw(x2, -1).real, lambertw(x2).real], [x2, x2], 'o', color=c_blue, ms=8, zorder=5)
    ax.set_xlim(-6, 1.3)
    ax.set_ylim(-0.7, 2.5)
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$f(w) = w\,e^{w}$')
    ax.legend(frameon=False, loc='upper left', handlelength=2.5)
    fig.tight_layout()
    fig.savefig('fig3_horizontal_lines.png', dpi=200)

# Figure 4: y = x exp(x) and y = W(x) on the same axes.
# Each branch is drawn exactly as the curve (x exp(x), x): the matching piece of x exp(x) mirrored in y = x.
xa = np.linspace(-1, 0.8526, 400)                     # piece x >= -1, up to y = 2
xb = np.linspace(-4, -1, 400)                         # piece x <= -1
fig, ax = plt.subplots(figsize=(6.2, 6.2))
style(ax)
ax.axhline(0, color=c_axis, lw=0.8)
ax.axvline(0, color=c_axis, lw=0.8)
ax.plot([-4, 2], [-4, 2], ':', color='0.4', lw=1.2)
ax.plot(xa, xa*np.exp(xa), '--', color=c_lblue, lw=1.8, label=r'$y = x\,e^{x}$, $x \geq {-1}$')
ax.plot(xb, xb*np.exp(xb), '--', color=c_lred, lw=1.8, label=r'$y = x\,e^{x}$, $x \leq {-1}$')
ax.plot(xa*np.exp(xa), xa, color=c_blue, lw=2.6, label=r'$y = W_0(x)$')
ax.plot(xb*np.exp(xb), xb, color=c_red, lw=2.6, label=r'$y = W_{-1}(x)$')
ax.plot([-1, -np.exp(-1)], [-np.exp(-1), -1], 'ko', ms=6, zorder=5)
ax.text(1.35, 1.0, r'$y = x$', color='0.4', fontsize=12)
ax.set_xlim(-4, 2)
ax.set_ylim(-4, 2)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.legend(frameon=False, loc='upper left', fontsize=12)
fig.tight_layout()
fig.savefig('fig4_branches.png', dpi=200)


# ============================================================================
# Figure 5 and Tables 1 and 2
# ============================================================================

# Figure 5
x = 1.5
w1, E1, E1_act = fixed_point_iterations(x, 0.0, n_iter)
w2, E2, E2_act = Newton_iterations(x, 0.0, n_iter)
w3, E3, E3_act = Halley_iterations(x, 0.0, n_iter)

fig, ax = plt.subplots(figsize=(6.9, 4.3))
ax.grid(True, color=c_grid, linewidth=0.6)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.semilogy(np.arange(len(E1_act)), np.maximum(E1_act, eps), 'o-', color=c_fp, mfc=c_fp, ms=3, lw=1.4, label='Fixed point')
ax.semilogy(np.arange(len(E2_act)), np.maximum(E2_act, eps), 's-', color=c_nt, mfc=c_nt, ms=6, lw=1.8, label='Newton')
ax.semilogy(np.arange(len(E3_act)), np.maximum(E3_act, eps), '^-', color=c_hl, mfc=c_hl, ms=6, lw=1.8, label='Halley')
ax.set_xlim(0, 105)
ax.set_ylim(1e-17, 10)
ax.set_yticks([1e-15, 1e-10, 1e-5, 1])
ax.minorticks_off()
ax.set_xlabel(r'iteration $n$')
ax.set_ylabel(r'error $|w_n - W_0(1.5)|$')
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig('fig5_error_x15.png', dpi=200)

# Table 1 image, built from the same runs
x = -2*np.exp(-2)
row_names = [r'Fixed point, $w = x\,e^{-w}$', r'Fixed point, $w = \ln(x/w)$', 'Newton', 'Halley']
cells = []
for i in range(4):
    row = []
    for w_init in [-0.5, -1.5]:
        w, E, E_act = methods[i](x, w_init, n_iter)
        branch = r'$W_0$' if w[-1] >= -1 else r'$W_{-1}$'
        row.append(f'{branch}, {len(E)} iterations')
    cells.append(row)

plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.serif': ['cmr10']})
fig = plt.figure(figsize=(6.6, 1.75))
col_x = [0.02, 0.45, 0.74]
y_head = 0.86
dy = 0.17
fig.text(col_x[0], y_head, 'Method', va='center', fontsize=12)
fig.text(col_x[1], y_head, r'From $w_0 = -0.5$', va='center', fontsize=12)
fig.text(col_x[2], y_head, r'From $w_0 = -1.5$', va='center', fontsize=12)
for i in range(4):
    y = y_head - (i + 1.35)*dy
    fig.text(col_x[0], y, row_names[i], va='center', fontsize=12)
    fig.text(col_x[1], y, cells[i][0], va='center', fontsize=12)
    fig.text(col_x[2], y, cells[i][1], va='center', fontsize=12)
for y, lw in [(0.98, 1.2), (y_head - 0.6*dy, 0.7), (y_head - 4.85*dy, 1.2)]:
    fig.add_artist(plt.Line2D([0.01, 0.99], [y, y], color='k', lw=lw))
fig.savefig('table1.png', dpi=180, facecolor='white')

# Table 2 image, built from the same runs
x = 1.5
row_names = [r'$w_0 = -0.999$', r'$w_0 = -1.3$']
cells = []
for w_init in [-0.999, -1.3]:
    w2, E2, E2_act = Newton_iterations(x, w_init, n_iter)
    w3, E3, E3_act = Halley_iterations(x, w_init, n_iter)
    w1, E1, E1_act = fixed_point_iterations(x, w_init, n_iter)
    i_bad = np.argmax(~np.isfinite(w2))                       # first iterate that is not a finite number
    i_big = i_bad - 1                                         # last finite iterate
    m, p = ('%.1e' % w2[i_big]).split('e')
    if abs(w2[i_big]) < 1e6:
        big = r'$w_{%d} \approx %d$' % (i_big, round(w2[i_big]))
    else:
        big = r'$w_{%d} \approx {%s} \times 10^{%d}$' % (i_big, m, int(p))
    if np.isinf(w2[i_bad]):
        big = big + r', then $-\infty$'                       # iterate that is infinite before NaN
    cells.append([big + ', then NaN', r'$W_0(1.5)$, %d iterations' % len(E3), r'$W_0(1.5)$, %d iterations' % len(E1)])

fig = plt.figure(figsize=(8.6, 1.15))
col_x = [0.015, 0.155, 0.535, 0.77]
y_head = 0.80
dy = 0.25
for c, h in zip(col_x, ['Starting guess', 'Newton', 'Halley', 'Fixed point']):
    fig.text(c, y_head, h, va='center', fontsize=12)
for i in range(2):
    y = y_head - (i + 1.35)*dy
    fig.text(col_x[0], y, row_names[i], va='center', fontsize=12)
    for j in range(3):
        fig.text(col_x[j+1], y, cells[i][j], va='center', fontsize=12)
for y, lw in [(0.98, 1.2), (y_head - 0.6*dy, 0.7), (y_head - 2.95*dy, 1.2)]:
    fig.add_artist(plt.Line2D([0.005, 0.995], [y, y], color='k', lw=lw))
fig.savefig('table2.png', dpi=180, facecolor='white')
