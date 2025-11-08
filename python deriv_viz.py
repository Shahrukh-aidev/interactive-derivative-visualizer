"""
deriv_viz.py
Interactive derivative visualizer using matplotlib widgets.

Features:
- Top plot: user function f(x) with moving point and tangent line
- Bottom plot: derivative f'(x) with synced moving point
- Controls: Play/Pause, Speed slider, Domain sliders, Resolution slider, Function TextBox, Trail length
- Color-coded moving point (blue -> red) by slope sign/magnitude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, TextBox

# Safe eval environment
_safe_dict = {
    'np': np,
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs,
    'pi': np.pi, 'e': np.e,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'floor': np.floor, 'ceil': np.ceil,
    'x': None
}

def safe_eval_expr(expr, x_vals):
    local = _safe_dict.copy()
    local['x'] = x_vals
    try:
        y = eval(expr, {"__builtins__": {}}, local)
        y = np.array(y, dtype=float)
        y[np.isinf(y)] = np.nan
        return y
    except:
        return np.full_like(x_vals, np.nan, dtype=float)

DEFAULT_FUNC = "sin(x)"
XMIN_DEFAULT = -2 * np.pi
XMAX_DEFAULT = 2 * np.pi
SAMPLES_DEFAULT = 800
SPEED_DEFAULT = 40
TRAIL_DEFAULT = 18

tooltip_text = (
    "Controls:\n"
    "- Space: Play/Pause\n"
    "- ←/→: Step backward/forward\n"
    "- Edit function & press Enter\n"
    "- Use sliders for speed, samples, trail\n"
    "- Drag point (hold mouse near it) [coming soon]"
)

plt.style.use('dark_background')
fig = plt.figure(figsize=(10,7))
ax1 = plt.subplot2grid((6,4), (0,0), colspan=4, rowspan=3)
ax2 = plt.subplot2grid((6,4), (3,0), colspan=4, rowspan=3, sharex=ax1)
plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.18, hspace=0.35)

state = {
    'func_str': DEFAULT_FUNC,
    'xmin': XMIN_DEFAULT,
    'xmax': XMAX_DEFAULT,
    'samples': SAMPLES_DEFAULT,
    'speed': SPEED_DEFAULT,
    'trail': TRAIL_DEFAULT,
    'playing': False,
    'index': 0,
    'mode': 'dark'
}

def recompute_mesh():
    xs = np.linspace(state['xmin'], state['xmax'], state['samples'])
    ys = safe_eval_expr(state['func_str'], xs)
    dx = (state['xmax'] - state['xmin']) / (state['samples'] - 1 + 1e-12)
    dys = np.gradient(ys, dx)
    return xs, ys, dys

xs, ys, dys = recompute_mesh()

(line_f,) = ax1.plot(xs, ys, lw=2, color='cyan', label='f(x)')
(point,) = ax1.plot([], [], 'o', color='crimson', markersize=8, label='moving point')
(tangent_line,) = ax1.plot([], [], '--', color='lime', lw=2, label='tangent')
(deriv_line,) = ax2.plot(xs, dys, lw=2, color='magenta', label="f'(x)")
(deriv_point,) = ax2.plot([], [], 'o', color='dodgerblue', markersize=7, label='slope point')

ax1.set_title("f(x) and Tangent Line", color='w')
ax2.set_title("Derivative f'(x)", color='w')
ax1.set_xlim(state['xmin'], state['xmax'])
ax2.set_xlim(state['xmin'], state['xmax'])
ax1.set_ylim(np.nanmin(ys) - 0.5, np.nanmax(ys) + 0.5)
ax2.set_ylim(np.nanmin(dys) - 0.5, np.nanmax(dys) + 0.5)
ax1.grid(alpha=0.2)
ax2.grid(alpha=0.2)

slope_text = ax1.text(0.75, 0.88, "", transform=ax1.transAxes, fontsize=12,
                      bbox=dict(facecolor='white', alpha=0.8), color='black')

trail_scatter = ax1.scatter([], [], s=40, alpha=0.5, edgecolors='none')

axcolor = 'lightgoldenrodyellow'
axfunc = plt.axes([0.12, 0.08, 0.56, 0.045], facecolor=axcolor)
text_func = TextBox(axfunc, 'Function f(x):', initial=state['func_str'])
ax_speed = plt.axes([0.12, 0.02, 0.3, 0.03], facecolor=axcolor)
slider_speed = Slider(ax_speed, 'Speed (fps)', 1.0, 120.0, valinit=state['speed'], valfmt='%0.0f')
ax_range = plt.axes([0.75, 0.08, 0.22, 0.045], facecolor=axcolor)
text_range = TextBox(ax_range, 'x min,x max', initial=f"{state['xmin']:.3f}, {state['xmax']:.3f}")
ax_samples = plt.axes([0.75, 0.02, 0.1, 0.03], facecolor=axcolor)
slider_samples = Slider(ax_samples, 'Samples', 200, 2500, valinit=state['samples'], valfmt='%0.0f')
ax_trail = plt.axes([0.87, 0.02, 0.1, 0.03], facecolor=axcolor)
slider_trail = Slider(ax_trail, 'Trail', 0, 100, valinit=state['trail'], valfmt='%0.0f')
ax_play = plt.axes([0.02, 0.02, 0.08, 0.05])
btn_play = Button(ax_play, 'Play ▶', color='lightgreen', hovercolor='green')
ax_pause = plt.axes([0.02, 0.08, 0.08, 0.05])
btn_pause = Button(ax_pause, 'Pause ⏸', color='lightcoral', hovercolor='red')
ax_mode = plt.axes([0.85, 0.08, 0.12, 0.045])
btn_mode = Button(ax_mode, 'Toggle Light/Dark', color='gray', hovercolor='lightgray')
ax_tooltip = plt.axes([0.85, 0.02, 0.12, 0.045])
btn_tooltip = Button(ax_tooltip, "?", color='deepskyblue', hovercolor='dodgerblue')

tooltip_box = fig.text(0.85, 0.5, tooltip_text, fontsize=9, color='white',
                       bbox=dict(facecolor='black', alpha=0.7), visible=False)

def update_plot_elements():
    global xs, ys, dys
    xs, ys, dys = recompute_mesh()
    line_f.set_data(xs, ys)
    deriv_line.set_data(xs, dys)
    ax1.set_xlim(state['xmin'], state['xmax'])
    ax2.set_xlim(state['xmin'], state['xmax'])
    if np.all(np.isnan(ys)):
        ax1.set_ylim(-1, 1)
    else:
        y_min = np.nanmin(ys)
        y_max = np.nanmax(ys)
        if not np.isfinite(y_min): y_min = -1
        if not np.isfinite(y_max): y_max = 1
        margin = max(0.5, 0.1 * (y_max - y_min))
        ax1.set_ylim(y_min - margin, y_max + margin)
    if np.all(np.isnan(dys)):
        ax2.set_ylim(-1, 1)
    else:
        d_min = np.nanmin(dys)
        d_max = np.nanmax(dys)
        if not np.isfinite(d_min): d_min = -1
        if not np.isfinite(d_max): d_max = 1
        margin = max(0.5, 0.1 * (d_max - d_min))
        ax2.set_ylim(d_min - margin, d_max + margin)
    fig.canvas.draw_idle()

def get_tangent_segment(x0, y0, slope, span_frac=0.15):
    tspan = (state['xmax'] - state['xmin']) * span_frac
    tx = np.linspace(x0 - tspan/2, x0 + tspan/2, 50)
    ty = slope * (tx - x0) + y0
    return tx, ty

def animate(frame):
    idx = state['index'] % max(1, len(xs))
    xi = xs[idx]
    yi = ys[idx] if not np.isnan(ys[idx]) else np.nan
    slope_val = dys[idx] if not np.isnan(dys[idx]) else np.nan

    tx, ty = get_tangent_segment(xi, yi, slope_val)
    tangent_line.set_data(tx, ty)

    point.set_data([xi], [yi])
    deriv_point.set_data([xi], [slope_val])

    if np.isfinite(slope_val):
        t = np.tanh(slope_val)
        r = int(255 * (t + 1) / 2)
        b = int(255 * (1 - (t + 1) / 2))
        color_rgb = (r/255, 0.15, b/255)
        point.set_markerfacecolor(color_rgb)
    else:
        point.set_markerfacecolor('gray')

    slope_text.set_text(f"Slope = {slope_val:.4f}" if np.isfinite(slope_val) else "Slope = NaN")

    trail_len = int(state['trail'])
    if trail_len > 0:
        start = max(0, idx - trail_len)
        txs = xs[start:idx+1]
        tys = ys[start:idx+1]
        trail_scatter.set_offsets(np.c_[txs, tys])
    else:
        trail_scatter.set_offsets(np.empty((0, 2)))

    return tangent_line, point, deriv_point, slope_text, trail_scatter

def submit_func(text):
    state['func_str'] = text.strip() if text.strip() != "" else "sin(x)"
    state['index'] = 0
    update_plot_elements()

def update_speed(val):
    state['speed'] = float(val)

def update_samples(val):
    state['samples'] = int(val)
    state['index'] = 0
    update_plot_elements()

def update_trail(val):
    state['trail'] = int(val)

def submit_range(text):
    try:
        parts = text.split(',')
        a = float(parts[0].strip())
        b = float(parts[1].strip())
        if a == b:
            raise ValueError("xmin != xmax required")
        state['xmin'], state['xmax'] = min(a, b), max(a, b)
        state['index'] = 0
        update_plot_elements()
    except:
        text_func.set_val(state['func_str'])
        text_range.set_val(f"{state['xmin']:.3f}, {state['xmax']:.3f}")

def play(event):
    state['playing'] = True

def pause(event):
    state['playing'] = False

def toggle_mode(event):
    if state['mode'] == 'dark':
        state['mode'] = 'light'
        fig.patch.set_facecolor('white')
        for ax in [ax1, ax2]:
            ax.set_facecolor('white')
            ax.title.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.tick_params(colors='black')
            ax.grid(color='gray', alpha=0.3)
        line_f.set_color('blue')
        tangent_line.set_color('green')
        deriv_line.set_color('purple')
        point.set_markerfacecolor('red')
        deriv_point.set_markerfacecolor('darkblue')
        slope_text.set_color('black')
        fig.texts[0].set_color('black')
    else:
        state['mode'] = 'dark'
        fig.patch.set_facecolor('black')
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            ax.grid(color='white', alpha=0.2)
        line_f.set_color('cyan')
        tangent_line.set_color('lime')
        deriv_line.set_color('magenta')
        animate(None)
        slope_text.set_color('black')
        fig.texts[0].set_color('white')
    fig.canvas.draw_idle()

def toggle_tooltip(event):
    tooltip_box.set_visible(not tooltip_box.get_visible())
    fig.canvas.draw_idle()

text_func.on_submit(submit_func)
slider_speed.on_changed(update_speed)
slider_samples.on_changed(update_samples)
slider_trail.on_changed(update_trail)
text_range.on_submit(submit_range)
btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_mode.on_clicked(toggle_mode)
btn_tooltip.on_clicked(toggle_tooltip)

def on_key(event):
    if event.key == ' ':
        state['playing'] = not state['playing']
    elif event.key == 'right':
        state['index'] = (state['index'] + 1) % len(xs)
    elif event.key == 'left':
        state['index'] = (state['index'] - 1) % len(xs)

fig.canvas.mpl_connect('key_press_event', on_key)

def timer_loop():
    if state['playing']:
        state['index'] = (state['index'] + 1) % max(1, len(xs))
        animate(None)
        fig.canvas.draw_idle()
    new_fps = max(1, int(state['speed']))
    new_interval = int(1000 / new_fps)
    fig.canvas.new_timer(interval=new_interval).add_callback(timer_loop)

main_timer = fig.canvas.new_timer(interval=int(1000 / max(1, state['speed'])))
main_timer.add_callback(timer_loop)
main_timer.start()

ani = FuncAnimation(fig, animate, frames=2000000, interval=50, blit=False)

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
fig.text(0.01, 0.96, "Space = Play/Pause | ← → to step | Edit function and press Enter", fontsize=9, color='white')

plt.show()
