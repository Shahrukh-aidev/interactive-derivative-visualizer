<div align="center">

# Interactive Derivative Visualizer

### Visualizing Calculus Through Motion, Geometry, and Intuition

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge)

<br>

**See a function and its derivative animate together in real time.**

Track tangent lines, visualize slopes instantly, and build a true geometric intuition for differentiation.

</div>

---

## Demo

### Function & Derivative Visualization

<p align="center">
  <img src="pic1.jpg" width="850">
</p>

### Real-Time Tangent Tracking

<p align="center">
  <img src="pic2.jpg" width="850">
</p>

### Animation Preview

<p align="center">
  <img src="W8me6GPdMF.mp4">
</p>

> For the best GitHub presentation, convert the MP4 into a GIF and place it at the top of this README.

---

# The Problem

Most students learn derivatives by memorizing formulas:

```text
d/dx(x²) = 2x
d/dx(sin x) = cos x
d/dx(eˣ) = eˣ
```

Yet many never develop an intuition for what a derivative actually represents.

A derivative is not merely a formula.

It is:

- The slope of a curve
- The instantaneous rate of change
- The velocity of a function
- A geometric relationship between two graphs

Unfortunately, textbooks usually present static diagrams that make these ideas difficult to visualize.

## The Solution

**Interactive Derivative Visualizer** transforms differentiation into a real-time visual experience.

Instead of staring at equations, users can watch:

- A point move along a function
- Its tangent line update continuously
- The derivative graph evolve simultaneously
- Slope magnitude and direction encoded through color

This creates an intuitive understanding of calculus through direct visual feedback.

---

# Features

| Feature | Description |
|----------|-------------|
| Real-Time Synchronization | Function and derivative animate together |
| Moving Tangent Line | Displays instantaneous slope at every point |
| Dynamic Color Mapping | Slope encoded through color transitions |
| Custom Function Input | Enter your own mathematical expressions |
| Adjustable Domain | Modify graph ranges instantly |
| Animation Controls | Play, pause, and step frame-by-frame |
| Resolution Control | Increase or decrease graph precision |
| Trail Effect | Leave historical points behind the animation |
| Dark & Light Themes | Switch between presentation and study modes |
| Numerical Differentiation | Works on arbitrary valid functions |

---

# How It Works

```text
                    User Function Input
                              │
                              ▼
                Safe Mathematical Parser
                              │
                              ▼
                     NumPy Evaluation
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼

      Function f(x)                       Derivative f'(x)

          │                                       │
          └───────────────────┬───────────────────┘
                              │
                              ▼
                 Matplotlib Animation Engine
                              │
                              ▼
                 Interactive Visual Experience
```

---

# Architecture

```text
╔══════════════════════════════════════════════════════╗
║                VISUALIZER PIPELINE                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  User Function Input                                ║
║          │                                           ║
║          ▼                                           ║
║  Safe Expression Evaluation                          ║
║          │                                           ║
║          ▼                                           ║
║  NumPy Function Computation                          ║
║          │                                           ║
║          ├──► f(x)                                   ║
║          │                                           ║
║          └──► f'(x) using np.gradient()              ║
║                     │                                ║
║                     ▼                                ║
║         Matplotlib Animation Loop                    ║
║                     │                                ║
║                     ▼                                ║
║        Interactive Visualization Canvas             ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

---

# Mathematical Model

The derivative is approximated numerically using finite differences:

```python
dy_dx = np.gradient(y_values, dx)
```

Conceptually:

```text
f'(x) ≈ Δy / Δx
```

This allows the application to compute and visualize the instantaneous rate of change for virtually any supported function.

---

# Dynamic Slope Color Mapping

One of the most visually distinctive features of the project is slope-based color encoding.

The slope is transformed using the hyperbolic tangent function:

```math
t = tanh(f'(x))
```

This smoothly compresses extreme slope values into the interval:

```text
[-1 , 1]
```

The value is then mapped into RGB color space.

| Slope | Color |
|---------|---------|
| Large Negative | Blue |
| Zero | Purple |
| Large Positive | Red |

### Visual Interpretation

```text
Negative Slope  →  Blue
Zero Slope      →  Purple
Positive Slope  →  Red
```

This enables users to instantly recognize:

- Increasing regions
- Decreasing regions
- Local maxima
- Local minima
- Stationary points

without reading numerical values.

---

# Interactive Controls

| Control | Function |
|----------|----------|
| Function Input | Enter custom mathematical expressions |
| Domain Input | Set graph boundaries |
| Speed Slider | Control animation speed |
| Samples Slider | Adjust graph resolution |
| Trail Slider | Change history trail length |
| Play / Pause | Start or stop animation |
| Theme Toggle | Switch between Dark and Light mode |
| Arrow Keys | Step through frames manually |

---

# Supported Functions

### Trigonometric

```python
sin(x)
cos(x)
tan(x)
```

### Inverse Trigonometric

```python
arcsin(x)
arccos(x)
arctan(x)
```

### Hyperbolic

```python
sinh(x)
cosh(x)
tanh(x)
```

### Algebraic

```python
x**2
sqrt(x)
abs(x)
```

### Exponential & Logarithmic

```python
exp(x)
log(x)
```

### Constants

```python
pi
e
```

### Example Inputs

```python
sin(x)
sin(x) * exp(-0.2*x)
x**3 - 3*x
cos(x) + x**2
sqrt(abs(x))
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/Shahrukh-aidev/interactive-derivative-visualizer.git

cd interactive-derivative-visualizer
```

## Install Dependencies

```bash
pip install numpy matplotlib
```

## Run the Application

```bash
python python_deriv_viz.py
```

---

# Project Structure

```text
interactive-derivative-visualizer
│
├── python_deriv_viz.py
├── pic1.jpg
├── pic2.jpg
├── W8me6GPdMF.mp4
├── LICENSE
└── README.md
```

---

# Educational Value

This project is useful for:

- Calculus Students
- Mathematics Educators
- STEM Demonstrations
- Interactive Learning
- Classroom Presentations
- Self-Study and Concept Building

It helps bridge the gap between symbolic mathematics and geometric intuition.

---

# Future Enhancements

- Symbolic differentiation using SymPy
- Second derivative visualization
- Integral visualization mode
- Multi-function comparison
- 3D derivative surfaces
- Export animations to GIF/MP4
- Function presets library
- Tangent slope value display

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Core application |
| NumPy | Mathematical computation |
| Matplotlib | Visualization and animation |
| Matplotlib Widgets | Interactive controls |

---

# License

Licensed under the Apache License 2.0.

You are free to use, modify, distribute, and build upon this project in accordance with the license terms.

---

<div align="center">

### Built to make calculus intuitive.

**If this project helped you learn or teach calculus, consider starring the repository.**

⭐ Star the project to support its development.

</div>
