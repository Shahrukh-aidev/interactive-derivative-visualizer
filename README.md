<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/Engine-Matplotlib-orange?style=for-the-badge&logo=python" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Math-NumPy-013243?style=for-the-badge&logo=numpy" alt="NumPy" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="License" />

  <br/><br/>

  <h1>Interactive Derivative Visualizer</h1>
  <pre>
██████╗ ███████╗██████╗ ██╗██╗   ██╗
██╔══██╗██╔════╝██╔══██╗██║██║   ██║
██║  ██║█████╗  ██████╔╝██║██║   ██║
██║  ██║██╔══╝  ██╔══██╗██║╚██╗ ██╔╝
██████╔╝███████╗██║  ██║██║ ╚████╔╝ 
╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝  
  </pre>
  <p><em>First-principles calculus intuition, rendered in real-time.</em></p>
</div>

<hr />

<h2>The Problem We're Solving</h2>
<p>Calculus is traditionally taught through rote memorization of abstract rules (power rule, chain rule) before students ever develop an intuition for what a derivative <em>actually is</em>. Staring at static textbook graphs makes it incredibly difficult to connect a function's changing slope with the amplitude of its derivative.</p>
<p>The <strong>Interactive Derivative Visualizer</strong> bridges that gap. It is a real-time analytical engine designed to teach calculus through visual first principles:</p>
<ul>
  <li><strong>Animates</strong> the connection between <code>f(x)</code> and <code>f'(x)</code> simultaneously.</li>
  <li><strong>Tracks</strong> the tangent line smoothly along the geometric curve.</li>
  <li><strong>Communicates</strong> slope magnitude and direction instantly through dynamic color-coding.</li>
</ul>

<hr />

<h2>Key Features</h2>
<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Real-Time Synchronization</strong></td>
      <td>Moving points on both <code>f(x)</code> and <code>f'(x)</code> plots update frame-by-frame.</td>
    </tr>
    <tr>
      <td><strong>Kinetic Color Mapping</strong></td>
      <td>The tracking point smoothly transitions from blue (negative slope) to red (positive slope).</td>
    </tr>
    <tr>
      <td><strong>Safe Expression Parsing</strong></td>
      <td>Enter custom mathematical functions securely (e.g., <code>sin(x)*exp(-0.1*x)</code>).</td>
    </tr>
    <tr>
      <td><strong>Comprehensive UI Control</strong></td>
      <td>Adjust speed, resolution, domain bounds, and ghost-point trail length on the fly.</td>
    </tr>
    <tr>
      <td><strong>Dual Viewing Modes</strong></td>
      <td>Instantly toggle between Dark Mode (presentation) and Light Mode (print/study).</td>
    </tr>
  </tbody>
</table>

<hr />

<h2 id="architecture">Architecture</h2>
<pre><code>
╔════════════════════════════════════════════════════════════════╗
║                      VISUALIZER PIPELINE                       ║
║                                                                ║
║   ┌───────────────────────────────────────────────────────┐    ║
║   │                   INPUT PARSER                        │    ║
║   │   TextBox Input: "sin(x) * x"                         │    ║
║   │   [Safe Eval Environment via NumPy Dictionary]        │    ║
║   └───────────────────────┬───────────────────────────────┘    ║
║                           │                                    ║
║                           ▼                                    ║
║   ┌───────────────────────────────────────────────────────┐    ║
║   │                 MATH ENGINE (NumPy)                   │    ║
║   │                                                       │    ║
║   │   1. Generate Mesh: xs = np.linspace(xmin, xmax)      │    ║
║   │   2. Evaluate f(x): ys = f(xs)                        │    ║
║   │   3. Gradient f'(x): dys = np.gradient(ys, dx)        │    ║
║   └───────────────────────┬───────────────────────────────┘    ║
║                           │                                    ║
║                           ▼                                    ║
║   ┌───────────────────────────────────────────────────────┐    ║
║   │             MATPLOTLIB ANIMATION LOOP                 │    ║
║   │                                                       │    ║
║   │   Frame i:                                            │    ║
║   │   ├─► Update f(x) point & tangent line                │    ║
║   │   ├─► Update f'(x) point                              │    ║
║   │   ├─► Compute Tanh color mapping for slope            │    ║
║   │   └─► Render Trail Scatter                            │    ║
║   └───────────────────────┬───────────────────────────────┘    ║
║                           │                                    ║
║                           ▼                                    ║
║              [ Interactive GUI / Matplotlib Canvas ]           ║
╚════════════════════════════════════════════════════════════════╝
</code></pre>

<hr />

<h2 id="math-model">Mathematical Model — Dynamic Color Mapping</h2>
<p>To provide an immediate, intuitive sense of the derivative's behavior, the moving point changes color based on the instantaneous slope.</p>
<p>Rather than a hard conditional toggle, the visualizer uses the <strong>Hyperbolic Tangent</strong> to smoothly scale the RGB values. The slope <em>m=f'(x)</em> is mapped to an activation value <em>t</em>:</p>

<p align="center"><code>t = tanh(m)</code></p>

<p>This guarantees <em>t</em> is smoothly bounded between [-1, 1], no matter how steep the slope gets. We then project this into RGB space:</p>

<p align="center"><code>Red = 255 × (t + 1) / 2</code><br/>
<code>Blue = 255 × (1 - (t + 1) / 2)</code></p>

<h3>Intuition</h3>
<ul>
  <li><strong>Steep Positive Slope:</strong> t ≈ 1 → Pure <strong style="color:red;">Crimson Red</strong></li>
  <li><strong>Zero Slope (Min/Max):</strong> t = 0 → Perfect <strong style="color:purple;">Purple</strong> (Equal parts Red/Blue)</li>
  <li><strong>Steep Negative Slope:</strong> t ≈ -1 → Pure <strong style="color:dodgerblue;">Dodger Blue</strong></li>
</ul>
<p>This mapping ensures the user can "feel" the velocity and direction of the function purely through visual color feedback.</p>

<hr />

<h2 id="controls">Interactive Controls</h2>
<p>The application features a fully interactive Matplotlib widget dashboard located at the bottom of the canvas.</p>
<table>
  <thead>
    <tr>
      <th>Control Type</th>
      <th>Variable</th>
      <th>Function</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>TextBox</strong></td>
      <td><code>Function f(x)</code></td>
      <td>Accepts standard math syntax (e.g., <code>tan(x)</code>, <code>x**2</code>, <code>sqrt(abs(x))</code>).</td>
    </tr>
    <tr>
      <td><strong>TextBox</strong></td>
      <td><code>x min, x max</code></td>
      <td>Sets the mathematical domain. Comma-separated (e.g., <code>-6.28, 6.28</code>).</td>
    </tr>
    <tr>
      <td><strong>Slider</strong></td>
      <td><code>Speed (fps)</code></td>
      <td>Adjusts the animation frame rate from 1 to 120 fps.</td>
    </tr>
    <tr>
      <td><strong>Slider</strong></td>
      <td><code>Samples</code></td>
      <td>Graph resolution. Higher = smoother curve, Lower = faster compute.</td>
    </tr>
    <tr>
      <td><strong>Slider</strong></td>
      <td><code>Trail</code></td>
      <td>Number of scatter points to leave behind the moving tangent.</td>
    </tr>
    <tr>
      <td><strong>Button</strong></td>
      <td><code>Play / Pause</code></td>
      <td>Halts or resumes the animation (Shortcut: <strong>Spacebar</strong>).</td>
    </tr>
    <tr>
      <td><strong>Button</strong></td>
      <td><code>Toggle Light/Dark</code></td>
      <td>Switches styling between deep black and clean white backgrounds.</td>
    </tr>
    <tr>
      <td><strong>Keyboard</strong></td>
      <td><code>←</code> / <code>→</code></td>
      <td>Step backwards or forwards frame-by-frame while paused.</td>
    </tr>
  </tbody>
</table>

<hr />

<h2>Supported Mathematical Functions</h2>
<p>The safe evaluation environment supports standard trigonometric and algebraic operations. Use <code>x</code> as your independent variable.</p>
<ul>
  <li><strong>Trigonometric:</strong> <code>sin(x)</code>, <code>cos(x)</code>, <code>tan(x)</code></li>
  <li><strong>Inverse Trig:</strong> <code>arcsin(x)</code>, <code>arccos(x)</code>, <code>arctan(x)</code></li>
  <li><strong>Hyperbolic:</strong> <code>sinh(x)</code>, <code>cosh(x)</code>, <code>tanh(x)</code></li>
  <li><strong>Algebraic:</strong> <code>sqrt(x)</code>, <code>exp(x)</code>, <code>log(x)</code>, <code>abs(x)</code>, <code>x**2</code></li>
  <li><strong>Constants:</strong> <code>pi</code>, <code>e</code></li>
</ul>
<p><em>Try entering:</em> <code>sin(x) * exp(-0.2 * x)</code></p>

<hr />

<h2 id="setup">Setup & Installation</h2>

<p><strong>1. Clone the repository:</strong></p>
<pre><code>git clone https://github.com/Shahrukh-aidev/interactive-derivative-visualizer.git
cd interactive-derivative-visualizer</code></pre>

<p><strong>2. Install dependencies:</strong></p>
<pre><code>pip install numpy matplotlib</code></pre>

<p><strong>3. Run the visualizer:</strong></p>
<pre><code>python deriv_viz.py</code></pre>

<hr />

<h2>License</h2>
<pre><code>Apache License 2.0 — Free to use, modify, and distribute.</code></pre>
