# An introduction to the Lambert W function

Code for the Medium article *An introduction to the Lambert W function: and how to numerically compute it from scratch*.

`lambert_w_intro.py` reproduces every figure, table and number in the article:

- Figures 1 to 5 and Tables 1 and 2, saved as PNG files
- all iteration counts, final values and errors quoted in "Comparing the methods", printed as text

The three methods (fixed-point iteration, Newton's method, Halley's method) are written from scratch with NumPy. SciPy's `scipy.special.lambertw` is used only as a reference value.

Run:

```
python lambert_w_intro.py
```

Needs `numpy`, `scipy` and `matplotlib`.
