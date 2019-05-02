# Differentiable Spatial to Numerical Transform
An unofficial Tensorflow implementation of differentiable spatial to numerical (DSNT) layer.

Code in this project implements ideas presented in the research paper [""Numerical Coordinate Regression with Convolutional Neural Networks" by Nibali et al"](https://arxiv.org/abs/1801.07372). If you use it in your own research project, please be sure to cite the original paper appropriately.

Also included is a small [Sonnet](https://github.com/deepmind/sonnet) module wrapper around the DSNT layer.

## Provided Files:

- `dsnt.py` - The layer implementation and its supporting functions
- `dsnt_snt.py` - A Sonnet module wrapping the layer
- `DSNT_sample.ipynb` - A notebook demonstrating the usage of the DSNT layer.


## Example usage:
The instructions vary slightly depending on if Sonnet or raw Tensorflow is used.
### Begin by importing the module:

**Raw Tensorflow**:
```
import dsnt
```
**Sonnet**:
```
import dsnt
from dsnt_snt import DSNT
```

### Insert the layer
The layer can be inserted at the end of a stack of convolutional layers, where the final tensor shape is `[batch, height, width, 1]`.
The function's input tensor will be rectified, then passed through the transform. `dsnt.dsnt` returns the rectified input heatmaps and the produced coordinates tensor of shape `[batch, x, y]`:

**Raw Tensorflow**
```
norm_heatmaps, coords = dsnt.dsnt(my_tensor)
```
**Sonnet**:
```
norm_heatmaps, coords = DSNT()(my_tensor)
```

There are different rectification methods available, which can be provided as an additional argument, e.g: `dsnt.dsnt(my_tensor, method='relu')`


### Add the loss terms
The loss function must be composed of two components. Mean-Squared-Error or similar for the coordinate regression, and Jensen-Shannon Divergence for regularization.
```
# Coordinate regression loss
loss_1 = tf.losses.mean_squared_error(targets, coords)
# Regularization loss
loss_2 = dsnt.js_reg_loss(norm_heatmaps, targets)

loss = loss_1 + loss_2
```
You can specify the size of the Gaussian used for regularization by passing an additional argument to the loss function, e.g: `dsnt.js_reg_loss(norm_heatmaps, targets, fwhm=3)`. This argument is the [Full Width at Half Maximum](https://en.wikipedia.org/wiki/Full_width_at_half_maximum), which can be thought of as the radius of the drawn heatmap.
