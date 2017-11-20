# Differentiable Spatial to Numerical Transform
A Tensorflow implementation of the DSNT layer, as taken from the paper "Numerical Coordinate Regression with Convolutional Neural Networks".

### Provided Files:

- `dsnt.py` - The layer implementation and its supporting functions
- `DSNT_sample.ipynb` - A notebook demonstrating the usage of the DSNT layer.



### Example usage:
Begin by importing the module:
```
import dsnt
```

The layer can be inserted at the end of a stack of convolutional layers, where the final tensor shape is `[batch, height, width, 1]`.
The function's input tensor will be rectified, then passed through the transform. `dsnt.dsnt` returns the rectified input heatmaps and the produced coordinates tensor of shape `[batch, x, y]`:
```
norm_heatmaps, coords = dsnt.dsnt(my_tensor)
```
There are different rectification methods available, which can be provided as an additional argument, e.g: `dsnt.dsnt(my_tensor, 'relu')`


The loss function must be composed of two components. Mean-Squared-Error or similar for the coordinate regression, and Jensen-Shannon Divergence for regularization.
```
# Coordinate regression loss
loss_1 = tf.losses.mean_squared_error(input_y, predictions)
# Regularization loss
loss_2 = dsnt.js_reg_loss(heatmaps, input_y)

loss = loss_1 + loss_2
```