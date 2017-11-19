# Differentiable Spatial to Numerical Transform
A Tensorflow implementation of the DSNT layer, as taken from the paper "Numerical Coordinate Regression with Convolutional Neural Networks".

### Provided Files:

- `dsnt.py` - The layer implementation and its supporting functions
- `DSNT_sample.ipynb` - A notebook demonstrating the usage of the DSNT layer.



### Example usage:
The layer can be inserted at the end of a stack of convolutional layers, where the final tensor shape is `[batch, height, width, 1]`.


Begin by importing the module:
```
import dsnt
```

The tensor must then be first rectified/normalised using one of the supported methods:
```
my_tensor = dsnt.normalise_heatmap(my_tensor)
```

This normalised heatmap is then used to produce a tensor of shape `[batch, x, y]` by passing it through a DSNT layer:
```
coords = dsnt.dsnt_layer(my_tensor)
```

The loss function must be composed of two components. Mean-Squared-Error for the coordinate regression, and Jensen-Shannon Divergence for regularization.
```
# Coordinate regression loss
loss_1 = tf.losses.mean_squared_error(targets, predictions)
# Regularization loss - in this example the targets are in range [0, 1], 
# but need to be in range [-1, 1] for the regularization loss
loss_2, target_gauss = dsnt.js_reg_loss(heatmaps, (targets + 1) / 2)

loss = loss_1 + loss_2
```
