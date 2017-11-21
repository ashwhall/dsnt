'''
A Sonnet wrapper for the DSNT layer, as taken from the paper "Numerical Coordinate
Regression with Convolutional Neural Networks"
'''

import tensorflow as tf
import sonnet as snt
import dsnt

import sonnet as snt

class DSNT(snt.AbstractModule):
    '''
    A Sonnet wrapper for the DSNT layer, as taken from the paper "Numerical Coordinate
    Regression with Convolutional Neural Networks"
    '''
    def __init__(self, name='DSNT', method='softmax'):
        '''
        DSNT module constructor
        Arguments:
            name - The name of the module
            method - The desired rectification method (see dsnt.py for options)
        '''
        super(DSNT, self).__init__(name=name)
        self._method = 'softmax'

    def _build(self, inputs):
        '''
        Builds the transformation operations
        Arguments:
            inputs - The input volume on which to apply the operation
        Returns:
            norm_heatmap - The given heatmap with normalisation/rectification applied
            coords - A tensor of shape [batch, 2] containing the [x, y] coordinate pairs
        '''
        norm_heatmap, coords = dsnt.dsnt(inputs, self._method)
        return norm_heatmap, coords
    
