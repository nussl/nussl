from torch import nn

def _compute_padding(conv_layer, input_shape):
    """Since PyTorch doesn't have a SAME padding functionality, this is a function
    that replicates that. 

    Padding formula:
        o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    Solving for p:
        p = [s * (o - 1) - i + k + (k-1)*(d-1)] / 2
    
    Arguments:
        conv_layer
        input_shape {[type]} -- [description]
    Returns:
        A ConstantPad2D layer with 0s such that when passed through the convolution,
        the input and output shapes stay identical.
    """
    padding_key = str(conv_layer) + str(input_shape)
    dilation = conv_layer.dilation
    stride = conv_layer.stride
    padding = conv_layer.padding
    kernel_size = conv_layer.kernel_size
    input_shape = input_shape[-2:]
    output_size = input_shape

    computed_padding = []
    for dim in range(len(kernel_size)):
        _pad = (
            stride[dim] * (output_size[dim] - 1) - input_shape[dim] + 
            kernel_size[dim] + (kernel_size[dim] - 1) * (dilation[dim] - 1)
        )
        _pad = int(_pad / 2 + _pad % 2)
        computed_padding += [_pad, _pad]
    computed_padding = tuple(computed_padding)
    return nn.ConstantPad2d(computed_padding, 0.0)