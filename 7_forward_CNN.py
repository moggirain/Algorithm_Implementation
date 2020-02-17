import numpy as np


class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        kernel_size (tuple): the shape of the kernel. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, kernel_height, kernel_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            out_channels).
        kernel_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, kernel_size, stride, padding):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        h_out = np.floor((x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1).astype(np.uint8)
        print("h_out", h_out.dtype)
        w_out = np.floor((x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1).astype(np.uint8)
        print("h_out", w_out.dtype)
        out = (self.kernel_size[0], h_out, w_out)
        # initiate an output volume with zeros
        output = np.zeros(out)
        for c in range(out[0]): # channel
            for h in range(h_out):  # height
                for w in range(w_out):  # width
                    h_start = h * s[0]
                    h_end = h_start + self.kernel_size[2]
                    w_start = w * s[1]
                    w_end = w_start + self.kernel_size[3]
                    output[c, h, w] = np.sum(x_padded[:, h_start: h_end, w_start : w_end] * self.W[c,:,:,:]) + self.b[c]
        print("output", output, output.dtype)
        return output




class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        h_out = np.floor((x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1).astype(np.uint8)
        w_out = np.floor((x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1).astype(np.uint8)
        out = (x_padded.shape[0], h_out, w_out)
        # initiate an output volume with zeros
        maxpool  = np.zeros(out)
        for c in range(out[0]): # channel
            for h in range(h_out):  # height
                for w in range(w_out):  # width
                    h_start = h * s[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = w * s[1]
                    w_end = w_start + self.kernel_size[1]
                    maxpool[c, h, w] = np.max(x_padded[c, h_start: h_end, w_start : w_end]) 
        return maxpool



class AvgPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        h_out = np.floor((x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1).astype(np.uint8)
        w_out = np.floor((x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1).astype(np.uint8)
        out = (x_padded.shape[0], h_out, w_out)
        # initiate an output volume with zeros
        avgpool  = np.zeros(out)
        for c in range(out[0]): # channel
            for h in range(h_out):  # height
                for w in range(w_out):  # width
                    h_start = h * s[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = w * s[1]
                    w_end = w_start + self.kernel_size[1]
                    avgpool[c, h, w] = np.average(x_padded[c, h_start: h_end, w_start : w_end]) 
        return avgpool
