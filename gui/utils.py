from PIL import Image


def dim_to_2d(dimension):

    if dimension == 1:
        return (1, 1)

    elif dimension == 3:
        return (3, 1)

    elif dimension == 10:
        return (1, 10)

    elif dimension == 16:
        return (2, 8)

    elif dimension == 32:
        return (4, 8)

    elif dimension == 64:
        return (4, 16)

    elif dimension == 128:
        return (8, 16)

    elif dimension == 256:
        return (8, 32)

    elif dimension == 512:
        return (16, 32)

    elif dimension == 1024:
        return (16, 64)

    elif dimension == 2048:
        return (32, 64)

    elif dimension == 4096:
        return (32, 128)

    elif dimension == 8192:
        return (64, 128)

    elif dimension % 10 == 0:
        H = 10
        W = dimension // 10
        return (H, W)

    else:
        raise NotImplementedError('dimension should take one of the values 1, 10, 32, 64, 128, 256, 512, 1024 or 2048 but got {}'.format(dimension))


def to_4d(layer_shape):

    if len(layer_shape) == 1:
        dimension = layer_shape[0]
        H1, W1 = dim_to_2d(dimension)
        return H1, W1, 1, 1

    elif len(layer_shape) == 3:
        dimension = layer_shape[0]
        H1, W1 = dim_to_2d(dimension)
        H, W = layer_shape[1:]
        return H1, W1, H, W
    else:
        raise NotImplementedError('layer_shape should be a tensor with either 1 or 3 dimensions but got {}'.format(len(layer_shape)))


def to_2d(layer_shape):

    if len(layer_shape) == 1:
        dimension = layer_shape[0]
        return dim_to_2d(dimension)

    elif len(layer_shape) == 3:
        dimension = layer_shape[0]
        H1, W1 = dim_to_2d(dimension)
        H, W = layer_shape[1:]
        if H1 == 3 and W1 == 1:
            return 3, H, W
        else:
            return H1 * H, W1 * W
    else:
        raise NotImplementedError('layer_shape should be a tensor with either 1 or 3 dimensions but got {}'.format(len(layer_shape)))


def to_shape_image(shape_2d):
    if len(shape_2d)==2:
        H, W = shape_2d
        if H == 1 and W == 10:
            return (250, 25)
        else:
            return W * 5, H * 5
    else:
        D, H, W = shape_2d
        return W * 5, H * 5

def aux(array):
    if len(array.shape)==2:
        return Image.fromarray(array)
    else:
        return Image.fromarray(array, mode='RGB')
