'''Convolution of 2D or 3D data using np or scipy.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-26 08:46:34.
'''

import numpy as np
from scipy.signal import fftconvolve

def padArray(var, pad1, pad2=None):
    '''Pad array with 0s

    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var

    return var_pad

def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.

    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]

    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs

def conv3D3(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.

    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var

    view = asStride(var_pad, kernel.shape, stride)
    if np.ndim(kernel) == 2:
        conv = np.sum(view*kernel, axis=(2, 3))
    else:
        conv = np.sum(view*kernel, axis=(2, 3, 4))

    return conv

def checkShape(var, kernel):
    '''Check shapes for convolution

    Args:
        var (ndarray): 2d or 3d input array for convolution.
        kernel (ndarray): 2d or 3d convolution kernel.
    Returns:
        kernel (ndarray): 2d kernel reshape into 3d if needed.
    '''
    var_ndim = np.ndim(var)
    kernel_ndim = np.ndim(kernel)

    if var_ndim not in [2, 3]:
        raise Exception("<var> dimension should be in 2 or 3.")
    if kernel_ndim not in [2, 3]:
        raise Exception("<kernel> dimension should be in 2 or 3.")
    if var_ndim < kernel_ndim:
        raise Exception("<kernel> dimension > <var>.")
    if var_ndim == 3 and kernel_ndim == 2:
        kernel = np.repeat(kernel[:, :, None], var.shape[2], axis=2)

    return kernel

def pickStrided(var, stride):
    '''Pick sub-array by stride

    Args:
        var (ndarray): 2d or 3d ndarray.
        stride (int): stride/step along the 1st 2 dimensions to pick
            elements from <var>.
    Returns:
        result (ndarray): 2d or 3d ndarray picked at <stride> from <var>.
    '''
    if stride < 0:
        raise Exception("<stride> should be >=1.")
    if stride == 1:
        result = var
    else:
        result = var[::stride, ::stride, ...]
    return result

def conv3D2(var, kernel, stride=1, pad=0):
    '''3D convolution by sub-matrix summing.

    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        result (ndarray): convolution result.
    '''
    var_ndim = np.ndim(var)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    result = 0
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var

    for ii in range(ky*kx):
        yi, xi = divmod(ii, kx)
        slabii = var_pad[yi:2*pad+ny-ky+yi+1:1,
                         xi:2*pad+nx-kx+xi+1:1, ...]*kernel[yi, xi]
        if var_ndim == 3:
            slabii = slabii.sum(axis=-1)
        result += slabii

    if stride > 1:
        result = pickStrided(result, stride)

    return result

def conv3D(var, kernel, stride=1, pad=0):
    '''3D convolution using scipy.signal.fftconvolve.

    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    stride = int(stride)
    kernel = checkShape(var, kernel)
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var

    conv = fftconvolve(var_pad, kernel, mode='valid')

    if stride > 1:
        conv = pickStrided(conv, stride)

    return conv

def interLeave(arr, sy, sx):
    '''Interleave array with rows/columns of 0s.

    Args:
        arr (ndarray): input 2d or 3d array to interleave in the first 2 dimensions.
        sy (int): number of rows to interleave.
        sx (int): number of columns to interleave.
    Returns:
        result (ndarray): input <arr> array interleaved with 0s.

    E.g.
        arr = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]
        interLeave(arr, 1, 2) ->

        [[1, 0, 0, 2, 0, 0, 3],
         [0, 0, 0, 0, 0, 0, 0],
         [4, 0, 0, 5, 0, 0, 6],
         [0, 0, 0, 0, 0, 0, 0],
         [7, 0, 0, 8, 0, 0, 9]]
    '''

    ny, nx = arr.shape[:2]
    shape = (ny+sy*(ny-1), nx+sx*(nx-1))+arr.shape[2:]
    result = np.zeros(shape)
    result[0::(sy+1), 0::(sx+1), ...] = arr
    return result

def compSize(n, f, s):
    '''Compute the shape of a full convolution result

    Args:
        n (int): length of input array x.
        f (int): length of kernel.
        s (int): stride.
    Returns:
        nout (int): lenght of output array y.
        pad_left (int): number padded to the left in a full convolution.
        pad_right (int): number padded to the right in a full convolution.

    E.g. x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    f = 3, s = 2.
    A full convolution is done on [*, *, 0], [0, 1, 2], [2, 3, 4], ..., [6, 7, 8],
    [9, 10, *]. Where * is missing outside of the input domain.
    Therefore, the full convolution y has length 6. pad_left = 2, pad_right = 1.
    '''

    nout = 1
    pad_left = f-1
    pad_right = 0
    idx = 0   # index of the right end of the kernel
    while True:
        idx_next = idx+s
        win_left = idx_next-f+1
        if win_left <= n-1:
            nout += 1
            idx = idx+s
        else:
            break
    pad_right = idx-n+1

    return nout, pad_left, pad_right

def fullConv3D(var, kernel, stride):
    '''Full mode 3D convolution using stride view.

    Args:
        var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
        kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
            is 2d, create a dummy dimension to be the 3rd dimension in kernel.
    Keyword Args:
        stride (int): stride along the 1st 2 dimensions. Default to 1.
    Returns:
        conv (ndarray): convolution result.

    Note that the kernel is not filpped inside this function.
    '''
    stride = int(stride)
    ny, nx = var.shape[:2]
    ky, kx = kernel.shape[:2]
    # interleave 0s
    var2 = interLeave(var, stride-1, stride-1)
    # pad boundaries
    nout, pad_left, pad_right = compSize(ny, ky, stride)
    var2 = padArray(var2, pad_left, pad_right)
    # convolve
    conv = conv3D3(var2, kernel, stride=1, pad=0)

    return conv

