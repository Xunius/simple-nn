'''Convolution of 2D or 3D data using np or scipy, vectorized version.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-26 08:46:34.
'''

import numpy as np
from conv3d import compSize


def padArray(var, pad1, pad2=None):
    '''Pad array with 0s

    Args:
        var (ndarray): 4d ndarray. Padding is done on the mid 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 4d ndarray with 0s padded along the mid 2 dimensions.
    '''

    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var

    var_pad = np.zeros((var.shape[0], var.shape[1] + pad1 + pad2, var.shape[2] + pad1 + pad2,
        var.shape[3]))
    var_pad[:, pad1:-pad2, pad1:-pad2, :] = var

    return var_pad

def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.

    Args:
        arr (ndarray): input array of rank 4, with shape (m, hi, wi, ci).
        sub_shape (tuple): window size: (f1, f2).
        stride (int): stride of windows in both 2nd and 3rd dimensions.
    Returns:
        subs (view): strided window view.

    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape (m, hi, wi, ci), and is transformed
    to a strided view with shape (m, ho, wo, f, f, ci). where:
        m: number of records.
        hi, wi: height and width of input image.
        ci: channels of input image.
        f: kernel size.
    The convolution kernel has shape (f, f, ci, co).
    Then the vectorized 3d convolution can be achieved using either an einsum()
    or a tensordot():

        conv = np.einsum('myxfgc,fgcz->myxz', arr_view, kernel)
        conv = np.tensordot(arr_view, kernel, axes=([3, 4, 5], [0, 1, 2]))

    See also skimage.util.shape.view_as_windows()
    '''

    sm, sh, sw, sc = arr.strides
    m, hi, wi, ci = arr.shape
    f1, f2 = sub_shape

    view_shape = (m, 1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2, ci)
    strides = (sm, stride*sh, stride*sw, sh, sw, sc)

    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)

    return subs

def conv3D3(var, kernel, stride=1, pad=0):
    '''3D convolution by strided view.

    Args:
        var (ndarray): 4d array to convolve along the last 3 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        kernel (ndarray): 4d filter to convolve with. Shape is (fy, fx, ci, co).
            where fy, fx: filter size. co: number of filters.
    Keyword Args:
        stride (int): stride along the mid 2 dimensions. Default to 1.
        pad (int): number of columns/rows to pad at edges.
    Returns:
        conv (ndarray): convolution result.
    '''
    if np.ndim(var) != 4:
        raise Exception("<var> dimension should be 4.")
    if np.ndim(kernel) != 4:
        raise Exception("<kernel> dimension should be 4.")
    if pad > 0:
        var_pad = padArray(var, pad, pad)
    else:
        var_pad = var
    view = asStride(var_pad, kernel.shape[:2], stride)
    #conv = np.einsum('myxfgz,fgzo->myxo', view,kernel)
    conv = np.tensordot(view, kernel, axes=([3,4,5],[0,1,2]))

    return conv

def conv3Dgrad(act, delta):
    '''Compute gradients of convolution layer filters

    Args:
        act (ndarray): activation array as input to the filters. With shape
            (m, hi, wi, ci).  Where m: number of records.
            hi, wi: height and width of the input into the filters.
            ci: channels of input into the filters.
        delta (ndarray): error term as output from the filters. With shape
            (m, ho, wo, co): ho, wo: height and width of the output from the filters.
            co: number of filters in the convolution layer.
    Returns:
        conv (ndarray): gradients of filters, defined as:

            \partial J / \partial W^{(l)} = \sum[ a^{(l-1)} \bigotimes \delta^{(l)}]

        NOTE that the gradients are summed across the m records in <act> and
        <delta>.
    '''
    m, hi, wi, ci = act.shape
    m, ho, wo, co = delta.shape
    view = asStride(act, (ho, wo), stride=1)
    #conv = np.einsum('myxfgz,mfgo->yxzo', view, delta)
    conv = np.tensordot(view, delta, axes=([0, 3, 4], [0, 1, 2]))
    return conv

def interLeave(arr, sy, sx):
    '''Interleave array with rows/columns of 0s.

    Args:
        arr (ndarray): 4d array to interleave in the mid 2 dimensions.
        sy (int): number of rows to interleave.
        sx (int): number of columns to interleave.
    Returns:
        result (ndarray): input <arr> array interleaved with 0s.

    E.g.
        arr[0,:,:,0] = [[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]
        interLeave(arr, 1, 2)[0,:,:,0] ->

        [[1, 0, 0, 2, 0, 0, 3],
         [0, 0, 0, 0, 0, 0, 0],
         [4, 0, 0, 5, 0, 0, 6],
         [0, 0, 0, 0, 0, 0, 0],
         [7, 0, 0, 8, 0, 0, 9]]
    '''

    m, hi, wi, ci = arr.shape
    shape = (m, hi+sy*(hi-1), wi+sx*(wi-1), ci)
    result = np.zeros(shape)
    result[:, 0::(sy+1), 0::(sx+1), :] = arr
    return result

def fullConv3D(var, kernel, stride):
    '''Full mode 3D convolution using stride view.

    Args:
        var (ndarray): 4d array to convolve along the last 3 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        kernel (ndarray): 4d filter to convolve with. Shape is (f1, f2, ci, co).
            where f1, f2: filter size. co: number of filters.
    Keyword Args:
        stride (int): stride along the mid 2 dimensions. Default to 1.
    Returns:
        conv (ndarray): convolution result.

    Note that the kernel is not filpped inside this function.
    '''
    if np.ndim(var) != 4:
        raise Exception("<var> dimension should be 4.")
    if np.ndim(kernel) != 4:
        raise Exception("<kernel> dimension should be 4.")
    stride = int(stride)
    m, hi, wi, ci = var.shape
    f1, f2, ci, co = kernel.shape

    # interleave 0s
    var2 = interLeave(var, stride-1, stride-1)
    # pad boundaries
    nout, pad_left, pad_right = compSize(hi, f1, stride)
    var2 = padArray(var2, pad_left, pad_right)
    # transpose kernel
    kernel = np.transpose(kernel, [0, 1, 3, 2])
    # convolve
    conv = conv3D3(var2, kernel, stride=1, pad=0)

    return conv
