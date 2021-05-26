'''Max or average pooling of 2D or 3D data, vectorized version.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-09 10:11:29.
'''

import numpy as np
from conv3d_vectorized import asStride

def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 4D data.

    Args:
        mat (ndarray): input array to do pooling on the mid 2 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        f (int): pooling kernel size in row/column.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.

    See also unpooling().
    '''
    m, hi, wi, ci = mat.shape
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(hi, stride)
        nx = _ceil(wi, stride)
        size = (m, (ny-1)*stride+f, (nx-1)*stride+f, ci)
        mat_pad = np.full(size, 0)
        mat_pad[:, :hi, :wi, ...] = mat
    else:
        mat_pad = mat[:, :(hi-f)//stride*stride+f, :(wi-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(3, 4), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(3, 4), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result, axis=(3,4))
        return result, pos
    else:
        return result

def unpooling(mat, pos, ori_shape, stride):
    '''Undo a max-pooling of a 4d array to a larger size

    Args:
        mat (ndarray): 4d array to unpool on the mid 2 dimensions.
        pos (ndarray): array recording the locations of maxima in the original
            array. If <mat> is 4d, <pos> is 6d with shape (m, h, w, y, x, c).
            Where h/w are the numbers of rows/columns in <mat>,
            y/x are the sizes of the each pooling window.
            c is the number of channels in <mat>.
        ori_shape (tuple): original shape to unpool to: (m, h2, w2, c).
        stride (int): stride used during the pooling process.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    assert np.ndim(pos)==6, '<pos> should be rank 6.'
    result = np.zeros(ori_shape)
    im, ih, iw, iy, ix, ic = np.where(pos == 1)
    ih2 = ih*stride
    iw2 = iw*stride
    ih2 = ih2+iy
    iw2 = iw2+ix
    values = mat[im, ih, iw, ic].flatten()
    result[im, ih2, iw2, ic] = values

    return result

def unpoolingAvg(mat, f, ori_shape):
    '''Undo an average-pooling of a 4d array to a larger size

    Args:
        mat (ndarray): input array to do unpooling on the mid 2 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        f (int): pooling kernel size in row/column.
        ori_shape (tuple): original shape to unpool to: (m, h2, w2, c).
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    m, hi, wi, ci = ori_shape
    ny = hi//f
    nx = wi//f

    tmp = np.reshape(mat, (m, ny, 1, nx, 1, ci))
    tmp = np.repeat(tmp, f, axis=2)
    tmp = np.repeat(tmp, f, axis=4)
    tmp = np.reshape(tmp, (m, ny*f, nx*f, ci))
    result=np.zeros(ori_shape)

    result[:, :tmp.shape[1], :tmp.shape[2], :] = tmp

    return result



# -------------Main---------------------------------
if __name__ == '__main__':

    n = 10
    f = 3
    s = 2

    np.random.seed(200)
    mat = np.random.randint(0, 20, [n, n])
    print('mat.shape', mat.shape, 'filter size', f, 'stride (if overlap)', s)
    print(mat)

    from pooling import pooling as pool2d
    from pooling import poolingOverlap as pool2dOverlap
    from pooling import unpooling as unpool2d
    from pooling import unpoolingAvg as unpool2dAvg

    p5, p5pos = pool2dOverlap(mat, f, stride=s, pad=False, return_max_pos=True)
    print('Overlap pooling, no pad. Result shape:', p5.shape)
    print(p5)

    p5unp = unpool2d(p5, p5pos, mat.shape, s)

    p6 = pool2d(mat, f, pad=False, method='mean')
    print('non-overlap pooling, no pad. Result shape:', p6.shape)
    print(p6)

    p6unp = unpool2dAvg(p6, f, mat.shape)

    #---------------------4D tests---------------------
    mat3 = np.repeat(mat[:, :, None], 5, axis=2)
    mat4 = np.array([mat3, mat3*10])
    p54, p54pos = poolingOverlap(mat4, f, stride=s, pad=False, return_max_pos=True)
    print('4D data, overlap pooling, no pad. Result shape:', p54.shape)
    print('p54[0,:,:,0] == p5 ?', np.allclose(p54[0, :, :, 0], p5))

    p54unp = unpooling(p54, p54pos, mat4.shape, s)
    print('p54unp[0,:,:,0] == p5unp ?', np.allclose(p54unp[0, :, :, 0], p5unp))
    print('p54unp[0,:,:,0] == p5unp ?', np.allclose(p54unp[0, :, :, 1], p5unp))
    print('p54unp[1,:,:,0] == 10*p5unp ?', np.allclose(p54unp[1, :, :, 4], 10*p5unp))

    p64 = poolingOverlap(mat4, f, f, method='mean', pad=False)
    p64unp = unpoolingAvg(p64, f, mat4.shape)
    print('p64unp[0,:,:,0] == p6unp ?', np.allclose(p64unp[0, :, :, 0], p6unp))

