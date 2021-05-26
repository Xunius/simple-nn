'''Max or average pooling of 2D or 3D data.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2021-05-09 10:11:29.
'''

import numpy as np
from conv3d import asStride

def pooling(mat, f, method='max', pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
    Returns:
        result (ndarray): pooled array.
    '''
    m, n = mat.shape[:2]
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, f)
        nx = _ceil(n, f)
        size = (ny*f, nx*f)+mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m//f
        nx = n//f
        mat_pad = mat[:ny*f, :nx*f, ...]

    new_shape = (ny, f, nx, f)+mat.shape[2:]

    if method == 'max':
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result

def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.

    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
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
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1

    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]

    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)

    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result

def unpooling(mat, pos, ori_shape, stride):
    '''Undo a max-pooling of a 2d or 3d array to a larger size

    Args:
        mat (ndarray): 2d or 3d array to unpool on the first 2 dimensions.
        pos (ndarray): array recording the locations of maxima in the original
            array. If <mat> is 2d, <pos> is 4d with shape (iy, ix, cy, cx).
            Where iy/ix are the numbers of rows/columns in <mat>,
            cy/cx are the sizes of the each pooling window.
            If <mat> is 3d, <pos> is 5d with shape (iy, ix, cy, cx, cc).
            Where cc is the number of channels in <mat>.
        ori_shape (tuple): original shape to unpool to.
        stride (int): stride used during the pooling process.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    assert np.ndim(pos) in [4, 5], '<pos> should be rank 4 or 5.'
    result = np.zeros(ori_shape)

    if np.ndim(pos) == 5:
        iy, ix, cy, cx, cc = np.where(pos == 1)
        iy2 = iy*stride
        ix2 = ix*stride
        iy2 = iy2+cy
        ix2 = ix2+cx
        values = mat[iy, ix, cc].flatten()
        result[iy2, ix2, cc] = values
    else:
        iy, ix, cy, cx = np.where(pos == 1)
        iy2 = iy*stride
        ix2 = ix*stride
        iy2 = iy2+cy
        ix2 = ix2+cx
        values = mat[iy, ix].flatten()
        result[iy2, ix2] = values

    return result

def unpoolingAvg(mat, f, ori_shape):
    '''Undo an average-pooling of a 2d or 3d array to a larger size

    Args:
        mat (ndarray): 2d or 3d array to unpool on the first 2 dimensions.
        f (int): pooling kernel size.
        ori_shape (tuple): original shape to unpool to.
    Returns:
        result (ndarray): <mat> unpoolled to shape <ori_shape>.
    '''
    m, n = ori_shape[:2]
    ny = m//f
    nx = n//f

    tmp = np.reshape(mat, (ny, 1, nx, 1)+mat.shape[2:])
    tmp = np.repeat(tmp, f, axis=1)
    tmp = np.repeat(tmp, f, axis=3)
    tmp = np.reshape(tmp, (ny*f, nx*f)+mat.shape[2:])
    result=np.zeros(ori_shape)

    result[:tmp.shape[0], :tmp.shape[1], ...]= tmp

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

    p1 = pooling(mat, f, pad=True)
    print('non-overlap pooling, pad. Result shape:', p1.shape)
    print(p1)

    p2 = pooling(mat, f, pad=False)
    print('non-overlap pooling, no pad. Result shape:', p2.shape)
    print(p2)

    p3 = poolingOverlap(mat, f, stride=s, pad=True)
    print('Overlap pooling, pad. Result shape:', p3.shape)
    print(p3)

    p4 = poolingOverlap(mat, f, stride=s, pad=False)
    print('Overlap pooling, no pad. Result shape:', p4.shape)
    print(p4)

    p5, p5pos = poolingOverlap(
        mat, f,
        stride=s,
        pad=False, return_max_pos=True)
    print('Overlap pooling, no pad. Result shape:', p5.shape)
    print(p5)

    p5unp = unpooling(p5, p5pos, mat.shape, s)

    p6 = pooling(mat, f, pad=False, method='mean')
    print('non-overlap pooling, no pad. Result shape:', p6.shape)
    print(p6)

    p6unp = unpoolingAvg(p6, f, mat.shape)

    # --------------------Make it 3D--------------------
    mat3 = np.repeat(mat[:, :, None], 5, axis=2)

    p13 = pooling(mat3, f, pad=True)
    print('3D data, non-overlap pooling, pad. Result shape:', p13.shape)
    print('p13[:,:,0] == p1 ?', np.allclose(p13[:, :, 0], p1))

    p23 = pooling(mat3, f, pad=False)
    print('3D data, non-overlap pooling, no pad. Result shape:', p23.shape)
    print('p23[:,:,0] == p2 ?', np.allclose(p23[:, :, 0], p2))

    p33 = poolingOverlap(mat3, f, stride=s, pad=True)
    print('3D data, overlap pooling, pad. Result shape:', p33.shape)
    print('p33[:,:,0] == p3 ?', np.allclose(p33[:, :, 0], p3))

    p43 = poolingOverlap(mat3, f, stride=s, pad=False)
    print('3D data, overlap pooling, no pad. Result shape:', p43.shape)
    print('p43[:,:,0] == p4 ?', np.allclose(p43[:, :, 0], p4))

    p53, p53pos = poolingOverlap(
        mat3, f,
        stride=s,
        pad=False, return_max_pos=True)
    print('3D data, overlap pooling, no pad. Result shape:', p53.shape)
    print('p53[:,:,0] == p5 ?', np.allclose(p53[:, :, 0], p5))

    p53unp = unpooling(p53, p53pos, mat3.shape, s)
    print('p53unp[:,:,0] == p5unp ?', np.allclose(p53unp[:, :, 0], p5unp))
    print('p53unp[:,:,0] == p5unp ?', np.allclose(p53unp[:, :, 1], p5unp))
    print('p53unp[:,:,0] == p5unp ?', np.allclose(p53unp[:, :, 4], p5unp))

    p63 = pooling(mat3, f, pad=False, method='mean')
    print('3D data, non-overlap pooling, no pad. Result shape:', p63.shape)
    print('p63[:,:,0] == p6 ?', np.allclose(p63[:, :, 0], p6))

    p63unp = unpoolingAvg(p63, f, mat3.shape)
    print('p63unp[:,:,0] == p6unp ?', np.allclose(p63unp[:, :, 0], p6unp))
    print('p63unp[:,:,3] == p6unp ?', np.allclose(p63unp[:, :, 3], p6unp))

