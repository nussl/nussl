# -*- coding: utf-8 -*-
"""
Copyright (c) 2015, Antoine Liutkus, Inria
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""

import numpy as np
import itertools


def splitinfo(sigShape,frameShape,hop):
    
    # making sure input shapes are tuples, not simple integers
    if np.isscalar(frameShape):
        frameShape = (frameShape,)
    if np.isscalar(hop):
        hop = (hop,)

    # converting frameShape to array, and building an aligned frameshape, which is 1 whenever the frame
    # dimension is not given. For instance, if frameShape=(1024,) and sigShape=(10000,2),
    # frameShapeAligned is set to (1024,1)
    frameShape = np.array(frameShape)
    fdim = len(frameShape)
    frameShapeAligned = np.append(
        frameShape, np.ones(
            (len(sigShape) - len(frameShape)))).astype(int)

    # same thing for hop
    hop = np.array(hop)
    hop = np.append(hop, np.ones((len(sigShape) - len(hop)))).astype(int)

    # building the positions of the frames. For each dimension, gridding from
    # 0 to sigShape[dim] every hop[dim]
    framesPos = np.ogrid[[slice(0, size, step)
                          for (size, step) in zip(sigShape, hop)]]

    # number of dimensions
    nDim = len(framesPos)

    # now making sure we have at most one frame going out of the signal. This is possible, for instance if
    # the overlap is very large between the frames
    for dim in range(nDim):
        # for each dimension, we remove all frames that go beyond the signal
        framesPos[dim] = framesPos[dim][
            np.nonzero(
                np.add(
                    framesPos[dim],
                    frameShapeAligned[dim]) < sigShape[dim])]
        # are there frames positions left in this dimension ?
        if len(framesPos[dim]):
            # yes. we then add a last frame (the one going beyond the signal), if it is possible. (it may NOT be
            # possible in some exotic cases such as hopSize[dim]>1 and
            # frameShapeAligned[dim]==1
            if framesPos[dim][-1] + hop[dim] < sigShape[dim]:
                framesPos[dim] = np.append(
                    framesPos[dim], framesPos[dim][-1] + hop[dim])
        else:
            # if there is no more frames in this dimension (short signal in
            # this dimension), then at least consider 0
            framesPos[dim] = [0]

    # constructing the shape of the framed signal
    framedShape = np.append(frameShape, [len(x) for x in framesPos])
    return (framesPos,framedShape,frameShape,hop,fdim,nDim,frameShapeAligned)

    
def split(sig, frameShape, hop, weightFrames=True, verbose=False):
    """splits a ndarray into overlapping frames
    sig : ndarray
    frameShape : tuple giving the size of each frame. If its shape is
                 smaller than that of sig, assume the frame is of size 1
                 for all missing dimensions
    hop : tuple giving the hopsize in each dimension. If its shape is
          smaller than that of sig, assume the hopsize is 1 for all
          missing dimensions
    weightFrames : return frames weighted by a ND hamming window
    verbose : whether to output progress during computation"""

    # signal shape
    sigShape = np.array(sig.shape)
    
    (framesPos,framedShape,frameShape,hop,fdim,nDim,frameShapeAligned) = splitinfo(sigShape,frameShape,hop)

    if weightFrames:
        # constructing the weighting window. Choosing hamming for convenience
        # (never 0)
        win = 1
        for dim in range(len(frameShape) - 1, -1, -1):
            win = np.outer(np.hamming(frameShapeAligned[dim]), win)
        win = np.squeeze(win)

    # alocating memory for framed signal
    framed = np.zeros(framedShape, dtype=sig.dtype)

    # total number of frames (for displaying)
    nFrames = np.prod([len(x) for x in framesPos])

    # for each frame
    for iframe, index in enumerate(itertools.product(*[range(len(x)) for x in framesPos])):
        # display from time to time if asked for
        if verbose and (not iframe % 100):
            print 'Splitting : frame ' + str(iframe) + '/' + str(nFrames)

        # build the slice to use for extracting the signal of this frame.
        frameRange = [Ellipsis]
        for dim in range(nDim):
            frameRange += [slice(framesPos[dim][index[dim]],
                                 min(sigShape[dim],
                                     framesPos[dim][index[dim]] + frameShapeAligned[dim]),
                                 1)]

        # extract the signal
        sigFrame = sig[frameRange]
        sigFrame.shape = sigFrame.shape[:fdim]

        # the signal may be shorter than the normal size of a frame (at the end of the signal). We build a slice that corresponds to
        # the actual size we got here
        sigFrameRange = [slice(0, x, 1) for x in sigFrame.shape[:fdim]]

        # puts the signal in the output variable
        framed[sigFrameRange + list(index)] = sigFrame

        if weightFrames:
            # multiply by the weighting window
            framed[[Ellipsis] + list(index)] *= win

    frameShape = [int(x) for x in frameShape]
    return framed


def overlapadd(S, fdim, hop, shape=None, weightedFrames=True, verbose=False):
    """n-dimensional overlap-add
    S    : ndarray containing the stft to be inverted
    fdim : the number of dimensions in S corresponding to
           frame indices.
    hop  : tuple containing hopsizes along dimensions.
           Missing hopsizes are assumed to be 1
    shape: Indicating the original shape of the
           signal for truncating. If None: no truncating is done
    weightedFrames: True if we need to compensate for the analysis weighting
                    (weightFrames of the split function)
    verbose: whether or not to display progress
            """

    # number of dimensions
    nDim = len(S.shape)

    frameShape = S.shape[:fdim]
    trueFrameShape = np.append(
        frameShape,
        np.ones(
            (nDim - len(frameShape)))).astype(int)

    # same thing for hop
    if np.isscalar(hop):
        hop = (hop,)
    hop = np.array(hop)
    hop = np.append(hop, np.ones((nDim - len(hop)))).astype(int)

    sigShape = [
        (nframedim - 
         1) * 
        hopdim + 
        frameshapedim for (
            nframedim,
            hopdim,
            frameshapedim) in zip(
            S.shape[
                fdim:],
            hop,
            trueFrameShape)]

    # building the positions of the frames. For each dimension, gridding from
    # 0 to sigShape[dim] every hop[dim]
    framesPos = [
        np.arange(size) * 
        step for (
            size,
            step) in zip(
            S.shape[
                fdim:],
            hop)]

    # constructing the weighting window. Choosing hamming for convenience
    # (never 0)
    win = np.array(1)
    for dim in range(fdim):
        if trueFrameShape[dim] == 1:
            win = win[...,None]
        else:
            key=((None,)*len(win.shape)+(Ellipsis,))
            win = win[...,None]* np.hamming(trueFrameShape[dim]).__getitem__(key)

    
    #if we need to compensate for analysis weighting, simply square window
    if weightedFrames:
        win2 = win ** 2
    else:
        win2=win

    sig = np.zeros(sigShape, dtype=S.dtype)

    # will also store the sum of all weighting windows applied during
    # overlap and add. Traditionally, window function and overlap are chosen
    # so that these weights end up being 1 everywhere. However, we here are
    # not restricted here to any particular hopsize. Hence, the price to pay
    # is this further memory burden
    weights = np.zeros(sigShape)

    # total number of frames (for displaying)
    nFrames = np.prod(S.shape[fdim:])

    # could use memmap or stuff
    S *= win[[Ellipsis] + [None] * (len(S.shape) - len(win.shape))]

    # for each frame
    for iframe, index in enumerate(itertools.product(*[range(len(x)) for x in framesPos])):
        # display from time to time if asked for
        if verbose and (not iframe % 100):
            print 'overlap-add : frame ' + str(iframe) + '/' + str(nFrames)

        # build the slice to use for overlap-adding the signal of this frame.
        frameRange = [Ellipsis]
        for dim in range(nDim-fdim):
            frameRange += [slice(framesPos[dim][index[dim]],
                                 min(sigShape[dim],
                                     framesPos[dim][index[dim]] + trueFrameShape[dim]),
                                 1)]

        # put back the reconstructed weighted frame into place
        frameSig = S[[Ellipsis] + list(index)]
        sig[frameRange] += frameSig[[Ellipsis] + 
                                    [None] * 
                                    (len(sig[frameRange].shape) - 
                                     len(frameSig.shape))]

        # also store the corresponding window contribution
        weights[frameRange] += win2[[Ellipsis] + 
                                    [None] * 
                                    (len(weights[frameRange].shape) - 
                                     len(win2.shape))]

    # account for different weighting at different places
    sig /= weights

    # truncate the signal if asked for
    if shape is not None:
        sig_res = np.zeros(shape, S.dtype)
        truncateRange = [slice(0, min(x, sig.shape[i]), 1) for (i, x) in enumerate(shape)]
        sig_res[truncateRange] = sig[truncateRange]
        sig = sig_res

    # finished
    return sig



def stft(sig, frameShape, hop, real=True, verbose=False):
    """n-dimensional STFT (Short Time Fourier Transform)
    sig : ndarray
    frameShape : tuple giving the size of each frame. If its shape is
                 smaller than that of sig, assume the frame is of size 1
                 for all missing dimensions
    hop : tuple giving the hopsize in each dimension. If its shape is
          smaller than that of sig, assume the hopsize is 1 for all
          missing dimensions
    real: if True, use rfft (discard negative frequencies), if False, use
          fft
    verbose : whether to output progress during computation"""
    if np.isscalar(frameShape):
        frameShape = (frameShape,)
    stft = split(sig, frameShape, hop, True,verbose)
    
    # at the end, we apply the fft function. We do this at the end for the speed of calling it only once, but I've
    # noticed this may cause memory error problems. I decided not to care. One
    # could use memmap or stuff
    # if the signal is real, use rfft, else use fft
    if real:
        fftFunction = np.fft.rfftn
    else:
        fftFunction = np.fft.fftn
    stft = fftFunction(stft, frameShape, axes=range(len(frameShape)))
    return stft


def istft(S, fdim, hop, real=True, shape=None, single=False, verbose=False):
    """n-dimensional inverse STFT
    S    : ndarray containing the stft to be inverted
    fdim : the number of dimensions in S corresponding to
           frequency indices.
    hop  : tuple containing hopsizes along dimensions.
           Missing hopsizes are assumed to be 1
    real : if True, using irfft, if False, using ifft
    shape: Indicating the original shape of the
           signal for truncating. If None: no truncating is done
    single: if True, single precision
    verbose: whether or not to display progress


            """

    # alocating memory for stft
    if real:
        if single:
            typeSig = 'float32'
        else:
            typeSig = 'float64'
        ifftFunction = np.fft.irfftn
    else:
        if single:
            typeSig = 'complex64'
        else:
            typeSig = 'complex128'
        ifftFunction = np.fft.ifftn

    # before overlap-add, we apply the ifft function. We do this now for the speed of calling it only once, but I've
    # noticed this may cause memory error problems. I decided not to care. One
    # could use memmap or stuff
    S = ifftFunction(S, axes=range(fdim)).astype(typeSig)
    sig = overlapadd(S, fdim, hop, shape, True, verbose)

    # finished
    return sig
