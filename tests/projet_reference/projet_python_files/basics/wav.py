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
import wave
import os


def wavinfo(filename):
    """returns metadata properties of a file"""
    fileHandle = wave.open(filename, 'r')
    nChans = fileHandle.getnchannels()
    fs = fileHandle.getframerate()
    length = fileHandle.getnframes()
    sampleWidth = fileHandle.getsampwidth()
    fileHandle.close()
    return length, nChans, fs, sampleWidth


def wavwrite(signal, fs, destinationFile, nbytes=2,verbose=True):
    """writes audio data into a file"""
    fileHandle = wave.open(destinationFile, 'wb')
    fileHandle.setparams(
        (signal.shape[1],
         nbytes,
         fs,
         signal.shape[0],
         'NONE',
         'not compressed'))
    pos = 0
    length = signal.shape[0]
    batchSize = 3000000

    while pos < length:
        batchSize = min(batchSize, length - pos)
        tempdata = np.minimum(np.abs(signal[pos:pos+batchSize, :].flatten()), 0.98)*np.sign(signal[pos:pos+batchSize,:].flatten())
        dataVec = (2 ** (nbytes * 8 - 1) * tempdata).astype(np.int16)
        values = dataVec.tostring()
        fileHandle.writeframes(values)
        pos += batchSize
    fileHandle._file.flush()
    os.fsync(fileHandle._file)
    fileHandle.close()
    if verbose:
        print 'File written to ' + destinationFile


def wavread(fileName,lmax=np.infty):
    """reads the wave file file and returns a NDarray and the sampling frequency"""

    def isValid(filename):
        if not fileName:
            return False
        try:
            fileHandle = wave.open(fileName, 'r')
            fileHandle.close()
            return True
        except:
            return False
    if not isValid(fileName):
        print "invalid WAV file. Aborting"
        return None

    # metadata properties of a file
    length, nChans, fs, sampleWidth = wavinfo(fileName)
    length = min(length,lmax*fs)
    waveform = np.zeros((length, nChans))

    # reading data
    fileHandle = wave.open(fileName, 'r')
    pos = 0
    batchSizeT = 3000000
    while pos < length:
        batchSize = min(batchSizeT, length - pos)
        str_bytestream = fileHandle.readframes(batchSize)
        tempData = np.fromstring(str_bytestream, 'h')
        tempData = tempData.astype(float)
        tempData = tempData.reshape(batchSize, nChans)
        waveform[pos:pos+batchSize, :] = tempData / float(2**(8*sampleWidth - 1))
        pos += batchSize
    fileHandle.close()
    return (waveform, fs)
