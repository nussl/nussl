'''test for PROJET

simply looping over the wav files and applying PROJET on each of them
'''

import PROJET
import delaysPROJET
import os, random, time, glob,re


source_path="./toydata"
dir_path='./output'


for root, dirs, files in os.walk(source_path):
    random.shuffle(dirs)
    wavfiles =  sorted(glob.glob(os.path.join(root,'*love.wav')))
    print root
    target_root = re.sub(source_path,dir_path,root)
    if not os.path.isdir(target_root):
        os.mkdir(target_root)

    random.shuffle(wavfiles)
    for filename in wavfiles:
        print 'Launching PROJET on file', filename
        start =  time.time()

        #with standard PROJET
        PROJET.separate('toydata/beatles_short.wav', '.',6,41,15,200)

        #with delay PROJET
        delaysPROJET.separate(filename, dir_path, 6, 21, 5, 21, 5, maxDelaySamples=3, nIter=400,alpha=1,beta=1,firstCentered = False)

        print 'done. time elapsed : %0.1fs'%(time.time()-start)