# In this demo the KAM algorithm is tested

import matplotlib.pyplot as plt
import numpy as np
from KAM import AudioSignal,Kernel,kam #,Kernelmat
from scikits.audiolab import wavread, play
import time    


# inputs of the 'kam' function

SourceKernels=[['vertical',np.mat('10')]]#,['vertical',np.mat('10')],['cross',np.mat('5,5')]]
Numit=1


start_time = time.clock()
Inputfile=['dev1_nodrums_inst_mix.wav',0.1,0]
shat=kam(Inputfile,SourceKernels,Numit)[0]
for numblock in range(1,2):
   Inputfile=['dev1_nodrums_inst_mix.wav',0.1,0.1*numblock]
   shat_temp=kam(Inputfile,SourceKernels,Numit)[0]
   shat=np.append(shat,shat_temp,axis=0)      
print time.clock() - start_time, "seconds"   



plt.figure()
plt.subplot(3,1,1)
plt.plot(shat[:,0,0])
plt.title(r'$\hat{s}_1$')

plt.subplot(3,1,2)
plt.plot(shat[:,0,1])
plt.title(r'$\hat{s}_2$')

plt.subplot(3,1,3)
plt.plot(shat[:,0,2])
plt.title(r'$\hat{s}_3$')