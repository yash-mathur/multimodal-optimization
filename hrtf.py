

import sys
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.signal import freqz
import scipy.io


# In[2]:


def score_calculator(z,y,warpFactor):
    fNorm = np.linspace(0, 1, len(y))
    barkWarp = lambda w,x : np.angle((np.exp(1j*w) - x)/(1-x*np.exp(1j*w)))
    w = np.pi*fNorm
    wWarp = barkWarp( w, -warpFactor )
    yWarp = np.interp(w, y, wWarp)
    n = len(z)
    b = z[0:int(n/2)]
    a = z[int(n/2):n]
    w,h = freqz(b,a,len(y))
    h = np.abs(h)
    wWarp = barkWarp( w, -warpFactor );
    hWarp = np.interp(w, h, wWarp)
    opp = np.nansum(np.abs(yWarp-np.transpose(hWarp)))
    #opp = np.nansum(np.abs(yWarp*np.ones((len(yWarp),len(yWarp)))-np.transpose(hWarp*np.ones((len(hWarp),len(hWarp))))))
    return opp


# In[3]:


def parameter_setter_dev(objfunc,order,NRe):
    if order%2 == 0:
        if NRe%2 == 1:
            print('NRe MUST be an even number')
            objfunc=''
    else:
        if NRe%2 == 0:
            print('NRe MUST be an odd number')
            objfunc=''

    Lb,Ub=-5*np.ones(order+1), 5*np.ones(order+1)
    LRe,URe=-0.99*np.ones(NRe),0.99*np.ones(NRe)
    NCo=int((order-NRe)/2)
    LP,UP=np.zeros(NCo), np.pi*np.ones(NCo)
    LMa,UMa=0.5*np.ones(NCo), 0.99*np.ones(NCo)
    L=np.array(Lb.tolist()+LRe.tolist()+LMa.tolist()+LP.tolist())
    U=np.array(Ub.tolist()+URe.tolist()+UMa.tolist()+UP.tolist())
    L=np.transpose(L)
    U=np.transpose(U)
    z=L+np.abs(U-L)*[random.random() for i in range(len(L))]
    varargout[1]=z
    varargout[2]=L
    varargout[3]=U
    return(z,L,U)


# In[4]:


def pol2cart(phi,rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def audio_fitness_coeff_poles_parametric_dev(z,order,NRe,y,warpFactor):
    z=np.transpose(z)
    b = z[0:order+1]
    start_pol=order+1
    RePoles=z[start_pol: start_pol+NRe]
    start_C_pol=start_pol+NRe
    NCo=int((order-NRe)/2)
    M1=z[start_C_pol:start_C_pol+NCo]
    Ph1=z[start_C_pol+NCo:]
    Ph2=2*np.pi-Ph1
    M=np.array(M1.tolist()+M1.tolist())
    Ph=np.array(Ph1.tolist()+Ph2.tolist())
    [ar1,ar2]=pol2cart(Ph,M)
    ar=ar1+ar2*1j
    ar=np.array(RePoles.tolist()+ar.tolist())
    a=np.poly(ar)
    z=np.array(b.tolist()+a.tolist())
    opp = score_calculator(z,y,warpFactor)
    return(opp)
    


# In[5]:


varargout = {}
mat = scipy.io.loadmat('data/y.mat')
order = 24
NRe = 0
n = 512
cmaesMaxIter = 25000
warpFactor = 0.6
y=mat['y'][0]
MaxIter = cmaesMaxIter
obj_func = 'audio_fitness_coeff_poles_parametric_dev'


# In[6]:


[z, L, U]=parameter_setter_dev(obj_func,order, NRe)


# In[7]:


audio_fitness_coeff_poles_parametric_dev(z,order,NRe,y,warpFactor)


# In[8]:


print(z.shape,z)


# In[9]:


print(len(L),L)


# In[10]:


print(len(U),U)


# In[ ]:





# In[ ]:




