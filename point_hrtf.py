
import numpy as np
import scipy as sp
from scipy.signal import freqz
import scipy.io
import random

# from test_functions import evaluate
def parameter_setter_dev(order=24,NRe=0):
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
    z1=L+np.abs(U-L)*[random.random() for i in range(len(L))]
    return(z1,L,U)


# In[4]:


def pol2cart(phi,rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def audio_fitness_coeff_poles_parametric_dev(z1,order=24,NRe=0):
    mat = scipy.io.loadmat('y.mat')
    y=mat['y'][0]
    warpFactor = 0.6
    z1=np.transpose(z1)
    b = z1[0:order+1]
    start_pol=order+1
    RePoles=z1[start_pol: start_pol+NRe]
    start_C_pol=start_pol+NRe
    NCo=int((order-NRe)/2)
    M1=z1[start_C_pol:start_C_pol+NCo]
    Ph1=z1[start_C_pol+NCo:]
    Ph2=2*np.pi-Ph1
    M=np.array(M1.tolist()+M1.tolist())
    Ph=np.array(Ph1.tolist()+Ph2.tolist())
    [ar1,ar2]=pol2cart(Ph,M)
    ar=ar1+ar2*1j
    ar=np.array(RePoles.tolist()+ar.tolist())
    a=np.poly(ar)
    z1=np.array(b.tolist()+a.tolist())
    #opp = score_calculator(z,y,warpFactor)
    return(z1)

class Point:
    def __init__(self, dim=2, upper_limit=10, lower_limit=-10, objective=None):
        self.dim = dim
        self.coords = np.zeros(self.dim)
        self.range_lower_limit = np.zeros(self.dim)
        self.range_upper_limit = np.zeros(self.dim)
        self.z = None
        self.objective = objective
        self.evaluate_point()

    def generate_random_point(self):
        z1,self.range_lower_limit,self.range_upper_limit = parameter_setter_dev(self.dim)
        self.coords = audio_fitness_coeff_poles_parametric_dev(z1)
        #self.coords = np.random.uniform(self.range_lower_limit, self.range_upper_limit, (self.dim,))
        self.evaluate_point()
    
    def generate_definite_point(self,z1):
        self.coords = z1
        self.evaluate_point()
    
    def evaluate_point(self):
        # self.z = evaluate(self.coords)
        self.z = self.objective.evaluate(self.coords)


if __name__ == '__main__':
    print("Point class defined in this script")
