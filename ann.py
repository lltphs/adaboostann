from functools import reduce
from typing import Any
from data_loader import *
from numpy import concatenate as concat

class ANN:
    def __init__(self):
        self.shapes=None
        self.W=None
        self.num_layers=0

    def layer(self,out):
        self.shapes=[(None,out)] if self.shapes is None else self.shapes+[(self.shapes[-1][1]+1,out)]
        self.num_layers+=1
        return self
    
    def fit(self,x,y):
        one=np.array([1])
        if self.W is None:
            self.shapes[0]=(x.shape[1]+1,self.shapes[0][1])
            self.W=[np.random.rand(*shape).T/shape[0] for shape in self.shapes]
        for i in range(y.shape[0]):
            o=reduce(lambda o,w:o+[1/(1+np.exp(-np.dot(w,concat((o[-1],one)))))],self.W,[x[i]])
            dLu=(o[-1]-y[i])*o[-1]*(1-o[-1])
            for j in range(self.num_layers,0,-1):
                dLW=np.dot(dLu.reshape(-1,1),concat((o[j-1],one)).reshape(-1,1).T)
                dLu=np.dot(self.W[j-1][:,:-1].T,dLu)*o[j-1]*(1-o[j-1])
                self.W[j-1]-=0.1*dLW
        return self
    
    def predict(self,x):
        return reduce(lambda o,w:1/(1+np.exp(-np.dot(concat((o,np.ones((o.shape[0],1))),1),w.T))),self.W,x)

    def evaluate(self,x,y):
        return np.average(np.argmax(y,1)==np.argmax(self.predict(x),1))
    
    def save(self,file):
        pickle.dump(self.W,open(f'{file}','wb'))
        return self

    def load(self,file):
        self.W=pickle.load(open(f'{file}','rb'))
        self.num_layers=len(self.W)
        return self