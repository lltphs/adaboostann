from ann import *

class AdaBoost:
    def __init__(self):
        self.clsf=[]
        self.num_clsf=0
        self.beta=[]
    
    def put(self,clsf):
        self.clsf.append(clsf)
        self.num_clsf+=1
        self.beta.append(2)
        return self

    def sample(self,w):
        acc_w=np.add.accumulate(w)
        sub=np.zeros(w.shape[0],'int')
        rd=np.random.rand(w.shape[0])*acc_w[-1]
        for j in range(w.shape[0]):
            l,m,r=0,(w.shape[0]-1)//2,w.shape[0]-1
            while rd[j]<acc_w[m-1] or acc_w[m]<=rd[j]:
                if rd[j]<acc_w[m-1]:
                    m,r=(l+m)//2,m
                    if m==0:
                        break
                elif acc_w[m]<=rd[j]:
                    l,m=m,(m+r)//2+(m+r)%2
            sub[j]=m
        return sub
        
    def fit(self,x,y):
        w=np.ones(y.shape[0])
        for i in range(self.num_clsf):
            err=1
            while err>0.5:
                sub=self.sample(w)
                x_sub,y_sub=x[sub],y[sub]
                e=p2i(self.clsf[i].fit(x_sub,y_sub).predict(x_sub))==p2i(y_sub)
                err=1-np.sum(w[sub]*e)/np.sum(w[sub])
                if err>0.5:
                    w=np.ones(y.shape[0])
                else:
                    self.beta[i]=err/(1-err)
                    w[sub[e]]*=self.beta[i]
                    w/=np.sum(w)
        return self
        
    def predict(self,x):
        return np.sum([np.log(1/self.beta[i])*p2o(self.clsf[i].predict(x)) for i in range(self.num_clsf)],0)
        
    def evaluate(self,x,y):
        return np.average(p2i(y)==p2i(self.predict(x)))
    
    def save(self,file):
        pickle.dump((self.clsf,self.beta),open(f'{file}','wb'))
        return self

    def load(self,file):
        self.clsf,self.beta=pickle.load(open(f'{file}','rb'))
        self.num_clsf=len(self.clsf)
        return self