from adaboost import *
import matplotlib.pyplot as plt,cv2 as cv
from time import time

[(x_train,y_train),(x_test,y_test)]=fashion_mnist()

file = open('result.txt','w')
output='n_ann\ttrain\ttest\ttime'
print(output)
for num_ann in range(1,200,2):
    ann=[ANN().layer(10) for i in range(num_ann)]
    adaboost=AdaBoost()
    start=time()
    temp=reduce(lambda x,y:x.put(y),ann,adaboost).fit(x_train,y_train).save(f'adaboost{num_ann}').evaluate(x_train,y_train)
    stop=time()
    temp2=max([adaboost.evaluate(x_test,y_test) for i in range(adaboost.num_clsf)])
    output+=f'\n{num_ann}\t{round(temp,4)}\t{round(temp2,4)}\t{round(stop-start,0)}'
    print(f'{num_ann}\t{round(temp,4)}\t{round(temp2,4)}\t{round(stop-start,0)}')

file.write(output)

# cv.namedWindow('OpenCV Introduction',cv.WINDOW_NORMAL)
# img=cv.imread('YenNhi.jpg')
# cv.imshow('OpenCV Introduction',img)
# cv.waitKey()
# file=open('result (1).txt','r')
# file.readline()
# num_ann=[]
# train_err=[]
# train_rec=[]
# test_err=[]
# test_err=[]
# l=0.1
# p=9
# while True:
#     try:
#         a=file.readline().split(' ')[0].split('\t')
#         num_ann.append(float(a[0]))
#         train_err.append(1-float(a[1]))
#         train_rec.append(np.exp(-float(a[0])/7)/25+0.1445)
#         test_err.append(1-float(a[2]))    
#     except:
#         break

# plt.plot(num_ann,train_err,num_ann,test_err,num_ann,train_rec,'r')
# plt.show()
