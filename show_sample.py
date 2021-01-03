import dataget,numpy as np,cv2 as cv

x_train,y_train,x_test,y_test=dataget.image.fashion_mnist(global_cache=True).get()

idx_each_type=[1,16,5,3,19,8,18,6,23,0]
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cv.namedWindow('Fashion MNIST',cv.WINDOW_NORMAL)
img=None
for i in range(2):
    img1=None
    for j in range(5):
        temp=x_train[idx_each_type[i*2+j]]
        temp=np.repeat(np.repeat(temp,7,axis=0),7,axis=1)
        temp=np.concatenate((temp,np.ones((28*2,28*7))*255))
        temp=cv.cvtColor(temp.astype('uint8'),cv.COLOR_GRAY2BGR)
        font=cv.FONT_HERSHEY_SIMPLEX
        label=class_names[y_train[idx_each_type[i*2+j]]]
        cv.putText(temp,label,(15,28*8+10),font,1,(0,0,0),2,cv.LINE_AA)
        if img1 is None:
            img1=temp
        else:
            img1=np.concatenate((img1,temp),1)
    if img is None:
        img=img1
    else:
        img=np.concatenate((img,img1),0)

cv.imshow('Fashion MNIST',img)
cv.waitKey()