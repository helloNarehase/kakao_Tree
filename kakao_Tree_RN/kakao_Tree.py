import numpy as np
import matplotlib.pyplot as plt
import cv2
from kakao_Seed import *
import time
class sigmoid:
    def __init__(self, alpha = 0) -> None:
        self.alpha = alpha
        self.kuki = None

    def foword(self, x):
        # if x < 0:
        #     return np.exp(-x,dtype=np.double) / (1 + np.exp(-x,dtype=np.double))
        # else:
        #     return 1 / (1 + np.exp(-x,dtype=np.double))
        return 1 / (1 + np.exp(-x,dtype=np.double))
    
    def backward(self, x, rh):
        # print(self.F.shape)
        # print(x.shape)
        # print(rh.shape)
        # x*(np.round(rh, 5) *self.alpha)
        
        return x*(rh > 0)


class reLu:
    def __init__(self, alpha:float = 0) -> None:
        self.alpha = alpha
        self.kuki = None

    def foword(self, x):
        # self.F = (x > x*self.alpha)
        # f = np.maximum(x*self.alpha, x)
        return (x)*(x > 0)

    def backward(self, x, rh):
        # print(self.F.shape)
        # print(x.shape)
        # print(rh.shape)
        # x*(np.round(rh, 5) *self.alpha)
        
        # return x*(np.round(rh, 5)*self.alpha<np.round(rh, 5))
        (rh > 0)
        return (x+1)*(rh > 0)



class Dense:
    def __init__(self, unit = 10,acfn = reLu(), InputShape = None) -> None:
        self.acfn = acfn
        self.unit = unit
        self.InputShape = InputShape

        self.h = 0.
        self.w = None
        self.b = np.zeros((unit), dtype=np.double)
    
    def sets(self, r_unit):
        self.w = np.random.randn(r_unit, self.unit) / np.sqrt(r_unit)
        return self.unit
    
    def in_out(self):
        # print(self.InputShape)
        return self.InputShape
    
    def foword(self, x):
        # self.b *= 0 
        self.rh = x
        # print(self.acfn)
        if None == self.acfn:
            h = np.dot(x, self.w) + self.b
        else:
            h = self.acfn.foword(np.dot(x, self.w) + self.b)
        self.h = h
        # h = dense_f(x, self.w, self.b, self.acfn)
        # self.h = h
        return h

    def backward(self, dscore, le):
        dw = np.dot(self.rh.T, dscore)
        db = np.sum(dscore, axis=0, dtype=np.double)
        if None == self.acfn:
            df = np.dot(dscore, self.w.T)
        else:
            df = self.acfn.backward(np.dot(dscore, self.w.T),self.rh)
        
        self.w -= le*dw
        self.b -= le*db
        
        return df

class flatten:
    def __init__(self) -> None:
        pass
    def foword(self, x):
        # print(x.shape)
        # print(self.si)
        a = 0
        for i in x.shape:
            a*= i
        self.F = x.shape[0]
        # print(x.shape)
        # print(np.array(x).reshape((x.shape[0],self.si)).shape)
        
        # print(np.array(x).reshape((x.shape[0],self.si))[0][0])
        # print(np.array(x).reshape((x.shape[0],self.si))[1][0])
        # print(np.mean(np.array(x[0])))
        # print(np.mean(np.array(x[1])))
        return np.array(x).reshape((x.shape[0],self.si))
    def sets(self, x):
        a = 1
        # print(x)
        for i in x:
            a*= i
            # print(i, a)
        self.si = a
        self.ba = x
        return a
    def backward(self, dscore, le):
        # print(self.ba) 
        a = []
        a.append(self.F)
        for i in self.ba:
            a.append(i)
        # print(dscore.reshape(a).shape)
        return dscore.reshape(a)

class Conv2D:
    def __init__(self, chanel, filter = (5, 5),acfn = reLu(), InputShape = None, bias = True):
        self.bias = np.zeros(chanel, dtype=np.double)
        self.filt = filter
        self.chanel = chanel
        self.acfn = acfn
        self.InputShape = InputShape
        self.kenel = None
        self.next = None
        self.next2acFn = None

    def in_out(self):
        # print(self.InputShape)
        return self.InputShape
    
    def sets(self, inputShape):
        # print(inputShape, "**")
        if self.InputShape != None:
            inputShape = self.InputShape
        self.InputShape = inputShape
        self.kenel = np.random.randn(self.chanel,self.filt[0],self.filt[1],inputShape[-1])
        k = self.kenel 
        print("="*6, "Conv2D", "="*6)
        print(self.InputShape)
        print(k.shape)
        print((inputShape[1]+1 - len(k[2]), inputShape[0]+1 - len(k[1]) ,self.chanel))
        print("="*20)
        return (inputShape[1]+1 - len(k[2]), inputShape[0]+1 - len(k[1]) ,self.chanel)

    def backward(self, out, le):
        k = self.kenel
        Nk = k.copy() * 0.
        db = np.zeros(self.bias.shape)
        x = self.x
        conv = self.next2acFn
        input_Shape = self.InputShape
        
        ws = input_Shape[1]+1 - len(k[2])
        hs = input_Shape[0]+1 - len(k[1])
        # print(len(k))
        # print(out.shape)
        # print(hs,ws)
        # for s in range(len(x)):
        #     for ks in range(len(k)):
        #         for h in range(hs):
        #             for w in range(ws):
        #                 poa = x[s, h:h+len(k[1]), w:w+len(k[2])]*out[s, w,h][ks]
        #                 Nk[ks] = Nk[ks][::-1, ::-1, :] + x[s][::-1,::-1,:][h:h+len(k[1]), w:w+len(k[2]), :] * out[s, h, w, ks]
        #                 db[ks] += out[s, h, w, ks]
        #                 pass

        Nk, db, out = Conv_b(x, ws, hs, k, Nk, db, out)

        self.kenel -= Nk * le
        self.bias -= db * le    
        nc = T_trejers(out, k, le)
        # print(db)
        # print(db.shape)
        # print(k.shape)
        
        return nc

    def foword(self, x):
        input_Shape = self.InputShape
        k = self.kenel
        # print()
        if list(x.shape[1:])!=(list(self.InputShape)):
            raise ValueError("err")

        ws = input_Shape[1]+1 - len(k[2])
        hs = input_Shape[0]+1 - len(k[1])
        bias = self.bias
        self.x = x
        # print(ws, hs)
        # mas = np.zeros((hs,ws,len(k)), dtype=np.double)
        # news = np.zeros((len(x), hs,ws,len(k)), dtype=np.double)
        # base = np.zeros((hs,ws), dtype=np.double)
        # for s in range(len(x)):
        #     mas *= 0
        #     for ks in range(len(k)):
        #         base *= 0
        #         for h in range(hs):
        #             for w in range(ws):
        #                 # print(h,w)
        #                 # print(h+len(k[1]))
        #                 # print(w+len(k[2]))
        #                 # print( x[s, h:h+len(k[1]), w:w+len(k[2])])
        #                 post = x[s, h:h+len(k[1]), w:w+len(k[2])]*k[ks]
        #                 # print(post.shape)

        #                 fast = np.sum(post, axis= 0)
        #                 just = np.sum(fast, axis= 1)
        #                 base[h,w] = np.sum(just)
        #         for yy in range(hs):
        #             for xx in range(ws):
        #                 mas[yy,xx][ks] = base[yy,xx] + self.bias[ks]

        #     news[s] = mas

        news = Conv_f(x, ws, hs, k, bias)
        self.next = news
        if self.acfn == None:
            self.next2acFn = news
        else:    
            self.next2acFn = self.acfn.foword(news)
        # print(np.max(self.next2acFn))
        # print(self.next2acFn.shape)
        # print(self.kenel)
        # print(self.next2acFn[0, :5, :5])
        # print(self.next2acFn[1, :5, :5])
        return(self.next2acFn)
    
class MLs:
    def __init__(self, layer:list = [], le = 0.0001) -> None:
        self.layer:list = layer
        self.le = le

    def add(self, layer = None):
        if layer == None:
            raise
        self.layer.append(layer)

    def Sets(self):
        a = self.layer[0]
        post = a.in_out()
        for i in self.layer:
            # print(post)
            post = i.sets(post)
            # print(post)
        
    def foword(self, x):
        for i in self.layer:
            x = i.foword(x)
            # print(i)
            # print(x[0][0])
        return x
    
    def backward(self, pre, y):
        dscore = pre - y
        # print(dscore)
        for i in self.layer[::-1]:
            # print(i)
            dscore = i.backward(dscore, self.le)
            # print(dscore)

    def runs(self, x,y, epoch = 100):
        # print("Start Run!")
        for i in range(epoch):
            e = time.time()
            ys = self.foword(x)
            self.backward(ys, y) 
            print(f"epoch {i+1}/{epoch} : {time.time() - e}'s")
        
if __name__ == "__main__":
    models = MLs(le= 0.000000000001)
    # models.add(Conv2D(16,(2,2), InputShape=(30,30,3), acfn= None))
    # models.add(Conv2D(64, (5,5), acfn= sigmoid()))
    # models.add(Dense(32, InputShape=(10)))
    models.add(Dense(32, acfn=reLu(0.3)))
    models.add(Dense(12, acfn= reLu(0.5)))
    models.add(Dense(4, acfn= reLu()))
    models.Sets()

    x = np.array([img1, img2]) / 255.

    # print(x[0])
    # x = np.array([np.zeros((15,15,3)), np.ones((15,15,3))*2])
    # x = np.array([np.random.rand(30,30,3), np.random.rand(30,30,3)])
    # x = np.array([np.zeros((10)), np.ones((10))])
    y = np.array([[0,1,0,0],
                [0,0,1,0]])
    # print(x.shape)
    # p = models.foword(x)
    # print(p, "*")

    # for i in range(20):
    #     if i % 2 == 0:
    #         print(i)
    #     p = models.foword(x)
    #     models.backward(p, y) 
    models.runs(x, y, 10)

    # p = models.foword(np.random.randn(10,10))
    # x = np.array([np.zeros((15,15,3)), nbn
    # p.ones((15,15,3))*255])
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    p = models.foword(x)
    print(p.shape, "*")
    print(np.round(p,1), "**")
    print(p)
    print(np.round(softmax(p[0]),3))
    print(np.round(softmax(p[1]),3))
