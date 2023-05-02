import numpy as np


def sigmoid_f(x):
    return 1 / (1 + np.exp(-np.round(x,25),dtype=np.double)) -1


def sigmoid_b(x, rh):
    return x*(rh > 0)

def Pooling(src, z = (2, 2), ad = np.max):
    f = np.zeros((int(len(src[0])/z[0]), int(len(src[1])/z[1]), len(src[0,0,:])))
    for i in range(0, len(src)):
        for j in range(0, len(src[0]), z[1]):
            ad(src[i*z[0]:i*z[0]+z[0], j*z[1]:j*z[1]+z[1]])
        

def Conv_f(x, ws, hs, k, bias):
    mas = np.zeros((hs,ws,len(k)), dtype=np.double)
    news = np.zeros((len(x), hs,ws,len(k)), dtype=np.double)
    base = np.zeros((hs,ws), dtype=np.double)
    
    for s in range(len(x)):
        mas *= 0
        for ks in range(len(k)):
            base *= 0
            for h in range(hs):
                for w in range(ws):
                    # print(h,w)
                    # print(h+len(k[1]))
                    # print(w+len(k[2]))
                    # print( x[s, h:h+len(k[1]), w:w+len(k[2])])
                    post = x[s, h:h+len(k[1]), w:w+len(k[2])]*k[ks]
                    # print(post.shape)

                    fast = np.sum(post, axis= 0)
                    just = np.sum(fast, axis= 1)
                    base[h,w] = np.sum(just)
            for yy in range(hs):
                for xx in range(ws):
                    mas[yy,xx][ks] = base[yy,xx] + bias[ks]
        news[s] = mas
    return news

def T_trejers(img, k, le):
    k = k.T
    
    kc, kh, kw, nc = k.shape
    n, h, w, c = img.shape
    f = np.zeros((n, h+kh-1, w+kw-1, kc))

    for index in range(n):
        for npc in range(kc):
            for i in range(h+kh-1):
                dgf = (i-kh+1)*(i-kh+1 >= 0)
                dgff = i +( h - i) * (i+kh > h+kh-2)

                kf = (i-(h-1))*(i > h-1)         
                kff = (i)*(i < kh) + (kh-1)*(i >= kh)   

                for j in range(w+kw-1):
                    d = (j-kw+1)*(j-kw+1 >= 0) 
                    df = j +( w - j) * (j+kw > w+kw-2)

                    p = (j-(w-1))*(j > w-1)         
                    pf = (j)*(j < kw) + (kw-1)*(j >= kw) 
                    
                    
                    Ap = img[index][dgf:dgff+1, d:df+1]
                    asia = k[npc][kf:kff+1, p:pf+1]
                    
                    f[index, i, j, npc] = np.sum(Ap*asia)*le
        
    return f


def     Conv_b(x, ws, hs, k, Nk, db, out):
    for s in range(len(x)):
        for ks in range(len(k)):
            for h in range(hs):
                for w in range(ws):
                    poa = x[s, h:h+len(k[1]), w:w+len(k[2])]*out[s, w,h][ks]
                    Nk[ks] = Nk[ks][::-1, ::-1, :] + x[s][::-1,::-1,:][h:h+len(k[1]), w:w+len(k[2]), :] * out[s, h, w, ks]
                    db[ks] += out[s, h, w, ks]
                    # print((poa*le).shape == k[ks].shape)
                    # print((x[s, h:h+len(k[1]), w:w+len(k[2])]-out[s, w,h][ks]*0.1))
                    # k[ks] = 
                    # if s == 0 and w == 0:
                    #     print(k[ks])

                    # print("**",Nk[ks].shape)
                    # print((x[s, h:h+len(k[1]), w:w+len(k[2])]).shape)

                    # print("::",out.shape)
                    # print("::",conv.shape)
                    
                    # # print(x.shape)
                    
                    # print(len(x))
                    # print("&&", Nk[ks].shape)

                    # Nk[ks] += 0
                    # print("$$",out[s, h, w, ks])
                    # print("$$",conv[s, h:h+len(k[1]), w:w+len(k[2]), :].shape)

                    # print(np.max(((out[s, w,h][ks]*0.1) * x[s, h:h+len(k[1]), w:w+len(k[2])])))
    return Nk, db, out


def dense_f(x, w, b, acfn):
    if None == acfn:
        h = np.dot(x, w) + b
    else:
        h = acfn.foword(np.dot(x, w) + b)
    return h