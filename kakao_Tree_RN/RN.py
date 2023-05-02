from Sngame import *
import numpy as np
from kakao_Tree import *

mas = G_Map.real_Map.shape[0]*G_Map.real_Map.shape[1]

def model() :
    models = MLs(le= 0.00001)
    # models.add(Conv2D(16,(2,2), InputShape=(30,30,1), acfn= None))
    # models.add(flatten())
    models.add(Dense(32, acfn=sigmoid(), InputShape=(4)))
    models.add(Dense(32, acfn= reLu(0.5)))
    models.add(Dense(4, acfn= reLu()))
    models.Sets()
    return models

def softmax_(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def Policy(state, entropy = 0.3):
    global Model_
    ys = Model_.foword(state)
    # print(np.array(ys))
    print(softmax_(ys))
    # print(np.max(softmax_(ys)))
    # print(list(softmax_(ys)))
    key = list(softmax_(np.round(ys, 5))).index(np.max(softmax_(np.round(ys, 5))))
    # print(key)
    if entropy <= np.round(np.random.rand(1), 1)[0]:
        key = np.random.randint(0, len(ys))
    return key

def apple_s():
    if Game_Val.head[0] == Game_Val.nu[0] or \
    Game_Val.head[1] == Game_Val.nu[1]:
        return 1
    return 0

def apple_a():
    if Game_Val.head[0] == Game_Val.nu[0] or \
    Game_Val.head[1] == Game_Val.nu[1]:
        a = abs(Game_Val.head[0] - Game_Val.nu[0])
        b = abs(Game_Val.head[1] - Game_Val.nu[1])
        return a,b
    else:
        a = Game_Val.head[0]
        b = Game_Val.head[1]
        return a,b

def les(p):
    y = [0,0,0,0]
    y[p] = 1
    return np.array(y)

Model_ = model()
reward = 0
import os
os.system("cls")
xs = []
ys = []
p = 0
for i in range(100000):
    time.sleep(0.01)
    Games.Bun() # 표시 함수 


    # Policy --------
    a = apple_a()
    state = [apple_s(), a[0], a[1], 1]
    action = Policy(np.array(state), 0.3)
    print(state)
    print(action)
    # ---------------

    # action --------
    Game_Val.poin = action
    Games.Faty() #게임 Update
    # ---------------

    # reward --------
    reward = Game_Val.point
    print(f"{reward=}  " , f"max p = {Game_Val.length}   ")
    print(f"{len(xs)=}  ")
    # ---------------
    if p < Game_Val.length:
        p = Game_Val.length
    state = [apple_s(), a[0], a[1], reward]
    if reward <= 0:
        if 0.15 > np.random.rand(1)[0]:    
            xs.append(state)
            ys.append(les(action))
    elif reward > 0:
        print(f"{reward=}  ")
        xs.append(state)
        ys.append(les(action))
    if i % 100 == 0:
        if len(xs) > 50:
            Model_.runs(np.array(xs), np.array(ys)) 
            xs = []
            ys = []
        os.system("cls")
    # yf = les(action)
    