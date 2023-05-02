import numpy as np
import msvcrt


class G_Map:
    mapi = np.zeros([10,10])
    real_Map = mapi.copy()
    
    def Nuton(maps): # Apple Axis!
        y=np.random.randint(0, maps.shape[0])
        x=np.random.randint(0, maps.shape[1])
        return [y,x]
    
class Game_Val:
    sn = []
    nu = G_Map.Nuton(G_Map.mapi)
    length = 1
    head = [5,5]

    pain = 2
    poin = 1
    low = 0.5
    point = 0

class Display:
    def display(maps, point = 0):
        pizz = ("#" * (maps.shape[0]+2 ))+ "\n#"
        for y in maps:
            for x in y:
                a = " "
                # a = "□"
                if x == 0:
                    a = " "
                elif x > 0 and x < 1:
                    a = "■"
                elif x == 1:
                    a = "▤"
                else:
                    a = "$"
                pizz+= a
            pizz+= "#\n#"
        pizz += ("#" * (maps.shape[0]+1 ))+"\n"
        # print("\033[H\033[J")
        print(f"\033[{0};{0}H",end="")
        print(pizz)

class Move(G_Map):
    def len_sp(sn, lens:int):
        sn = sn[::-1][:lens][::-1]
        return sn
    
    def Key_Input():
        if msvcrt.kbhit():
            byte_arr = msvcrt.getche()
            try:
                key = byte_arr.decode("utf-8")
                if key == "d":
                    Game_Val.poin = 2
                if key == "w":
                    Game_Val.poin = 3
                if key == "s":
                    Game_Val.poin = 1
                if key == "a":
                    Game_Val.poin = 0
                
                return Game_Val.poin
            
            except UnicodeDecodeError:
                print("WASD로만 플레이가 가능해요!")
                raise KeyError
            
            

    def wig(poin = [0,0], poins = 0, maps = [], sn = [], nus = [], lens = 5 , point = 0):
        if poins == 1:
            a = [poin[0]+1,poin[1]]
        elif poins == 2:
            a = [poin[0],poin[1]+1]
        elif poins == 3:
            a = [poin[0]-1,poin[1]]
        elif poins == 0:
            a = [poin[0],poin[1]-1]
        # print(a)        
        try:
            for i in sn:
                if i == a:
                    # print("tail_Touch")
                    raise
            if maps[a[0],a[1]] == 1:
                return False, a, lens, -1
            if a[0] < 0 or a[1] < 0:
                raise
            if a[0] > maps.shape[0] or a[1] > maps.shape[1]:
                raise
            # print(nus[0])
            
            if nus[0] == a[0] and nus[1] == a[1]: 
                # print("apple")
                nus = G_Map.Nuton(maps)
                lens += 1
                point += 1
                # print("apple")
        except:
            return False, a, nus, lens, -1
        return True, a, nus, lens, point
    
class Games(Move, G_Map, Display, Game_Val):
    def Bun():
        G_Map.real_Map = G_Map.mapi.copy()
        for i in Game_Val.sn:
            G_Map.real_Map[i[0],i[1]] = 0.5
        G_Map.real_Map[Game_Val.head[0],Game_Val.head[1]] = 1
        G_Map.real_Map[Game_Val.nu[0],Game_Val.nu[1]] = 2
        Display.display(G_Map.real_Map, Game_Val.point)

    def Faty():
        Game_Val.poin
        
        Game_Val.sn.append(Game_Val.head)
        Game_Val.sn = Move.len_sp(Game_Val.sn, Game_Val.length)
        
        _,Game_Val.head, Game_Val.nu, Game_Val.length, Game_Val.point = Move.wig(Game_Val.head, Game_Val.poin, G_Map.real_Map, Game_Val.sn, Game_Val.nu, Game_Val.length)
        # print(Game_Val.point)
        if not _:
            Game_Val.sn = []
            Game_Val.nu = G_Map.Nuton(G_Map.mapi)
            Game_Val.length = 1
            Game_Val.head = [5,5]

            Game_Val.pain = 2
            Game_Val.poin = 1
            Game_Val.low = 0.5
            # Game_Val.point = 0
        return _