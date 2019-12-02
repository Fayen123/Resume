import math
def ItemSimilarity(train):
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in N.keys():
                N[i] = 0
            N[i] += 1
            for j in items.keys():
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = dict()
                if j not in C[i].keys():
                    C[i][j] = 0
                C[i][j] += 1
    W = dict()
    for i, related_items in C.items():
        if i not in W.keys():
            W[i] = dict()
        for j,cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

if __name__ == '__main__':
    Train_Data = {
        'A':{
            '苹果':1,'香蕉':1,'西瓜':1
        },
        'B':{'苹果':1,'西瓜':1},
        'C':{'苹果':1,'香蕉':1,'菠萝':1},
        'D':{'香蕉':1,'葡萄':1},
        'E':{'葡萄':1,'菠萝':1},
        'F':{'香蕉':1,'西瓜':1}
    }
    W = ItemSimilarity(Train_Data)
    print(W)

