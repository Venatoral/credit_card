from bpNeuralNetworks import *
import numpy as np
import random
import matplotlib.pyplot as plt
import random
import copy
import math


class particle():
    def __init__(self, dropoutP1=0.06, dropoutP2=0.03,
                 lr=0.0009535, weight_decay=2.2e-6,
                 n_hidden1=44, n_hidden2=29,
                 n_hidden3=20, n_hidden4=12):
        self.dropoutP1 = dropoutP1*random.uniform(0, 2)
        self.dropoutP2 = dropoutP2*random.uniform(0, 2)
        self.lr = lr*random.uniform(0.8, 1.2)
        self.weight_decay = weight_decay*(0, 2)
        self.n_hidden1 = random.randint(17, 40)
        self.n_hidden2 = int(n_hidden1*random.uniform(0.6, 0.8))
        self.n_hidden3 = int(n_hidden2*random.uniform(0.6, 0.8))
        self.n_hidden4 = int(n_hidden3*random.uniform(0.6, 0.8))
        if self.n_hidden4 < 2:
            self.n_hidden4 = random.randint(3, 4)
        self.oldParameter = [self.dropoutP1, self.dropoutP2, self.lr, self.weight_decay,
                             self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_hidden4]

    def randomDisturbance(self):
        self.oldParameter = (self.dropoutP1, self.dropoutP2, self.lr, self.weight_decay,
                             self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_hidden4)
        self.dropoutP1 = self.dropoutP1*random.uniform(0.999, 1.0001)
        self.dropoutP2 = self.dropoutP2*random.uniform(0.999, 1.0001)
        self.lr = self.lr*random.uniform(0.999, 1.0001)
        self.weight_decay = self.weight_decay*(0.999, 1.0001)
        if random.random() > 0.95:
            self.n_hidden1 = self.n_hidden1+random.randint(-1, 1)
        if random.random() > 0.95:
            self.n_hidden2 = self.n_hidden2+random.randint(-1, 1)
        if random.random() > 0.95:
            self.n_hidden3 = self.n_hidden3+random.randint(-1, 1)
        if random.random() > 0.95:
            self.n_hidden4 = self.n_hidden4+random.randint(-1, 1)
        if self.n_hidden4 < 2:
            self.n_hidden4 = random.randint(3, 4)

    def fitness(self) -> float:
        return calculator(dropoutP1=self.dropoutP1, dropoutP2=self.dropoutP2,
                          lr=self.lr, weight_decay=self.weight_decay,
                          n_hidden1=self.n_hidden1, n_hidden2=self.n_hidden2,
                          n_hidden3=self.n_hidden3, n_hidden4=self.n_hidden4)

    def recover(self):
        self.dropoutP1, self.dropoutP2, self.lr, self.weight_decay, self.n_hidden1, self.n_hidden2, self.n_hidden3, self.n_hidden4 = self.oldParameter


def inputToFile(dropoutP1: float, dropoutP2: float,
                lr: float, weight_decay: float,
                n_hidden1: int, n_hidden2: int,
                n_hidden3: int, n_hidden4: int, accuracy: float):  # 将较优的超参数以及结果输出到文件中
    with open('trainResults.txt', 'a+') as f:
        f.write('dropoutP1:'+str(dropoutP1)+' dropoutP2:'+str(dropoutP2)+' lr:'+str(lr)+' weight_decay:'+str(weight_decay) +
                ' n_hidden1:'+str(n_hidden1)+' n_hidden2:'+str(n_hidden2)+' n_hidden3:'+str(n_hidden3)+' n_hidden4:'+str(n_hidden4)+' acc:'+str(accuracy)+'\n')


stopT = 1e-12
beginT = 1500
nowT = 1500
lastT = -1
dertaT = 0
bestSolution = particle()
solution = particle()
bestResult = -1
lastResult = -1
iterTimes = 1000
decay = 0.99


def decayT():
    global lastT, nowT, dertaT
    lastT, nowT = nowT, nowT*decay
    dertaT = lastT-nowT


while True:
    while nowT > stopT:
        if lastT == -1:
            decayT()
        for i in range(iterTimes):
            tmpResult = solution.fitness()
            if tmpResult > lastResult:
                lastResult = tmpResult
                if tmpResult > bestResult:
                    bestResult = tmpResult
                    bestSolution = copy.deepcopy(solution)
                    inputToFile(bestSolution.dropoutP1, bestSolution.dropoutP2, bestSolution.lr, bestSolution.weight_decay,
                                bestSolution.n_hidden1, bestSolution.n_hidden2, bestSolution.n_hidden3, bestSolution.n_hidden4)
            elif random.random() < math.exp(dertaT/beginT):
                lastResult = tmpResult
            else:
                solution.recover()
        decayT()
