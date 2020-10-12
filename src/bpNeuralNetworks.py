import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transCoding import transCoding
import numpy as np


resultTensor, dataTensor, n_features = transCoding()
n_hidden = 10
n_output = 2


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        # n_features输入层神经元数量，也就是特征数量
        # n_hidden隐层神经元数量
        # n_output输出层神经元数量
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_features, n_hidden, n_output)
print('bpNeuralNetworksInfo:', net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
# plt.ion()
accList = []
oneRecallList = []
zeroRecallList = []
iterTimes = []
maxIter = 30
target_result = resultTensor.data.numpy()
oneIndex = []
zeroIndex = []
for index in range(len(target_result)):
    if target_result[index] == 1:
        oneIndex.append(index)
    else:
        zeroIndex.append(index)

target_result_one = np.delete(target_result, zeroIndex)
target_result_zero = np.delete(target_result, oneIndex)

for t in range(maxIter):
    out = net(dataTensor.float())
    loss = loss_func(out, resultTensor.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        prediction = torch.max(out, 1)[1]
        print(prediction)
        pred_result = prediction.data.numpy()
        accuracy = float((pred_result == target_result).astype(
            int).sum()) / float(target_result.size)
        one_recall = float((np.delete(pred_result, zeroIndex) == target_result_one).astype(
            int).sum()) / float(target_result_one.size)
        zero_recall = float((np.delete(pred_result, oneIndex) == target_result_zero).astype(
            int).sum()) / float(target_result_zero.size)
        accList.append(accuracy)
        oneRecallList.append(one_recall)
        zeroRecallList.append(zero_recall)
        iterTimes.append(t)


iterNpArray = np.array(iterTimes)
####准确率曲线####
accNpArray = np.array(accList)
plt.plot(iterNpArray, accNpArray,label='accRate')

###信誉用户判断结果的召回率###
oneRecallNpArray = np.array(oneRecallList)
plt.plot(iterNpArray, oneRecallNpArray,label='TrustedUserRecallRate')


###失信用户判断结果的召回率###
zeroRecallNpArray = np.array(zeroRecallList)
plt.plot(iterNpArray, zeroRecallNpArray,label='UntrustworthyUserRecallUser')

plt.legend(loc = 'upper right')
plt.xlabel('iterTimes')
plt.ylabel('rateValue')
plt.show()
