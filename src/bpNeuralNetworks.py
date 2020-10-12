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
print('bpNewInfo:', net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
# plt.ion()
accList = []
iterTimes = []
maxIter = 300
for t in range(maxIter):
    out = net(dataTensor.float())
    loss = loss_func(out, resultTensor.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        prediction = torch.max(out, 1)[1]
        pred_result = prediction.data.numpy()
        target_result = resultTensor.data.numpy()
        accuracy = float((pred_result == target_result).astype(
            int).sum()) / float(target_result.size)
        accList.append(accuracy)
        iterTimes.append(t)
itemNpArray = np.array(iterTimes)
accNpArray = np.array(accList)
plt.plot(itemNpArray, accNpArray)
plt.xlabel('iterTimes')
plt.ylabel('accRate')
plt.show()
