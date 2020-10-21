import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from transCoding import *
import numpy as np
from confusionMatrix import *
import itertools

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data)
        init.xavier_normal(m.bias.data)


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_output):
        # n_features输入层神经元数量，也就是特征数量
        # n_hidden隐层神经元数量
        # n_output输出层神经元数量
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_features, n_hidden1)
        self.dropout1 = torch.nn.Dropout(p=0.06)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.dropout2 = torch.nn.Dropout(p=0.03)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.hidden5 = torch.nn.Linear(n_hidden4, n_hidden5)
        self.predict = torch.nn.Linear(n_hidden4, n_output)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.leaky_relu(self.hidden1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.hidden2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.hidden3(x))
        # x = self.dropout(x)
        x = F.leaky_relu(self.hidden4(x))
        # x = self.dropout(x)
        # x = F.relu(self.hidden5(x))
        x = self.predict(x)
        return x

###此处的n_hidden5并没有用上###


def calculator(dropoutP1=0.06, dropoutP2=0.03,
               lr=0.0009535, weight_decay=2.2e-6,
               n_hidden1=44, n_hidden2=29,
               n_hidden3=20, n_hidden4=12,
               n_hidden5=4, maxIter=10000) -> float:
    # resultTensor, dataTensor, n_features = transCoding()
    # resultTensor, dataTensor, resultTestTensor, dataTestTensor, n_features = underSamplingTransCoding()
    resultTensor, dataTensor, resultTestTensor, dataTestTensor, n_features = returnDealedData()
    # n_hidden1 = 22
    # n_hidden2 = 4
    n_output = 2
    net = Net(n_features, n_hidden1, n_hidden2,
              n_hidden3, n_hidden4, n_hidden5, n_output)
    net.apply(weights_init)
    if __name__ == "__main__":
        print('bpNeuralNetworksInfo:', net)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01,momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(
    ), lr=0.0009535, weight_decay=2.2e-6, eps=1e-8, betas=(0.9, 0.999), amsgrad=False)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, betas=(
    #     0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # lrScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    lrScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, 0)
    loss_func = torch.nn.CrossEntropyLoss()
    # plt.ion()
    accList = []
    oneRecallList = []
    zeroRecallList = []

    testAccList = []
    testOneRecallList = []
    testZeroRecallList = []

    iterTimes = []
    target_result = resultTensor.data.numpy()
    test_target_result = resultTestTensor.data.numpy()
    oneIndex = []
    zeroIndex = []
    for index in range(len(target_result)):
        if target_result[index] == 1:
            oneIndex.append(index)
        else:
            zeroIndex.append(index)

    testOneIndex = []
    testZeroIndex = []
    for index in range(len(test_target_result)):
        if test_target_result[index] == np.int64(1):
            testOneIndex.append(index)
        else:
            testZeroIndex.append(index)

    target_result_one = np.delete(target_result, zeroIndex)
    target_result_zero = np.delete(target_result, oneIndex)
    test_target_result_one = np.delete(test_target_result, testZeroIndex)
    test_target_result_zero = np.delete(test_target_result, testOneIndex)

    lossList = []

    if __name__ != '__main__':
        net.train()

    for t in range(maxIter):
        if __name__ == '__main__':
            print('times:', t)
            net.train()
        out = net(dataTensor.float())
        loss = loss_func(out, resultTensor.long())
        lossList.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lrScheduler.step()
        if __name__ == '__main__' and t % 2 == 0:
            prediction = torch.max(out, 1)[1]
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

            net.eval()
            out = net(dataTestTensor.float())
            # optimizer.zero_grad()
            prediction = torch.max(out, 1)[1]
            pred_result = prediction.data.numpy()
            accuracy = float((pred_result == test_target_result).astype(
                int).sum()) / float(test_target_result.size)
            one_recall = float((np.delete(pred_result, testZeroIndex) == test_target_result_one).astype(
                int).sum()) / float(test_target_result_one.size)
            zero_recall = float((np.delete(pred_result, testOneIndex) == test_target_result_zero).astype(
                int).sum()) / float(test_target_result_zero.size)
            testAccList.append(accuracy)
            testOneRecallList.append(one_recall)
            testZeroRecallList.append(zero_recall)

    # if __name__ != "__main__":

    if __name__ == '__main__':
        iterNpArray = np.array(iterTimes)
        ###训练准确率曲线####
        accNpArray = np.array(accList)
        plt.plot(iterNpArray, accNpArray, label='AccRate')

        ##信誉用户判断结果的训练召回率###
        oneRecallNpArray = np.array(oneRecallList)
        plt.plot(iterNpArray, oneRecallNpArray, label='TrustedUserRecallRate')

        ##失信用户判断结果的训练召回率###
        zeroRecallNpArray = np.array(zeroRecallList)
        plt.plot(iterNpArray, zeroRecallNpArray,
                 label='UntrustworthyUserRecallRate')

        ###测试准确率曲线####
        testAccNpArray = np.array(testAccList)
        plt.plot(iterNpArray, testAccNpArray, label='TestAccRate')

        ##信誉用户判断结果的测试召回率###
        testOneRecallNpArray = np.array(testOneRecallList)
        plt.plot(iterNpArray, testOneRecallNpArray,
                 label='TestTrustedUserRecallRate')

        ##失信用户判断结果的测试召回率###
        testZeroRecallNpArray = np.array(testZeroRecallList)
        plt.plot(iterNpArray, testZeroRecallNpArray,
                 label='TestUntrustworthyUserRecallRate')

        plt.legend(loc='upper right')
        plt.xlabel('iterTimes')
        plt.ylabel('rateValue')
        print('accuracy:', accList[-1])
        print('one_recall:', oneRecallList[-1])
        print('zero_recall:', zeroRecallList[-1])

        print('test_accuracy:', testAccList[-1])
        print('test_one_recall:', testOneRecallList[-1])
        print('test_zero_recall:', testZeroRecallList[-1])

        plt.text(2000, 0.6, 'TraningAcc:'+str(accList[-1]))
        plt.text(2000, 0.5, 'TraningOneRecall:'+str(oneRecallList[-1]))
        plt.text(2000, 0.4, 'TraningZeroRecall:'+str(zeroRecallList[-1]))
        plt.text(2000, 0.3, 'TestAcc:'+str(testAccList[-1]))
        plt.text(2000, 0.2, 'TestOneRecall'+str(testOneRecallList[-1]))
        plt.text(2000, 0.1, 'TestZeroRecall'+str(testZeroRecallList[-1]))

        plt.show()
        plot_confusion_matrix(np.array([[testOneRecallList[-1], 1-testOneRecallList[-1]], [
                              1-testZeroRecallList[-1], testZeroRecallList[-1]]]), [u'正常用户', u'失信用户'])

        # loss随着迭代的图
        lossNpArray = np.array(lossList)
        iterNpArray = np.array([i for i in range(maxIter)])
        plt.xlabel('iterTimes')
        plt.ylabel('loss')
        plt.plot(iterNpArray, lossNpArray,)
        plt.show()

        PATH = "./net.pt"
        torch.save(net.state_dict(), PATH)
        return 0
    else:
        net.eval()
        out = net(dataTestTensor.float())
        # optimizer.zero_grad()
        prediction = torch.max(out, 1)[1]
        pred_result = prediction.data.numpy()
        accuracy = float((pred_result == test_target_result).astype(
            int).sum()) / float(test_target_result.size)
        # dropoutP1=0.04, dropoutP2=0.04,
        #    lr=0.0009535, weight_decay=2.2e-6,
        #    n_hidden1=44, n_hidden2=29,
        #    n_hidden3=20, n_hidden4=12,
        #    n_hidden5=4,maxIter = 5000
        return accuracy


if __name__ == '__main__':
    calculator()
