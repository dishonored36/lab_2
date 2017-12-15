import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_svmlight_file


def loaddataset():
    # 读取数据
    x_train, y_train = load_svmlight_file("a9a.txt")
    x_validation, y_validation = load_svmlight_file("a9a(testing).txt")

    print(x_train, y_train)
    return x_train, x_validation, y_train, y_validation


def gradientDescent(alpha, maxCycles, X_data, y_data):

    num = y_data.shape[0]    #样本数量
    # 线性模型参数正态分布初始化
    w = np.random.normal(size=(X_data.shape[1]))
    b = np.random.normal(size=1)
    losses = []

    # 迭代次maxCycles次
    for n in range(maxCycles):
	    grad_w = np.zeros(X_data.shape[1])
	    grad_b = np.zeros(1)
	    loss = 0
	    for i in range(num):
		    y = 1./(1 + np.exp(-np.dot( X_data[i].data, w ) + b))
		    loss += np.power((y - y_data[i]),2) / ( 2 * num)
		    grad_w += ( y - y_data[i] ) * X_data[i].data / num
		    grad_b += ( y - y_data[i] ) / num
	    # 更新模型参数
	    w -= alpha * grad_w
	    b -= alpha * grad_b
	    losses.append(loss)
	    print("loss = %f" % loss)
    return losses


def plotLossPerTime(n, losses_train, losses_validation):
	plt.xlabel('iteration times')
	plt.ylabel('loss of validation')
	plt.title('linear regression & gradient decrease')
	n_cycles = range(1,n+1)
	plt.plot(n_cycles, losses_train, label = "Loss of Train", color='blue', linewidth=3)
	plt.plot(n_cycles, losses_validation, label = "Loss of Validation", color='red', linewidth=3)
	plt.legend(loc=0)
	plt.grid()
	plt.show()


# main
X_train, X_validation, y_train, y_validation = loaddataset()
alpha = 0.01
maxCycles = 8
losses_train = gradientDescent(alpha, maxCycles, X_train, y_train)
losses_validation = gradientDescent(alpha, maxCycles, X_validation, y_validation)
plotLossPerTime(maxCycles, losses_train, losses_validation)

