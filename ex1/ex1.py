import numpy as np
import matplotlib.pyplot as plt
import copy

# 2.1 Plotting the Data
data = np.loadtxt("ex1data1.txt", delimiter=',')
x = data[:, 0]
y = data[:, 1]
# print(x, y)
plt.subplot(2, 1, 1)
plt.title("Scatter plot of training data")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.plot(x, y, ".m")
# plt.show()
# print(data[:][0])

# 2.2 Gradient
n = x.shape
x = np.resize(x, (n[0], 1))
x = np.insert(x, 0, [x/x for x in range(1, n[0]+1)], axis=1)
theta = np.ones(2)  # 迭代参数两个

iterations = 1500
alpha = 0.01

def compute_cost(x, y, theta):
	'''
	计算损失函数
	:param x: n rows and m columns
	:param y: n rows and 1 columns
	:param theta: 1 row and 2 columns
	:return: cost of cost function
	'''
	n = y.shape[0]
	return float(1/(2*n)*(np.dot((np.dot(x,theta)-y),(np.dot(x,theta)-y))))


def gradient_descent(x, y, theta = np.zeros(2)):
	'''
	梯度下降法
	:param x: n rows and m columns
	:param y: n rows and 1 columns
	:param theta: m row and 1 columns
	:return: final theta
	'''
	nowTheta = copy.deepcopy(theta)
	Jval = []
	thetaHistory = []
	m = y.shape[0]
	for num in range(iterations):
		tempTheta = copy.deepcopy(nowTheta)
		Jval.append(compute_cost(x, y, tempTheta))
		thetaHistory.append(nowTheta.tolist())
		for i in range(0,len(theta)):
			nowTheta[i] = nowTheta[i] - alpha/m*np.sum((np.dot(x, tempTheta)-y)*x[:,i].T)

	return nowTheta, Jval, thetaHistory


def plotConverge(val):
	plt.figure(figsize=(5,5))
	plt.plot(range(len(val)), val, 'bo')
	plt.grid()
	plt.title("converge of cost function")
	plt.xlabel("Iteration times")
	plt.ylabel("cost function val")
	# plt.show()


resTheta, jav, thetaHistory = gradient_descent(x, y, theta)

# print(resTheta)

def plotRes(resTheta):
	x1 = np.arange(5, 22)
	y1 = resTheta[0] + resTheta[1]*x1
	plt.title("Scatter plot of training data")
	plt.xlabel("Population of City in 10,000s")
	plt.ylabel("Profit in $10,000s")
	plt.plot(x1, y1, "-b")
	# plt.show()


# plotRes(resTheta)
# plotConverge(jav)

# 3d绘图
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xvals = np.arange(-10, 10, 0.5)
yvals = np.arange(-1, 4, 0.1)
myx, myy, myz = [], [], []
# print(xvals, yvals)
for xv in xvals:
	for yv in yvals:
		myx.append(xv)
		myy.append(yv)
		res = compute_cost(x, y, np.array([xv, yv]))
		# print("cost:", res, " theta0:", xv, " theta1:", yv)
		myz.append(res)

myx, myy, myz = np.array(myx), np.array(myy), np.array(myz)

ax.scatter(xs=myx, ys=myy, zs=myz, c='r', marker='.')
ax.scatter([x[0] for x in thetaHistory], [x[1] for x in thetaHistory], jav, c='b', marker='o')
# plt.show()


# 3 multiple gradient
# 3.1特征归一化
def feature_normalization(feature):
	total = 0
	new_feature = []
	biggest = -1
	smallest = 10000
	for item in feature:
		total = total + item
		if item > biggest:
			biggest = item
		if item < smallest:
			smallest = item
	avg = total/len(feature)
	div = 0
	for item in feature:
		div = div + (item - avg)**2
	div = div/len(feature)
	div = np.sqrt(div)
	for item in feature:
		new_feature.append((item-avg)/div)
	return div, avg, new_feature


iterations2 = 1500
alpha2 = 0.01

def mul_compute_cost(x, y, theta):
	'''
	计算损失函数
	:param x: n rows and m columns
	:param y: n rows and 1 columns
	:param theta: m row and 1 columns
	:return: cost of cost function
	'''
	n = y.shape[0]
	return float(1/(2*n)*(np.dot((np.dot(x,theta)-y),(np.dot(x,theta)-y))))


def multiple_gradient_descent(x, y, theta = np.zeros(2)):
	'''
	梯度下降法
	:param x: n rows and m columns
	:param y: n rows and 1 columns
	:param theta: m row and 1 columns
	:return: final theta
	'''
	nowTheta = copy.deepcopy(theta)
	Jval = []
	thetaHistory = []
	m = y.shape[0]
	for num in range(iterations2):
		tempTheta = copy.deepcopy(nowTheta)
		Jval.append(mul_compute_cost(x, y, tempTheta))
		thetaHistory.append(nowTheta.tolist())
		for i in range(0,len(theta)):
			nowTheta[i] = nowTheta[i] - alpha2/m*np.sum((np.dot(x, tempTheta)-y)*x[:, i].T)

	return nowTheta, Jval, thetaHistory




data2 = np.loadtxt("ex1data2.txt", delimiter=',')
x2 = data2[:, 0:2]
y2 = data2[:, 2]
divx, avgx = [], []
m, n = np.shape(x2)
for i in range(n):
	cur_div1, cur_avg1, x2[:, i] = feature_normalization(x2[:, i])
	divx.append(cur_div1)
	avgx.append(cur_avg1)
divy, avgy, y2 = feature_normalization(y2)
# print(divx, avgx, x2)
# print(divy, avgy, y2)

x2 = np.array(x2)
x2 = np.insert(x2, 0, [x/x for x in range(1, len(x2)+1)], axis=1)

y2 = np.array(y2)
theta2 = np.zeros(len(x2[0]))

new_theta2, jval2,  thetaHistory2 = multiple_gradient_descent(x2, y2, theta2)

# cost function converge figure
fg = plt.figure(figsize=(5, 5))
plt.title("converge of cost function")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.plot(range(len(jval2)), jval2, "bo")
# plt.show()

print(new_theta2)

predict = [1, 1650, 3] # size=1650 bedroom=3 predict
for i in range(1, len(predict)):
	predict[i] = (predict[i] - avgx[i-1])/divx[i-1]
pre_res = np.dot(predict, new_theta2)
final_res = pre_res*divy + avgy
print("normalization predict value:", pre_res, "authentic predict value:", final_res)
