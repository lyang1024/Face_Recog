import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random

def feedforward(w,a,x):
	f = lambda s: 1/(1+np.exp(-s))

	w = np.array(w)
	temp = np.array(np.concatenate((a,x),axis=0))
	z_next = np.dot(w,temp)

	return f(z_next), z_next

def backprop(w,z,delta_next):
	f = lambda s: np.array(1/(1+np.exp(-s)))
	df = lambda s: f(s) * (1 - f(S))
	delta = df(z) * np.dot(w.T,delta_next)

	return delta

DataSet = scio.loadmat('yaleB_face_dataset.mat')
unlabeledData = DataSet['unlabeled_data']

dataset_size = 80
unlabeled_data = np.zeros(unlabeledData.shape)

for i in range(dataset_set):
	tmp = unlabeledData[:,i]/255.
	unlabeled_data[:,i] = (tmp - np.mean(tmp)) / np.std(tmp)

alpha = 0.5
max_epoch = 300
mini_batch = 10
height = 48
width = 42
imgSize = height * width

hidden_node = 60
hidden_layer = 2
layer_struc = [[imgSize,1],
				[0, hidden_node],
				[0, imgSize]]
layer_num = 3

#initialize variables of network
w = []
for l in range(layer_num-1):
	w.append(np.random.randn(layer_struc[l+1][1],sum(layer_struc[l])))

x = []
x.append(np.array(unlabeled_data[:,:]))
x.append(np.zeros((0,dataset_size)))
x.append(np.zeros((0,dataset_size)))

delta = []
for l in range(layer_num):
	delta.append([])

nRow = max_epoch / 100 + 1
nColumn = 4
eachFaceNum = 20

for iImg in range(nColumn):
	ax = plt.subplot(nRow, nColumn, iImg+1)
	plt.imshow(unlabeledData[:,eachFaceNum * iImg + 1].reshape((width,height)).T, cmap=plt.cm.gray)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

#start training

count = 0
print('Training start ...')
for ite in range(max_epoch):
	ind = list(range(dataset_size)
	random.shuffle(ind)

	a = []
	z = []
	z.append([])
	for i in range(int(np.ceil(dataset_size / mini_batch))):
		a.append(np.zeros((layer_struc[0][1],mini_batch)))
		x = []
		for l in range(layer_num):
			x.append(X[l][:,ind[i*mini_batch : min((i+1)*mini_batch, dataset_size)]])
		y = unlabeled_data[:,ind[i*mini_batch:min((i+1)*mini_batch,dataset_size)]]
		for l in range(layer_num-1):
			a.append([])
			z.append([])
			a[l+1],z[l+1] = feedforward(w[l],a[l],x[l])
		
		delta[layer_num-1] = np.array(a[layer_num-1] - y)*np.array(a[layer_num-1])
		delta[layer_num-1] = delta[layer_num-1]*np.array(1-a[layer_num-1])

		for l in range(layer_num-2,0,-1):
			delta[l] = backprop(w[l],z[l],delta[l+1])

		for l in range(layer_num-1):
			dw = np.dot(delta[l+1], np.concatenate((a[l],x[l]),axis=0).T) / mini_batch
			w[l] = w[l] - alpha * dw
	count += 1

	if np.mod(ite+1,100) == 0:
		b=[]
		b.append(np.zeros((layer_struc[0][1],datset_size)))

		for l in range(layer_num-1):
			tempA, tempZ = feedforward(w[l],b[l],X[l])
			b.append(tempA)

		for iImg in range(nColumn):
			ax = plt.subplot(nRow,nColumn, iImg + nColumn * (ite+1)/100+1)
			tmp = b[layer_num-1][:,eachFaceNum * iImg + 1]
			dis_result = ((tmp*np.std(tmp))+np.mean(tmp)).reshape(width,height).T
			plt.imshow(dis_result,cmap=plt.cm.gray)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		print('Learning epoch:', count, '/', max_epoch)
