import cv2
import numpy as np

dir_path = "/home/fuyan/kaggle/CIFAR_10/data/train/"
train_label_path = "/home/fuyan/kaggle/CIFAR_10/trainLabels.csv"
train_label = file(train_label_path, 'r') 
train = train_label.readlines()

label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for j in [10001, 20001, 30001, 40001, 50001]:
		f = file('batch_' + str(j / 10000) + '.bin', 'wb')
		for i in range(j - 10000, j):
			label = train[i].strip().split(',')[-1]
			label_index = label_list.index(label)
			img = cv2.imread(dir_path + str(i) + '.png')
			l = np.transpose(img, [2, 0, 1])
			l = np.reshape(l, [3072])
			f.write(chr(label_index))
			for temp in l:
				f.write(chr(temp))
			print i
		f.close()