import os
import numpy as np
import tarfile
import csv
from shutil import copyfile
import matplotlib.pyplot as plt
from scipy import misc


class DVS:
    # TODO add label list and check how data should be reshaped
    def __init__(self):
        self.url  = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/")
        self.name = 'CharDVS.tar.gz'
        self.working_dir  = os.path.join(os.path.dirname(__file__), "out/")

        # Extract the file
        copyfile(self.url+self.name, os.path.join(self.working_dir, self.name))
        tar = tarfile.open(os.path.join(self.working_dir, self.name), "r:gz")
        tar.extractall(self.working_dir)

        self.datafilenames = []
        self.labels = []
        lbl_filepath = os.path.join(self.working_dir, "CharDVS_data", "CharDVS_labels.csv")
        if os.path.exists(lbl_filepath):
            with open(lbl_filepath, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    self.datafilenames.append(row[0])
                    self.labels.append(row[1])
        else:
            print("Failed to find labels file " + lbl_filepath)

        self.events = []
        for fn in self.datafilenames[:]:
            fname = os.path.join(self.working_dir, "CharDVS_data", fn)
            if os.path.exists(fname):
                self.events.append(np.genfromtxt(fname, dtype=np.int32, delimiter=','))
            else:
                print("Failed to find data file " + fname)


dvs = DVS()
dvs_train = dvs.events[:36]
dvs_test = dvs.events[36:72]
dvs_labels = dvs.labels[:36]
dvs_labels_1 = dvs.labels[36:72]

print(dvs_train[1].shape)
print(dvs_test[1].shape)
print(dvs_labels)
print(dvs_labels_1)
dvs_sz = (32, 32)
source_img = None
label = []
t = 0

# Training data sets
for i in range(0, 36):

    num = int(dvs_train[i].shape[0]/150)
    row = num * 150
    dvs_train[i] = dvs_train[i][:row, :]

    dvs_train_split = np.vsplit(dvs_train[i], num)
    print(len(dvs_train_split))
    for j in range(0, len(dvs_train_split)):
        test_img = np.zeros(dvs_sz, dtype=np.int32)
        xx = dvs_train_split[j][:, 0]
        yy = dvs_train_split[j][:, 1]
        for k in range(0, 150):
            test_img[yy[k], xx[k]] += 1

        pix = np.array(test_img)
        print(pix.shape)
        if source_img is None:
            source_img = [np.ndarray.flatten(pix)]
        else:
            source_img = np.append(source_img, [np.ndarray.flatten(pix)], axis=0)
        misc.imsave(('DVS_figure/%3d.png'%t), test_img)
        label.append(i)
        t = t+1
        # plt.imshow(test_img, cmap='Greys_r')
        # plt.show()

img_out = source_img
print(img_out.shape)
label_new = np.array(label)
file = open('label_train.txt', 'w')
for k in label:
    file.write(str(k))
    file.write('\n')

np.save('DVS_datasets/x_train', img_out)
np.save('DVS_datasets/x_label', label_new)
print('Write finished')


x = np.load('DVS_datasets/x_train.npy')
implot = plt.imshow(np.reshape(x[1000], [32, 32]), cmap=plt.get_cmap('gray'))
print(np.shape(x))
print(np.shape(np.load('DVS_datasets/x_label.npy')))


plt.title('Displaying a packet of %i events' % 150)
plt.show()


# # Testing data sets
# for i in range(36, 72):
#
#     num = int(dvs_train[i].shape[0]/150)
#     row = num * 150
#     dvs_train[i] = dvs_train[i][:row, :]
#
#     dvs_train_split = np.vsplit(dvs_train[i], num)
#     print(len(dvs_train_split))
#     for j in range(0, len(dvs_train_split)):
#         test_img = np.zeros(dvs_sz, dtype=np.int32)
#         xx = dvs_train_split[j][:, 0]
#         yy = dvs_train_split[j][:, 1]
#         for k in range(0, 150):
#             test_img[yy[k], xx[k]] += 1
#
#         pix = np.array(test_img)
#         print(pix.shape)
#         if source_img is None:
#             source_img = [np.ndarray.flatten(pix)]
#         else:
#             source_img = np.append(source_img, [np.ndarray.flatten(pix)], axis=0)
#         misc.imsave(('DVS_test/%3d.png'%t), test_img)
#         label.append(i - 36)
#         t = t+1
#         # plt.imshow(test_img, cmap='Greys_r')
#         # plt.show()
#
# img_out = source_img
# print(img_out.shape)
# label_new = np.array(label)
# file = open('label_test.txt', 'w')
# for k in label:
#     file.write(str(k))
#     file.write('\n')
#
#
# np.save('DVS_datasets/y_train', img_out)
# np.save('DVS_datasets/y_label', label_new)
# print('Write finished')
#
#
# x = np.load('DVS_datasets/y_train.npy')
# implot = plt.imshow(np.reshape(x[1000], [32, 32]), cmap=plt.get_cmap('gray'))
# print(np.shape(x))
# print(np.shape(np.load('DVS_datasets/y_label.npy')))
#
#
# plt.title('Displaying a packet of %i events' % 150)
# plt.show()
