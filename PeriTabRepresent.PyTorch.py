import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

with open ('heusler') as fin:
    lines = fin.readlines()

ndata = len(lines)

pt = [[-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1],
      [-0.1, -0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.1, -0.1, -0.1, -0.1],
      [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
      [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
      [-0.1, -0.1,  0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]]

pt = np.array(pt)
x = [pt for i in range(ndata)]
x = np.array(x)
y = np.zeros(shape=(ndata),dtype=float)

pt_pos={ 'Li': [0, 0], 'Be': [0, 1],  'B': [0, 12],  'C': [0, 13],  'N': [0, 14],  'O': [0, 15],
         'Na': [1, 0], 'Mg': [1, 1], 'Al': [1, 12], 'Si': [1, 13],  'P': [1, 14],  'S': [1, 15],
          'K': [2, 0], 'Ca': [2, 1], 'Ga': [2, 12], 'Ge': [2, 13], 'As': [2, 14], 'Se': [2, 15],
         'Rb': [3, 0], 'Sr': [3, 1], 'In': [3, 12], 'Sn': [3, 13], 'Sb': [3, 14], 'Te': [3, 15],
         'Cs': [4, 0], 'Ba': [4, 1], 'Tl': [4, 12], 'Pb': [4, 13], 'Bi': [4, 14],

         'Sc': [2, 2], 'Ti': [2, 3],  'V': [2, 4], 'Cr': [2, 5], 'Mn': [2, 6], 'Fe': [2, 7], 'Co': [2, 8], 'Ni': [2, 9], 'Cu': [2, 10], 'Zn': [2, 11], 
          'Y': [3, 2], 'Zr': [3, 3], 'Nb': [3, 4], 'Mo': [3, 5], 'Tc': [3, 6], 'Ru': [3, 7], 'Rh': [3, 8], 'Pd': [3, 9], 'Ag': [3, 10], 'Cd': [3, 11], 
                       'Hf': [4, 3], 'Ta': [4, 4],  'W': [4, 5], 'Re': [4, 6], 'Os': [4, 7], 'Ir': [4, 8], 'Pt': [4, 9], 'Au': [4, 10], 'Hg': [4, 11]
}

ii = 0
for line in lines:
    s = line.split(' ')
    for i in range(3):
        x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 1.4
        if s[i][-1] == '2' :
            x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 2.8
    y[ii] = float(s[3].rstrip())
    ii+=1

x = np.expand_dims(x, axis = 1)
y = preprocessing.scale(y)
y = np.expand_dims(y, axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=50000, random_state=42)
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
trainset = TensorDataset(x_train, y_train)
trainloader = DataLoader(trainset, batch_size=5000)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 3, 1, 1)
        self.conv2 = nn.Conv2d(96, 96, 5, 1, 1)
        self.conv3 = nn.Conv2d(96, 96, 3)
        self.fc1 = nn.Linear(96 * 12 * 1, 192)
        self.fc2 = nn.Linear(192, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 96 * 12 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.005, momentum=0.85, nesterov=True)

for epoch in range(1000):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, target = data
        optimizer.zero_grad()
        loss = criterion(net(inputs), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 4:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

print('Finished Training')








