
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

with open ('Data/Heusler_energy') as fin:
    lines = fin.readlines()
ndata=len(lines)

pt = [[-0.01, -0.01,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, -0.01, -0.01, -0.01, -0.01],
      [-0.01, -0.01,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, -0.01, -0.01, -0.01, -0.01],
      [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
      [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
      [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]]

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
        x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 0.2
        if s[i][-1] == '2' :
            x[ii][pt_pos[s[i][:-1]][0]][pt_pos[s[i][:-1]][1]] = 0.4
    y[ii] = float(s[3].rstrip())
    ii+=1

#np.set_printoptions(threshold=np.nan)
#x = x.tolist()
#with open ('haha','w') as fout:
#    print(x,file=fout)

model = keras.Sequential([
  keras.layers.ZeroPadding2D(padding=(1, 1), input_shape=(5, 16, 1)),
  keras.layers.Conv2D(96, (3, 3), activation='relu'),
  keras.layers.ZeroPadding2D(padding=(1, 1)),
  keras.layers.Conv2D(96, (5, 5), activation='relu'), 
  keras.layers.Conv2D(96, (3, 3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(192, activation='relu'),
  keras.layers.Dense(1)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae'])

x = np.expand_dims(x, axis =3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
y_train_scaled = preprocessing.scale(y_train)
y_test_scaled = preprocessing.scale(y_test)
print(y_test.std())

model.fit(x_train, y_train_scaled, epochs=100, batch_size=32, validation_data=(x_test, y_test_scaled), verbose=2)
#test_loss = model.evaluate(x_test, y_test_scaled)
#print(np.array(test_loss) * y_test.std() + y_test.mean())

#model.save('ptr.h5')

