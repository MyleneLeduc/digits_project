import numpy as np
from data import DigitsDataset
from model import ClassifierNN
import torch.nn as nn
import torch
import pdb

train_df = DigitsDataset(file_name = "optdigits.tra")
test_df = DigitsDataset(file_name = "optdigits.tes")

mymodel = ClassifierNN(input_features=64, hidden_layer1=25, hidden_layer2=30, output_features=10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.01)

epochs = 100
losses = []
#train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
X_train = train_df.X
y_train = train_df.y
X_test = test_df.X
y_test = test_df.y

for i in range(epochs):
    y_pred = mymodel.forward(X_train)
    y_train = y_train.to(dtype=torch.long)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### Evaluate the model ###

preds = []
with torch.no_grad():
    for val in X_test:
        y_hat = mymodel.forward(val)
        preds.append(y_hat.argmax().item())


y_test = np.array(y_test)
preds = np.array(preds)
correct = (y_test == preds)
correct = correct.astype(int)
erreur = correct.sum()/len(correct)
print("Mymodel predictor : Erreur : {:.4f}".format(erreur))

pdb.set_trace()

