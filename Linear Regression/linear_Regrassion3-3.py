"""
이번 장(linear-regression-gluon)은
이걸 프레임워크의 고수준 API로 짧고 깔끔하게 구현하는 방법을 보여준다.
"""
import torch

num_inputs = 2
num_examples = 1000

true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

features = torch.randn(num_examples, num_inputs)
labels = features @ true_w + true_b
labels += torch.randn_like(labels) * 0.01

from torch.utils.data import TensorDataset, DataLoader

batch_size = 10
dataset = TensorDataset(features, labels) #하나의 데이터셋으로 묶음
data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True) #배치 단위로 꺼냄
#shuffle= True는 섞어서 학습하겠다는 것

for X,y in data_iter:
    print(X,y)
    break

from torch import nn
net = nn.Sequential(nn.Linear(2,1)) #입력2개, 출력1개의 선형변환 수행

from torch.nn import init
init.normal_(net[0].weight, mean=0.0, std=0.01)
init.zeros_(net[0].bias)

loss = nn.MSELoss()

from torch import optim
trainer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        pred = net(X)
        l = loss(pred, y.reshape(-1,1))

        trainer.zero_grad()
        l.backward()
        trainer.step()

    with torch.no_grad():
        epoch_loss = loss(net(features), labels.reshape(-1,1))
        print(f"epoch{epoch+1}, loss: {epoch_loss.item():.6f}")

w = net[0].weight.data
b = net[0].bias.data

print("estimated w:", w)
print("estimated b:", b)
print("error in estimating w:", true_w.reshape(1, -1) - w)
print("error in estimating b:", true_b - b)