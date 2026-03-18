"""
선형 회귀를 프레임워크 도움 없이 직접 구현하기
이 장의 대략적인 순서
1. 가짜 데이터 생성
2. 미니배치로 데이터 읽기
3. 파라미터 초기화
4. 선형 모델 정의
5. 제곱손실 정의
6. SGD로 학습
7. 학습된 𝑤,𝑏가 진짜 값과 얼마나 가까운지 확인
"""

from re import L

import torch
#data
num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2.0,-3.4])
true_b = 4.2

features = torch.randn(num_examples, num_inputs) # mean =0, std = 1인 정규분포 (1000,2)크기의 행렬
labels = features @ true_w + true_b # y = Xw + b 형태 선형식
# @는 행렬곱 연산자 이므로 알아두자.
labels += torch.randn_like(labels)* 0.01 #잡음 0.01을 더해 현실적인 데이터처럼

#===================================
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))
# plt.scatter(features[:,1].numpy(),labels.numpy(), s=5) #행 전체 두번째 특성, 
# plt.xlabel("feature 2")
# plt.ylabel("label")
# plt.title("Synthetic Linear Regression Data")
# plt.show()
#===================================
import random
# data ierator
def data_iter(batch_size, features, labels): #역할은 전체 데이터를 batch_size 크기로 잘라서 하나씩 내보내는 것.
    num_examples = len(features) #샘플 개수에 features의 길이 :행의 개수를 저장
    indices = list(range(num_examples)) #0부터 num_exampels -1까지 인덱스 리스트
    random.shuffle(indices) # 인덱스를 무작위로 섞음 -> 매 epoch마다 데이터 순서를 바꿔 학습 편향을 줄인다.

    for i in range(0, num_examples, batch_size):
        batch_indices =  torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] #yield를 서용하여 현재 배치의 입력과 라벨을 반환
# dataLoader를 사용하지 않고 SGD 학습에 사용할 미니배치 로더를 구현 - 개념 이해를 위함.

#테스트
batch_size = 10

for x,y in data_iter(batch_size, features, labels):
    print(x.shape, y.shape)
    break

#parameters
w = torch.normal(0, 0.01, size = (num_inputs, 1), requires_grad = True)
b = torch.zeros(1, requires_grad= True)

#model
def linreg(X,w,b):
    return X @ w + b
#loss
def squared_loss(y_hat, y):
    return (y_hat.reshape(y.shape) - y) ** 2 / 2 #reshape(y.shape)를 통해 y_hate을 y와 같은 shape로 맞춘 뒤 빼는 것임. 선형모델 출력의 shape 불일치가 나타날 수 있기 때문

#optimizer sgd 직접 구현
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#training
lr = 0.03
num_epochs = 3
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size,features,labels):
        l =squared_loss(linreg(X,w,b),y).sum() #예측 y_hat을 만들고, 각 샘플별 sum을 통해 손실 L을 만드는 과정
        l.backward() #역전파로 미분을 전파하며 각각의 기울기를 구해 w.grad, b.grad에 저장
        sgd([w,b],lr,batch_size)

    with torch.no_grad():
        train_l = squared_loss(linreg(features,w,b),labels).mean()
        print(f"epoch {epoch+1}, loss {float(train_l):.6f}")

print("estimated w:", w.reshape(-1))
print("estimated b:", b)
print("true w:", true_w)
print("true b:", true_b)
        