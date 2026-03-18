"""
선형 회귀는
입력값 x 로부터 연속적인 값 
y 를 예측하는 가장 기본적인 모델

y_hat = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b
y_hat=Xw+b

“입력에 가중치를 곱해서 더하고, 마지막에 bias를 더하는 모델” 

모델의 예측을 실제 정답과 얼마나 다른지 계산

한 샘플에 대한 손실:
l_i(w, b) = (1/2) * (y_hat^(i) - y^(i))^2

전체 데이터 셋의 손실 : 
L(w, b) = (1/n) * sum_{i=1}^n (1/2) * (w^T x^(i) + b - y^(i))^2

학습의 목표는 이 손실 함수의 값이 가장 작아지는(평균 오차를 최소로 하는) w,b를 찾는 것이다.

경사하강법(SGD) 기반 방법을 사용한다.
- 파라미터를 임의로 초기화
- 데이터 중 일부를 미니배치로 구성
- 해당 미니배치에 대한 손실의 기울기(gradient)를 계산
- 손실이 줄어드는 방향으로 파라미터를 update
반복

하이퍼파라미터라는 개념이 주어지는데.
모델에 배우며 찾는 것이 아닌, 사람이 정해줘야하는 값이다.
학습률과 미니배치 크기 같은 것들이다.

이 과정을 통해 학습이 끝나면 새로운 w_hat과 b_hat를 얻게된다.
이것을 통해 학습에는 업던 새로운 입력 x에 대해 예측을 시작한다.
이를 "예측 혹은 추론"이라 부른다 prdeiction or infercence
훈련이 끝난 후 모델을 실제 입력에 적용하는 단계이다.
"""

"""
이러한 선형회귀도 사실 뉴럴 네트워크다.

입력 노드 x_1,...,x_d
출력 노드 1개
모든 입력은 출력에 연결 됨.
"""

# from mxnet import nd 
import torch
import time

# a = nd.ones(shape = 10000)
# b = nd.onese(shape = 10000)

a = torch.ones(10000)
b = torch.ones(10000)

start = time.time()
c = torch.zeros(10000)

#for문으로 하나씩 더한다.
for i in range(10000):
    c[i] = a[i] + b[i]
print(time.time() - start) #0.06500434875488281

#혹은 벡터 전체를 한 번에 더한다. - 벡터화!
# 모델 학습 및 예측을 수행할 때, 벡터 연산을 사용하고 이를 통해서 여러 값들은 한번에 처리
start = time.time()
d = a + b
print(time.time() - start) #0.0



import math
import matplotlib.pyplot as plt

x= torch.arange(-7, 7, 0.01) #x축 -7 ~ 7 , 간격 0.01

#(mean,std) 쌍
parameters = [(0,1), (0,2), (3,1)] # 3개의 정규 분포 (평균,표준편차)

plt.figure(figsize=(10,6))

# p(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-((x - mu)^2) / (2*sigma^2))
for mu,sigma in parameters:
    p = (1 / math.sqrt(2*math.pi * sigma**2)) * torch.exp(-(0.5 / sigma**2)*(x - mu)**2)
    plt.plot(x.numpy(),p.numpy(),label = f'mean {mu}, std {sigma}') 

plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Normal Distributions')
plt.show()