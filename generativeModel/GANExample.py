#GAN Example 1
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_real_data(batch_size):
    """
    8개의 중심이 있는 2D 데이터셋을 생성하는 함수
    각 중심은 원형으로 배치되어 있으며, 각 중심에서 약간의 노이즈가 추가된 데이터 포인트를 생성
    Args:
        batch_size (int): 생성할 데이터 포인트의 수
    Returns:
        torch.Tensor: 생성된 데이터 포인트를 포함하는 텐서
    """
    centers = []
    for k in range(8): # 반지름이 2인 원 위에 8개의 중심이 배치되어 있다고 생각해보자.
        angle = 2 * math.pi * k / 8
        centers.append((2 * math.cos(angle), 2 * math.sin(angle)))
    centers = torch.tensor(centers, dtype=torch.float32)

    idx = torch.randint(0, 8, (batch_size,))
    chosen_centers = centers[idx]
    noise = torch.randn(batch_size, 2) * 0.1  # 각 데이터 포인트에 약간의 노이즈를 추가하여 다양성을 높임
    return chosen_centers + noise   # 8개의 클러스터를 갖는 분포를 반환하게 될 것임.

"""
    Generator와 Discriminator 클래스 정의에서 파라미터 형태가 다른 이유는 뭐죠?

Generator는 잠재 공간에서 샘플을 받아서 실제 데이터 공간으로 매핑하는 역할을 합니다. 
따라서 Generator의 입력은 잠재 공간의 차원(z_dim)이고, 출력은 실제 데이터 공간의 차원(x_dim)입니다.
---------------------------------------------------------------------------------------------
반면에 Discriminator는 실제 데이터와 생성된 데이터를 구분하는 역할을 합니다. 
따라서 Discriminator의 입력은 실제 데이터 공간의 차원(x_dim)이고, 출력은 이진 분류 결과(1차원)입니다.
----------------------------------------------------------------------------------------------
이러한 구조는 GAN의 기본 원리에 따라 Generator가 실제 데이터와 유사한 데이터를 생성하도록 학습되고, 
Discriminator가 실제 데이터와 생성된 데이터를 구분하도록 학습되는 방식에 맞춰져 있습니다.
"""
class Generator(nn.Module):
    # 잠재벡터 z_dim을 입력으로 받아서 실제 데이터 공간 x_dim으로 매핑
    def __init__(self, z_dim=2, hidden_dim=128, x_dim=2):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    # 실제 데이터 공간 x_dim을 입력으로 받아서 이진 분류 결과(1차원)로 매핑
    def __init__(self, x_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
    
z_dim = 2
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

g_optimizer = optim.Adam(G.parameters(), lr=1e-3)
d_optimizer = optim.Adam(D.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss는 시그모이드 활성화 함수와 이진 교차 엔트로피 손실을 결합한 손실 함수입니다.
# 시그모이드 활성화 함수는 Discriminator의 출력이 0과 1 사이의 확률로 해석될 수 있도록 합니다.
#바이너리 크로스 엔트로피 손실은 Discriminator가 실제 데이터와 생성된 데이터를 올바르게 분류하도록 학습하는 데 사용됩니다.

epochs = 5000
batch_size = 256

for step in range(epochs):
    real = sample_real_data(batch_size).to(device)
    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    d_real = D(real) # Discriminator가 실제 데이터(real)를 입력으로 받아서 출력하는 값
    d_fake = D(fake.detach()) # Discriminator가 생성된 데이터(fake)를 입력으로 받아서 출력하는 값

    #  =========================중요한 것이라고 함.============================
    # detach()는 fake 텐서를 그래프에서 분리하여 Discriminator가 업데이트될 때 Generator의 가중치가 영향을 받지 않도록 합니다.
    # 근데 real을 배울때는 상관 없나요? real은 진짜 데이터니까요.
    # 네, 맞습니다. real 데이터는 실제 데이터이므로 Discriminator가 이를 학습하는 데 영향을 주지 않습니다.
    # 반면에 fake 데이터는 Generator가 생성한 데이터이므로 Discriminator가 이를 학습할 때 Generator의 가중치가 업데이트되지 않도록 detach()를 사용하여 그래프에서 분리하는 것이 중요합니다.

    d_loss_real = criterion(d_real, real_labels) # Discriminator가 실제 데이터를 진짜로 분류하도록 학습하는 손실
    d_loss_fake = criterion(d_fake, fake_labels) # Discriminator가 생성된 데이터를 가짜로 분류하도록 학습하는 손실
    d_loss = d_loss_real + d_loss_fake # Discriminator의 총 손실은 실제 데이터와 생성된 데이터에 대한 손실의 합입니다.

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    z = torch.randn(batch_size, z_dim).to(device)
    fake = G(z)
    d_fake_for_g = D(fake)
    g_loss = criterion(d_fake_for_g, real_labels) # Generator가 생성된 데이터를 실제로 분류하도록 학습하는 손실입니다.

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:4d} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

with torch.no_grad():
    real = sample_real_data(2000).cpu()
    fake = G(torch.randn(2000, z_dim).to(device)).cpu()

plt.figure(figsize=(6, 6))
plt.scatter(real[:, 0], real[:, 1], s=10, alpha=0.5, label="Real")
plt.scatter(fake[:, 0], fake[:, 1], s=10, alpha=0.5, label="Generated")
plt.legend()
plt.title("GAN: Real vs Generated 2D Samples")
plt.axis("equal")
plt.show()