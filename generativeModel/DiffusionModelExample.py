import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def sample_real_data(batch_size):
    centers = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        centers.append([2 * math.cos(angle), 2 * math.sin(angle)])
    centers = torch.tensor(centers, dtype=torch.float32)

    idx = torch.randint(0, 8, (batch_size,))
    chosen_centers = centers[idx]
    noise = 0.1 * torch.randn(batch_size, 2)
    return chosen_centers + noise


T = 100 # 노이즈를 100단계에 걸쳐 점진적으로 추가하는 과정에서의 단계 수라고 보면 됨
betas = torch.linspace(1e-4, 0.02, T).to(device) # betas는 각 단계에서 추가되는 노이즈의 양을 나타내는 벡터
# betas는 0.0001에서 0.02까지 선형적으로 증가하는 T개의 값을 가지며, 각 값은 해당 단계에서 추가되는 노이즈의 양을 나타낸다고 보면 됨.
alphas = 1.0 - betas # alphas는 각 단계에서 남겨지는 원래 데이터의 비율을 나타내는 벡터
alpha_bars = torch.cumprod(alphas, dim=0)
# 중요!
# alpha_bars는 각 단계에서 남겨지는 원래 데이터의 비율을 "누적 곱"으로 계산한 벡터
# 각 시점 t까지 왔을때 원래 데이터 x0가 누적해서 얼마나 남아있는지를 나타내는 벡터라고 보면 됨.

# 초반에는 alpha_bars가 1에 가까워서 원래 데이터가 거의 남아있지만, 단계가 진행됨에 따라 alpha_bars가 감소하여 원래 데이터가 점점 더 노이즈로 대체된다고 보면 됨.
# x0에서 바로 xt를 만들 수 있는 공식이
#  xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise 라는 것을 알 수 있음.


class TimeEmbedding(nn.Module):
    """
        현재 시점 t가 몇번째 단계인지를 나타내는 벡터를 생성하는 클래스
        시간 t를 모델이 이해할 수 있는 형태로 변환하는 역할을 한다.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :] # t는 (batch_size,) 형태의 텐서이고, freqs는 (half,) 형태의 텐서입니다. t[:, None]는 t를 (batch_size, 1) 형태로 변환하고, freqs[None, :]는 freqs를 (1, half) 형태로 변환하여 두 텐서가 브로드캐스팅되어 곱셈이 수행됩니다.
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # sin과 cos 함수를 사용하여 시간 임베딩을 생성하는 방식은 Transformer 모델에서 사용되는 Positional Encoding과 유사한 방식이라고 한다.
        return emb


class DenoiseModel(nn.Module):
    def __init__(self, x_dim=2, time_dim=32, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(x_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x, t):
        # 2가지 입력
        # x는 현재 시점에서의 데이터 포인트(노이즈가 추가된 상태)이고, 
        # t는 현재 시점이 전체 단계 중에서 어디에 위치하는지를 나타내는 값
        t_emb = self.time_mlp(t)
        h = torch.cat([x, t_emb], dim=-1) 
        # x와 시간 임베딩을 결합하여 모델이 현재 시점에서의 데이터 포인트와 시간 정보를 모두 활용할 수 있도록 합니다.
        #h란 현재 시점에서의 데이터 포인트와 시간 임베딩이 결합된 벡터입니다. 모델은 이 벡터를 입력으로 받아서 노이즈를 제거하는 방향으로 학습됩니다.
        return self.net(h) 
        # 모델의 출력은 현재 시점에서의 데이터 포인트에서 제거해야 할 노이즈를 나타내는 2차원 벡터입니다. 모델은 이 벡터를 학습하여 노이즈를 제거하는 방향으로 업데이트됩니다.


model = DenoiseModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


def q_sample(x0, t_idx, noise=None):
    #진짜 데이터 x0에 시점 t_idx에 해당하는 노이즈를 추가하여 xt를 생성하는 함수
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[t_idx].unsqueeze(1)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return xt, noise
# 노이즈를 왜 반환하는가?
# 모델이 xt에서 제거해야 할 노이즈를 예측하도록 학습되기 때문입니다.
# 모델의 출력은 현재 시점에서의 데이터 포인트에서 제거해야 할 노이즈를 나타내는 벡터입니다. 따라서 모델이 예측해야 하는 타깃이 바로 이 노이즈가 됩니다. 모델이 xt에서 제거해야 할 노이즈를 정확하게 예측할 수 있도록 학습되기 위해서는 실제로 추가된 노이즈를 알고 있어야 합니다. 그래서 q_sample 함수는 xt와 함께 noise도 반환하는 것입니다.


epochs = 5000
batch_size = 256

for step in range(epochs):
    x0 = sample_real_data(batch_size).to(device)
    t_idx = torch.randint(0, T, (batch_size,), device=device)
    t = t_idx.float() / T

    xt, noise = q_sample(x0, t_idx)
    pred_noise = model(xt, t)

    loss = criterion(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")


@torch.no_grad()
def p_sample_loop(n_samples):
    #생성 단계. 노이즈로 시작해서 점진적으로 노이즈를 제거하면서 최종적으로 생성된 데이터 포인트를 얻는 과정입니다.
    x = torch.randn(n_samples, 2).to(device)

    for time_step in reversed(range(T)):
        # 역순으로 진행되는 루프에서 각 시점에 해당하는 노이즈 제거 과정을 수행합니다.
        t = torch.full((n_samples,), time_step / T, device=device)
        beta_t = betas[time_step]
        alpha_t = alphas[time_step]
        alpha_bar_t = alpha_bars[time_step]

        pred_noise = model(x, t)

        if time_step > 0:
            z = torch.randn_like(x) # 마지막 단계에서는 노이즈를 추가하지 않도록 합니다. 왜냐하면 마지막 단계에서는 이미 충분히 노이즈가 제거되어서 최종 생성된 데이터 포인트가 나오기 때문입니다. 따라서 time_step이 0보다 큰 경우에만 새로운 노이즈 z를 생성하여 추가하는 것입니다.
        else:
            z = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * z
        # 이 공식은 모델이 예측한 노이즈(pred_noise)를 사용하여 현재 시점에서의 데이터 포인트 x에서 노이즈를 제거하는 과정을 나타냅니다. 또한, time_step이 0보다 큰 경우에는 새로운 노이즈 z를 추가하여 다음 시점으로 넘어가는 과정을 포함하고 있습니다. 이렇게 함으로써 점진적으로 노이즈가 제거되면서 최종적으로 생성된 데이터 포인트가 나오게 됩니다.
    return x


with torch.no_grad():
    real = sample_real_data(2000).cpu()
    generated = p_sample_loop(2000).cpu()

plt.figure(figsize=(6, 6))
plt.scatter(real[:, 0], real[:, 1], s=10, alpha=0.5, label="Real")
plt.scatter(generated[:, 0], generated[:, 1], s=10, alpha=0.5, label="Generated")
plt.legend()
plt.title("Diffusion: Real vs Generated 2D Samples")
plt.axis("equal")
plt.show()