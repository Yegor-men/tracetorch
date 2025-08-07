import torch

n = 5

i = torch.randn(n)
d = torch.rand(n)
t = torch.abs(torch.randn(n))
print(f"i: {i}")
print(f"d: {d}")
print(f"t: {t}")

k = 5.0
excess = i - t * (1 - d)
frequency = torch.nn.functional.sigmoid(k * excess)

print(f"Excess: {excess}")
print(f"Frequency: {frequency}")

should_fire = (t < (i / (1 - d))).float()
print(f"Should fire: {should_fire}")
