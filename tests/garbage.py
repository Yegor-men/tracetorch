import torch

n = 5

i = torch.randn(n)
d = torch.rand(n)
t = torch.abs(torch.randn(n))
print(f"i: {i}")
print(f"d: {d}")
print(f"t: {t}")

will_fire = t < i / (1 - d)

print(f"will fire: {will_fire}")

valid_freq = torch.log(d) / torch.log(1 - (t / i) * (1 - d))
invalid_freq = 1 / (torch.exp(i + i * d) * torch.exp(t - i / (1 - d)))

print(f"Valid freq: {valid_freq}")
print(f"Invalid freq: {invalid_freq}")

f = torch.where(will_fire, valid_freq, invalid_freq)

print(f"Frequency: {f}")
