import torch

n = 5

i = torch.randn(n)
d = torch.rand(n)
t = torch.abs(torch.randn(n))
print(f"i: {i}")
print(f"d: {d}")
print(f"t: {t}")

# if t<1, frequency is 1
# if t>i/(1-d), frequency is 0
# we construct a function with the idea of plugging in the threshold returns the frequency

low = i
high = i / (1 - d)
mid = (low + high) / 2
dilation = torch.abs(high - mid)
# dilation is how much to stretch the graph

p_tail = 0.05
# we need the ptail to calculate the squeeze of the sigmoid so that s(-1)=ptail (otherwise it's too wide)
# after rearranging sigmoid function we can find the squeeze factor to be ln(1/p-1)
squeeze = torch.log(torch.tensor([1 / p_tail - 1]))

raw_sigmoid = -(squeeze / dilation) * (t - mid)
print(f"raw sigmo: {raw_sigmoid}")
frequency = torch.nn.functional.sigmoid(raw_sigmoid)
print(f"Frequency: {frequency}")
should_fire = (t < (i / (1 - d))).float()
print(f"Should fire: {should_fire}")
