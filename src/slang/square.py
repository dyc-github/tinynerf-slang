import torch
import slangpy

m = slangpy.loadModule('square.slang')

A = torch.randn((1024,), dtype=torch.float).cuda()

output = torch.zeros_like(A).cuda()

# Number of threads launched = blockSize * gridSize
m.square(input=A, output=output).launchRaw(blockSize=(32, 1, 1), gridSize=(64, 1, 1))

print(output)