import torch
import math
import torch.nn as nn 

def circular_mse_loss(output, target, period=360):
    diff = (output - target + period / 2) % period - period / 2
    return torch.mean(diff ** 2)

def cosine_similarity_loss(output, target):
    output_rad = output * (2 * math.pi / 360)
    target_rad = target * (2 * math.pi / 360)
    
    output_complex = torch.complex(torch.cos(output_rad), torch.sin(output_rad))
    target_complex = torch.complex(torch.cos(target_rad), torch.sin(target_rad))
    
    return 1 - torch.mean(torch.real(output_complex * torch.conj(target_complex)))

def combined_loss(output, target, alpha=0.5, beta=0.5):
    mse_loss = nn.MSELoss()(output, target)
    circular_loss = circular_mse_loss(output, target)
    return alpha * mse_loss + beta * circular_loss