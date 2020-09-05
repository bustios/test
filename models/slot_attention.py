import torch
from torch import nn


class SlotAttention(nn.Module):
  
  def __init__(self, num_slots=4, slot_dim=64, input_dim=64, iters=3, eps=1e-8, hidden_dim=128):
    super().__init__()
    self.num_slots = num_slots
    self.slot_dim = slot_dim
    self.iters = iters
    self.eps = eps
    self.scale = slot_dim ** -0.5

    self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
    self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

    self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
    self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
    self.to_v = nn.Linear(input_dim, slot_dim, bias=False)

    self.gru = nn.GRU(slot_dim, slot_dim)

    hidden_dim = max(slot_dim, hidden_dim)

    self.mlp = nn.Sequential(
      nn.Linear(slot_dim, hidden_dim),
      nn.ReLU(inplace=True),
      nn.Linear(hidden_dim, slot_dim)
    )

    self.norm_input  = nn.LayerNorm(input_dim)
    self.norm_slots  = nn.LayerNorm(slot_dim)
    self.norm_pre_ff = nn.LayerNorm(slot_dim)


  def forward(self, inputs, num_slots=None):
    n = inputs.size(0)
    if num_slots is None:
      num_slots = self.num_slots
    
    mu = self.slots_mu.expand(n, num_slots, -1)
    sigma = self.slots_sigma.expand(n, num_slots, -1)
    slots = torch.normal(mu, sigma)

    inputs = self.norm_input(inputs)        
    k, v = self.to_k(inputs), self.to_v(inputs)

    for _ in range(self.iters):
      slots_prev = slots

      slots = self.norm_slots(slots)
      q = self.to_q(slots)

      dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
      attn = dots.softmax(dim=1) + self.eps
      attn_n = attn / attn.sum(dim=-1, keepdim=True)

      updates = torch.einsum('bjd,bij->bid', v, attn_n)

      slots, _ = self.gru(
        updates.reshape(1, -1, self.slot_dim),
        slots_prev.reshape(1, -1, self.slot_dim)
      )

      slots = slots.reshape(n, -1, self.slot_dim)
      slots = self.norm_pre_ff(slots)
      slots = slots + self.mlp(slots)

    return slots