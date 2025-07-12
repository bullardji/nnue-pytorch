import torch
from torch import nn

class JepaPredictor(nn.Module):
    """Small MLP used to predict masked tokens."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def jepa_mask(
    x: torch.Tensor,
    mask_ratio: float,
    mode: str = "random",
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply feature masking along the latent dimension."""

    B, D = x.shape
    device = x.device
    if mode == "board_group" and group_size is not None and group_size > 0:
        num_groups = D // group_size
        num_mask = int(num_groups * mask_ratio)
        g_ids = torch.rand(B, num_groups, device=device).argsort(dim=1)[:, :num_mask]
        mask_groups = torch.zeros(B, num_groups, device=device, dtype=torch.bool)
        mask_groups.scatter_(1, g_ids, True)
        group_idx = torch.arange(D, device=device) // group_size
        mask = mask_groups[:, group_idx]
    else:
        num_mask = int(D * mask_ratio)
        mask = torch.zeros(B, D, device=device, dtype=torch.bool)
        if num_mask > 0:
            ids = torch.rand(B, D, device=device).argsort(dim=1)[:, :num_mask]
            mask.scatter_(1, ids, True)

    x_masked = x.clone()
    x_masked.masked_fill_(mask, 0.0)
    return x_masked, mask

def jepa_loss(
    encoder_out: torch.Tensor,
    predictor: JepaPredictor,
    mask_ratio: float = 0.25,
    mask_mode: str = "random",
    group_size: int | None = None,
) -> torch.Tensor:
    """Reconstruction loss for a single view."""

    x_masked, mask = jepa_mask(encoder_out, mask_ratio, mask_mode, group_size)
    pred = predictor(x_masked)
    target = encoder_out.detach()
    loss = (pred[mask] - target[mask]).abs().mean()
    return loss


class JepaWorldModel(nn.Module):
    """JEPA-style module with EMA teacher and optional action conditioning."""

    def __init__(
        self,
        embed_dim: int,
        action_dim: int = 0,
        teacher_momentum: float = 0.996,
        mask_mode: str = "random",
        feature_set=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.teacher_momentum = teacher_momentum
        self.mask_mode = mask_mode
        self.group_size = None
        if mask_mode == "board_group" and feature_set is not None:
            try:
                import importlib

                pt_count = None
                for block in feature_set.features:
                    mod = importlib.import_module(block.__class__.__module__)
                    if hasattr(mod, "NUM_PT_VIRTUAL"):
                        pt_count = mod.NUM_PT_VIRTUAL
                        break
                    if hasattr(mod, "NUM_PT"):
                        pt_count = mod.NUM_PT
                        break
                    if hasattr(mod, "NUM_PT_REAL"):
                        pt_count = mod.NUM_PT_REAL + 1
                        break
                if pt_count is None:
                    pt_count = 12
                self.group_size = embed_dim // (pt_count * 64)
            except Exception:
                self.group_size = embed_dim // (12 * 64)

        # Use lightweight encoders operating on the NNUE latent space. The
        # teacher parameters are an EMA copy of the student's parameters.
        self.student_encoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.teacher_encoder = nn.Linear(embed_dim, embed_dim, bias=False)
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = JepaPredictor(embed_dim + action_dim)

        if action_dim > 0:
            self.action_embed = nn.Embedding(1024, action_dim)
        else:
            self.action_embed = None

    @torch.no_grad()
    def update_teacher(self, student: nn.Module) -> None:
        """Momentum update of the teacher parameters from the student."""

        for p_t, p_s in zip(self.teacher_encoder.parameters(), student.parameters()):
            p_t.data.mul_(self.teacher_momentum).add_(p_s.data * (1.0 - self.teacher_momentum))

    def compute_loss(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        actions: torch.Tensor | None = None,
        mask_ratio: float = 0.25,
        mask_mode: str | None = None,
    ) -> torch.Tensor:
        mode = mask_mode or self.mask_mode
        masked, mask = jepa_mask(student_latent, mask_ratio, mode, self.group_size)
        if actions is not None and self.action_embed is not None:
            act = self.action_embed(actions)
            masked = torch.cat([masked, act], dim=1)
        pred = self.predictor(masked)
        loss = (pred[mask] - teacher_latent.detach()[mask]).abs().mean()
        return loss

