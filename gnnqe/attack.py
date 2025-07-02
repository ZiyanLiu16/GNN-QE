"""Attack methods."""
import torch


class RelationEmbAttacker:
    def __init__(self, model):
        self.model = model
        self.query_embedding = model.query  # nn.Embedding
        self.seed = 233

    def random_attack(self, scale):
        """Random perturbation to query embedding."""
        torch.manual_seed(self.seed)
        with torch.no_grad():
            noise = scale * torch.randn_like(self.query_embedding.weight)
            self.query_embedding.weight.add_(noise)

    def fgsm_attack(self, loss_fn, inputs, targets, scale):
        """FGSM attack on query embedding."""
        torch.manual_seed(self.seed)
        self.model.zero_grad()
        output = self.model(*inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        with torch.no_grad():
            grad = self.query_embedding.weight.grad
            perturb = scale * grad.sign()
            self.query_embedding.weight.add_(perturb)

    # def pgd_attack(self, loss_fn, inputs, targets, alpha=1e-3, steps=3):
    #     """PGD attack on query embedding."""
    #     original = self.query_embedding.weight.detach().clone()
    #     perturb = torch.zeros_like(original)
    #
    #     for _ in range(steps):
    #         self.query_embedding.weight.data = (original + perturb).detach().clone().requires_grad_()
    #         self.model.zero_grad()
    #         output = self.model(*inputs)
    #         loss = loss_fn(output, targets)
    #         loss.backward()
    #         grad = self.query_embedding.weight.grad
    #         perturb = perturb + alpha * grad.sign()
    #         perturb = torch.clamp(perturb, -self.epsilon, self.epsilon)
    #
    #     with torch.no_grad():
    #         self.query_embedding.weight.data = original + perturb