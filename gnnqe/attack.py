"""Attack methods."""
from enum import Enum
import sys
import logging

import torch
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.utils import comm, pretty

module = sys.modules[__name__]
logger = logging.getLogger(__name__)


class AttackMethod(Enum):
    RELATION_EMB_RANDOM = "relation_emb_random"
    RELATION_EMB_NORM2 = "relation_emb_norm2"
    RELATION_EMB_FGA = "relation_emb_fga"  # fast gradient attack


SEED = 233


class AdversarialEngine(core.Engine):
    """Engine specifically supports attack during evaluation."""
    def evaluate(self, split, attack_method=None, attack_scale=None, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not
            attack_method: adversarial attack method
            attack_scale: scale of the perturbation in attack applied

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            if attack_method in (
                    AttackMethod.RELATION_EMB_FGA.value,
                    AttackMethod.RELATION_EMB_NORM2.value,
            ):
                model.zero_grad()
                all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
                print("requires_grad:", all_loss.requires_grad)
                print("all_loss grad_fn:", all_loss.grad_fn)
                metric = {}
                pred, target = model.predict_and_target(batch, all_loss, metric)
                all_loss, metric = model.calculate_loss(pred, target, metric, all_loss)
                print("requires_grad after return:", all_loss.requires_grad)
                print("all_loss grad_fn after return:", all_loss.grad_fn)
                all_loss.backward()

            perturb = self.calculate_attack_perturbation(model.model.model.query, attack_method, attack_scale)
            print("perturb:", perturb)
            model.model.model.query_override = model.model.model.query.weight.detach().clone() + perturb

            with torch.no_grad():
                pred, target = model.predict_and_target(batch)

            preds.append(pred)
            targets.append(target)

            if attack_method in (AttackMethod.RELATION_EMB_RANDOM.value, AttackMethod.RELATION_EMB_FGA.value):
                del model.model.model.query_override

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        return metric

    @staticmethod
    def calculate_attack_perturbation(target, attack_method, attack_scale):
        if attack_method == AttackMethod.RELATION_EMB_RANDOM.value:
            torch.manual_seed(SEED)
            return attack_scale * torch.randn_like(target.weight)

        elif attack_method == AttackMethod.RELATION_EMB_FGA.value:
            grad = target.weight.grad
            return attack_scale * grad.sign()

        elif attack_method == AttackMethod.RELATION_EMB_NORM2.value:
            grad = target.weight.grad
            return attack_scale * grad / (grad.norm(p=2, dim=1, keepdim=True) + 1e-8)

        else:
            raise ValueError(f"Unknown Attack Method {attack_method}")


# class RelationEmbAttacker:
#     def __init__(self, model):
#         self.model = model
#         self.query_embedding = model.query  # nn.Embedding
#         self.seed = 233
#
#     def random_attack(self, scale):
#         """Random perturbation to query embedding."""
#         torch.manual_seed(self.seed)
#         with torch.no_grad():
#             noise = scale * torch.randn_like(self.query_embedding.weight)
#             self.query_embedding.weight.add_(noise)
#
#     def fgsm_attack(self, loss_fn, inputs, targets, scale):
#         """FGSM attack on query embedding."""
#         torch.manual_seed(self.seed)
#         self.model.zero_grad()
#         output = self.model(*inputs)
#         loss = loss_fn(output, targets)
#         loss.backward()
#         with torch.no_grad():
#             grad = self.query_embedding.weight.grad
#             perturb = scale * grad.sign()
#             self.query_embedding.weight.add_(perturb)

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