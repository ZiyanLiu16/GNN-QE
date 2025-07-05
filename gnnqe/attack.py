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
    RELATION_EMB_PGD = "relation_emb_pgd"


SEED = 233


class AdversarialEngine(core.Engine):
    """Engine specifically supports attack during evaluation."""
    def evaluate(
            self, split, *, attack_method=None, attack_scale=None, pdg_steps=None, pgd_eps=None, log=True, **kwargs
    ):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            attack_method: adversarial attack method
            attack_scale: scale of the perturbation in attack applied
            pdg_steps: number of steps to conduct projection in PGD
            pgd_eps: restrict perturbation value for projected gradient descent
            log (bool, optional): log metrics or not

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

            self.calculate_and_apply_attack_perturbation(attack_method, attack_scale, pdg_steps, pgd_eps, batch)

            with torch.no_grad():
                pred, target = model.predict_and_target(batch)

            preds.append(pred)
            targets.append(target)

            if attack_method in (
                    AttackMethod.RELATION_EMB_RANDOM.value,
                    AttackMethod.RELATION_EMB_FGA.value,
                    AttackMethod.RELATION_EMB_NORM2.value,
                    AttackMethod.RELATION_EMB_PGD.value,
            ):
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

    def calculate_and_apply_attack_perturbation(self, attack_method, attack_scale, pdg_steps, pgd_eps, batch):
        if attack_method in (
                AttackMethod.RELATION_EMB_RANDOM.value,
                AttackMethod.RELATION_EMB_FGA.value,
                AttackMethod.RELATION_EMB_NORM2.value,
        ):
            self._calculate_and_apply_attack_perturbation_non_pgd(attack_method, attack_scale, batch)

        elif attack_method == AttackMethod.RELATION_EMB_PGD.value:
            self._calculate_and_apply_attack_perturbation_pgd(attack_method, attack_scale, pdg_steps, pgd_eps, batch)

    def _calculate_and_apply_attack_perturbation_non_pgd(self, attack_method, attack_scale, batch):
        if attack_method != AttackMethod.RELATION_EMB_RANDOM.value:
            # use self.model.model.model.query in forward, get the gradient
            self.model.zero_grad()
            all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            metric = {}
            pred, target = self.model.predict_and_target(batch, all_loss, metric)
            all_loss, metric = self.model.calculate_loss(pred, target, metric, all_loss)
            all_loss.backward()

        if attack_method == AttackMethod.RELATION_EMB_RANDOM.value:
            torch.manual_seed(SEED)
            perturb = attack_scale * torch.randn_like(self.model.model.model.query.weight)

        elif attack_method == AttackMethod.RELATION_EMB_FGA.value:
            grad = self.model.model.model.query.weight.grad
            perturb = attack_scale * grad.sign()

        elif attack_method == AttackMethod.RELATION_EMB_NORM2.value:
            grad = self.model.model.model.query.weight.grad
            perturb = attack_scale * grad / (grad.norm(p=2, dim=1, keepdim=True) + 1e-8)

        else:
            raise ValueError(
                f"invalid Attack Method under _calculate_and_apply_attack_perturbation_non_pgd: {attack_method}"
            )

        self.model.model.model.query_override = self.model.model.model.query.weight.detach().clone() + perturb

    def _calculate_and_apply_attack_perturbation_pgd(self, attack_method, attack_scale, pdg_steps, pgd_eps, batch):
        original = self.model.model.model.query.weight.detach()

        # random values initial
        perturb = attack_scale * torch.randn_like(original, device=self.device)
        perturb = torch.clamp(perturb, -pgd_eps, pgd_eps)

        for _ in range(pdg_steps):
            # update and use override in forward in each step
            query_override = (original + perturb).requires_grad_()
            self.model.model.model.query_override = query_override

            self.model.zero_grad()
            all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            metric = {}
            pred, target = self.model.predict_and_target(batch, all_loss, metric)
            all_loss, metric = self.model.calculate_loss(pred, target, metric, all_loss)
            all_loss.backward()

            grad = self.model.model.model.query_override.grad

            # update the perturb given the gradient
            perturb = perturb + attack_scale * grad.sign()
            perturb = torch.clamp(perturb, -pgd_eps, pgd_eps)

        self.model.model.model.query_override = (original + perturb).detach()

