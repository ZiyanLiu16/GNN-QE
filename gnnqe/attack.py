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
    SOFT_EDGE_PGD = "soft_edge_pgd"


SEED = 233
torch.manual_seed(SEED)


class AdversarialEngine(core.Engine):
    """Engine specifically supports attack during evaluation."""
    def evaluate(
            self, split,
            *,
            attack_method=None,
            attack_scale=None,
            pgd_steps=None,
            pgd_eps=None,
            edge_ratio=None,
            log=True,
            **kwargs,
    ):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            attack_method: adversarial attack method
            attack_scale: scale of the perturbation in attack applied
            pgd_steps: number of steps to conduct projection in PGD
            pgd_eps: restrict perturbation value for projected gradient descent
            edge_ratio: soft edge perturbation bugdet as ratio (compared to number of original edge)
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

            elif attack_method in (
                    AttackMethod.RELATION_EMB_RANDOM.value,
                    AttackMethod.RELATION_EMB_FGA.value,
                    AttackMethod.RELATION_EMB_NORM2.value,
                    AttackMethod.RELATION_EMB_PGD.value,
                    AttackMethod.SOFT_EDGE_PGD.value,
            ):
                self.calculate_and_apply_attack_perturbation(
                    attack_method, attack_scale, pgd_steps, pgd_eps, edge_ratio, batch
                )

                with torch.no_grad():
                    pred, target = model.predict_and_target(batch)

                preds.append(pred)
                targets.append(target)

                if hasattr(model.model.model, "query_override"):
                    del model.model.model.query_override
                if hasattr(model.graph, "edge_soft_drop"):
                    del model.graph.edge_soft_drop

            else:
                with torch.no_grad():
                    pred, target = model.predict_and_target(batch)

                preds.append(pred)
                targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        return metric

    def calculate_and_apply_attack_perturbation(self, attack_method, attack_scale, pgd_steps, pgd_eps, edge_ratio, batch):
        if attack_method in (
                AttackMethod.RELATION_EMB_RANDOM.value,
                AttackMethod.RELATION_EMB_FGA.value,
                AttackMethod.RELATION_EMB_NORM2.value,
        ):
            self._calculate_and_apply_attack_perturbation_non_pgd(attack_method, attack_scale, batch)

        elif attack_method == AttackMethod.RELATION_EMB_PGD.value:
            self._calculate_and_apply_attack_perturbation_pgd(attack_method, attack_scale, pgd_steps, pgd_eps, batch)

        elif attack_method == AttackMethod.SOFT_EDGE_PGD.value:
            self._calculate_and_apply_attack_perturbation_pgd_soft_edge(attack_scale, pgd_steps, edge_ratio, batch)

        elif attack_method == AttackMethod.SOFT_EDGE_PGD.value:
            self._calculate_and_soft_edge_pgd_perturbation()

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

    def _calculate_and_apply_attack_perturbation_pgd(self, attack_method, attack_scale, pgd_steps, pgd_eps, batch):
        original = self.model.model.model.query.weight.detach()

        # random values initial
        perturb = attack_scale * torch.randn_like(original, device=self.device)
        perturb = torch.clamp(perturb, -pgd_eps, pgd_eps)

        for _ in range(pgd_steps):
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
            # torch.clamp creates a new tensor
            perturb = torch.clamp(perturb, -pgd_eps, pgd_eps)

        self.model.model.model.query_override = (original + perturb).detach()

    def _calculate_and_apply_attack_perturbation_pgd_soft_edge(self, attack_scale, pgd_steps, perturb_ratio, batch):
        num_edge = int(self.model.fact_graph.num_edge)
        edge_soft_drop_raw = torch.rand(1, num_edge, device=self.model.device, requires_grad=True)
        eps = num_edge * perturb_ratio

        for _ in range(pgd_steps):
            print("- " * 30)
            edge_soft_drop_raw = (self._restrict_soft_drop(edge_soft_drop_raw, eps)).squeeze(0)
            edge_soft_drop = edge_soft_drop_raw.detach().requires_grad_()

            # can't assign to graph, will be dropped during `traversal_dropout`.
            for layer in self.model.model.model.model.layers:
                layer.edge_soft_drop = edge_soft_drop

            self.model.zero_grad()
            all_loss = torch.tensor(0.0, device=self.model.device)
            metric = {}

            pred, target = self.model.predict_and_target(batch, all_loss, metric)
            all_loss, metric = self.model.calculate_loss(pred, target, metric, all_loss)
            all_loss.backward()

            grad = edge_soft_drop.grad
            print(f"edge_soft_drop: {edge_soft_drop[:6]}")
            print(f"grad: {grad[:6]}")
            with torch.no_grad():
                edge_soft_drop_raw = (edge_soft_drop + attack_scale * grad.sign()).detach().requires_grad_()

        with torch.no_grad():
            edge_soft_drop = self._restrict_soft_drop(edge_soft_drop_raw, eps).squeeze(0)
            edge_soft_drop = torch.bernoulli(edge_soft_drop).detach()
            print(f"(1 - edge_soft_drop).sum(): {(1 - edge_soft_drop).sum()}")
            for layer in self.model.model.model.model.layers:
                layer.edge_soft_drop = edge_soft_drop

        print("- " * 30 + "\n" + "- " * 30 + "- " * 30)
        return

    @staticmethod
    def _restrict_soft_drop(a, eps, xi=1e-5, ub=1):
        pa = torch.clamp(a, 0, ub)
        if pa.sum() <= eps:
            return pa
        else:
            mu_l = (a - 1).min()
            mu_u = a.max()

            while torch.abs(mu_u - mu_l) > xi:
                mu_a = (mu_u + mu_l) / 2
                gu = (torch.clamp(a - mu_a, 0, ub)).sum() - eps
                gu_l = (torch.clamp(a - mu_l, 0, ub)).sum() - eps
                if gu == 0:
                    break
                if torch.sign(gu) == torch.sign(gu_l):
                    mu_l = mu_a
                else:
                    mu_u = mu_a

            return torch.clamp(a - mu_a, 0, ub)


# class SoftEdgePerturb:
#     def __init__(self, steps: int, perturb_ratio: float):
#         self.steps = steps
#         self.perturb_ratio = perturb_ratio
#         # track the edge to relation mapping; during perturbation, new edges can be created
#         self.num_relations = None
#         self.edge_relation_map = None
#
#     def perturb_whole_graph(self, model):
#         self.edge_relation_map = self._build_edge_relation_map(model.graph)
#         self.num_relations = int(model.graph.num_relation)
#         adjacency_perturbed = self.init_perturbed_adjacency(model.graph.adjacency)
#         # TODO: replace all graph's components
#
#     def init_perturbed_adjacency(self, adjacency):
#         """initiate perturbed Adjacency matrix from the original."""
#         adjacency_dense = adjacency.coalesce()
#         adjacency_dense_perturbed = self._init_perturbed_adjacency_dense(adjacency_dense)
#         adjacency_perturbed = self._build_sparse_adjacency(adjacency_dense_perturbed)
#         return adjacency_perturbed
#
#     def _init_perturbed_adjacency_dense(self, adjacency_dense):
#         N, _, R = adjacency_dense.shape  # [N, N, R]
#         v_i, v_j, e_r = adjacency_dense.indices()
#
#         A = torch.zeros((N, N), device=adjacency_dense.device)
#         A[v_i, v_j] = 1
#         A_hat = 1 - 2 * A
#
#         # number of edges * perturb ratio
#         eps = len(adjacency_dense.values()) * self.perturb_ratio
#         S = self._build_symmetric_soft_edges(N, eps, adjacency_dense.device)
#
#         A_perturbed = A + A_hat * S
#         A_perturbed = A_perturbed.clamp(min=0, max=1)
#         # be sure diagonal is zero. no self edge
#         A_perturbed = A_perturbed.fill_diagonal_(0)
#
#         return A_perturbed
#
#     # TODO: make sure sampling in last step also conforms symmetric and no self-edge.
#     def _build_symmetric_soft_edges(self, N, eps, device):
#         S_raw = torch.rand((N, N), device=device)
#         S_raw = torch.nn.Parameter(S_raw)
#
#         # give same prob to same relation different directions (e.g. born_in & born_in^-1)
#         upper_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
#         S_upper = S_raw * upper_mask
#
#         S_upper = self._restrict_perturbation(a=S_upper, eps=eps)
#
#         S = S_upper + S_upper.T
#         S.fill_diagonal_(0)
#         return S
#
#     @staticmethod
#     def _restrict_perturbation(a, eps, xi=1e-5, ub=1):
#         """Restrict perturbation for S_upper by eps.
#
#         a: S_upper to be restricted
#         """
#         pa = torch.clamp(a, 0, ub)
#         if pa.sum().item() <= eps:
#             upper_S_update = pa
#         else:
#             mu_l = (a - 1).min()
#             mu_u = a.max()
#
#             while torch.abs(mu_u - mu_l) > xi:
#                 mu_a = (mu_u + mu_l) / 2
#                 gu = (torch.clamp(a - mu_a, 0, ub)).sum() - eps
#                 gu_l = (torch.clamp(a - mu_l, 0, ub)).sum() - eps
#                 if gu == 0:
#                     break
#                 if torch.sign(gu) == torch.sign(gu_l):
#                     mu_l = mu_a
#                 else:
#                     mu_u = mu_a
#
#             upper_S_update = torch.clamp(a - mu_a, 0, ub)
#
#         return upper_S_update
#
#     def _build_sparse_adjacency(self, A_perturbed):
#         # non-zero (perturbed) edges
#         nonzero_indices = (A_perturbed > 0).nonzero(as_tuple=False)  # shape: [E, 2]
#         i_idx = nonzero_indices[:, 0]
#         j_idx = nonzero_indices[:, 1]
#
#         relation_idx = self.edge_relation_map[i_idx, j_idx].unsqueeze(0)  # shape: [1, num_edges]
#
#         mask = (relation_idx == -1)
#         # consider just assign 5.
#         relation_new_assign = torch.randint(0, self.num_relations, size=(mask.sum(),))
#         relation_idx[mask] = relation_new_assign
#
#         indices_perturbed = torch.cat([nonzero_indices.T, relation_idx], dim=0)
#         values = A_perturbed[i_idx, j_idx]
#         A_sparse_perturbed = torch.sparse_coo_tensor(
#             indices=indices_perturbed,
#             values=values,
#             size=(*tuple(A_perturbed.shape), self.num_relations)
#         )
#         return A_sparse_perturbed
#
#     @staticmethod
#     def _build_edge_relation_map(graph):
#         """
#         Constructs a (num_nodes x num_nodes) matrix where each entry [i, j]
#         stores the relation id r for edge (i, j, r).
#
#         If multiple relations exist between (i, j), this will overwrite (use last seen).
#         """
#         num_nodes = int(graph.num_node)
#         device = graph.edge_list.device
#
#         i_idx, j_idx, r_idx = graph.edge_list.t()  # shape [num_edges, 3]
#
#         # Initialize mapping with -1 to indicate "no relation"
#         edge_relation_map = torch.full((num_nodes, num_nodes), -1, dtype=torch.int64, device=device)
#
#         # Assign relation ids
#         edge_relation_map[i_idx, j_idx] = r_idx
#
#         return edge_relation_map



