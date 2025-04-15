import os
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_scatter import gather_csr
from torch_scatter import segment_csr

from losses import FocalLoss
from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import JointBrier
from metrics import JointCR
from metrics import JointMR
from metrics import minJointADE
from metrics import minJointAHE
from metrics import minJointFDE
from metrics import minJointFHE
from metrics.utils import is_match
from modules import QCNeXtDecoder
from modules import QCNetEncoder
from utils import generate_waymo_motion_prediction_submission
from utils import nms
from utils import unbatch
from utils import wrap_angle

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object


class QCNeXt(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 ego_conditioned: bool,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 inter_m2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 **kwargs) -> None:
        super(QCNeXt, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.ego_conditioned = ego_conditioned
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.inter_m2m_radius = inter_m2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        self.encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.decoder = QCNeXtDecoder(
            dataset=dataset,
            ego_conditioned=ego_conditioned,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            inter_m2m_radius=inter_m2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        if dataset == 'waymo':
            self.cls_loss = FocalLoss(alpha=-1, gamma=2.0, reduction='none')
        else:
            self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim +
                                                                  ['von_mises'] * output_head,
                                           reduction='none')

        self.Brier = JointBrier(max_guesses=6)
        self.minADE = minJointADE(max_guesses=6)
        self.minAHE = minJointAHE(max_guesses=6)
        self.minFDE = minJointFDE(max_guesses=6)
        self.minFHE = minJointFHE(max_guesses=6)
        self.MR = JointMR(max_guesses=6)
        self.CR = JointCR(max_guesses=6)

        self.test_predictions = dict()

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        else:
            data['agent']['ptr'] = torch.tensor([0, data['agent']['num_nodes']], device=self.device)
            data['map_polygon']['ptr'] = torch.tensor([0, data['map_polygon']['num_nodes']], device=self.device)
            data['map_point']['ptr'] = torch.tensor([0, data['map_point']['num_nodes']], device=self.device)
        ptr = data['agent']['ptr']
        if self.dataset == 'argoverse_v2':
            eval_mask = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        elif self.dataset == 'waymo':
            eval_mask = data['agent']['target_mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        if self.ego_conditioned:
            eval_mask[data['agent']['av_index']] = False
        if self.dataset == 'waymo':
            data['agent']['predict_mask'][~(data['agent']['target_mask'] |
                                            data['agent']['interact_mask']
                                            ), self.num_historical_steps:] = False
            use_interact = gather_csr(src=(segment_csr(src=data['agent']['interact_mask'].long(),
                                                       indptr=ptr,
                                                       reduce='sum') == 2).long(),
                                      indptr=ptr).bool()
            data['agent']['predict_mask'][use_interact &
                                          (~data['agent']['interact_mask']), self.num_historical_steps:] = False
        if self.ego_conditioned:
            data['agent']['predict_mask'][data['agent']['av_index'], self.num_historical_steps:] = False
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        if self.dataset == 'waymo':
            cls_mask = (segment_csr(src=(data['agent']['predict_mask'][:, [40, 60, 90]].all(dim=-1) & eval_mask).long(),
                                    indptr=ptr,
                                    reduce='sum') ==
                        segment_csr(src=eval_mask.long(),
                                    indptr=ptr,
                                    reduce='sum'))
        else:
            cls_mask = segment_csr(src=(data['agent']['predict_mask'][:, -1] & eval_mask).long(), indptr=ptr,
                                   reduce='max')
        pred = self(data)
        if self.output_head:
            # [num_agent, num_modes, num_future_steps, output_dim + 1 + ouput_dim + 1]
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        # - :   [num_agent, num_modes, num_future_steps, output_dim]
        # norm: [num_agent, num_modes, num_future_steps]
        # sum : [num_agent, num_modes]
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        l2_norm[~eval_mask] = 0
        best_mode_batch = segment_csr(src=l2_norm, indptr=ptr, reduce='sum').argmin(dim=-1)
        best_mode = gather_csr(src=best_mode_batch, indptr=ptr)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        if self.dataset == 'waymo':
            cls_target = torch.zeros_like(pi)
            cls_target[torch.arange(cls_target.size(0)), best_mode_batch] = 1.0
            ignore = is_match(pred=traj_refine[..., :2],
                              target=gt[..., :2],
                              head=(data['agent']['heading'][:, self.num_historical_steps:] -
                                    data['agent']['heading'][:, self.num_historical_steps - 1].unsqueeze(-1)),
                              vel=data['agent']['velocity'][:, self.num_historical_steps - 1, :2],
                              mask=eval_mask,
                              ptr=ptr,
                              joint=True)
            ignore[torch.arange(ignore.size(0)), best_mode_batch] = 0.0
            cls_mask = cls_mask.unsqueeze(-1) & ~ignore.bool()
            cls_loss = self.cls_loss(pi, cls_target) * cls_mask
            cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
            cls_loss *= 10
        else:
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                     target=gt[:, -1:, :self.output_dim + self.output_head],
                                     prob=pi,
                                     mask=(reg_mask[:, -1] & eval_mask).unsqueeze(-1),
                                     ptr=ptr,
                                     joint=True) * cls_mask
            cls_loss = cls_loss.sum() / (reg_mask[:, -1] & eval_mask).sum().clamp_(min=1)
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        loss = reg_loss_propose + reg_loss_refine + cls_loss
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        else:
            data['agent']['ptr'] = torch.tensor([0, data['agent']['num_nodes']], device=self.device)
            data['map_polygon']['ptr'] = torch.tensor([0, data['map_polygon']['num_nodes']], device=self.device)
            data['map_point']['ptr'] = torch.tensor([0, data['map_point']['num_nodes']], device=self.device)
        ptr = data['agent']['ptr']
        if self.dataset == 'argoverse_v2':
            eval_mask = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        elif self.dataset == 'waymo':
            eval_mask = data['agent']['target_mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        if self.ego_conditioned:
            eval_mask[data['agent']['av_index']] = False
        if self.dataset == 'waymo':
            data['agent']['predict_mask'][~(data['agent']['target_mask'] |
                                            data['agent']['interact_mask']
                                            ), self.num_historical_steps:] = False
            use_interact = gather_csr(src=(segment_csr(src=data['agent']['interact_mask'].long(),
                                                       indptr=ptr,
                                                       reduce='sum') == 2).long(),
                                      indptr=ptr).bool()
            data['agent']['predict_mask'][use_interact &
                                          (~data['agent']['interact_mask']), self.num_historical_steps:] = False
        if self.ego_conditioned:
            data['agent']['predict_mask'][data['agent']['av_index'], self.num_historical_steps:] = False
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]
        if self.dataset == 'waymo':
            cls_mask = (segment_csr(src=(data['agent']['predict_mask'][:, [40, 60, 90]].all(dim=-1) & eval_mask).long(),
                                    indptr=ptr,
                                    reduce='sum') ==
                        segment_csr(src=eval_mask.long(),
                                    indptr=ptr,
                                    reduce='sum'))
        else:
            cls_mask = segment_csr(src=(data['agent']['predict_mask'][:, -1] & eval_mask).long(), indptr=ptr,
                                   reduce='max')
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        l2_norm[~eval_mask] = 0
        best_mode_batch = segment_csr(src=l2_norm, indptr=ptr, reduce='sum').argmin(dim=-1)
        best_mode = gather_csr(src=best_mode_batch, indptr=ptr)
        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        if self.dataset == 'waymo':
            cls_target = torch.zeros_like(pi)
            cls_target[torch.arange(cls_target.size(0)), best_mode_batch] = 1.0
            ignore = is_match(pred=traj_refine[..., :2],
                              target=gt[..., :2],
                              head=(data['agent']['heading'][:, self.num_historical_steps:] -
                                    data['agent']['heading'][:, self.num_historical_steps - 1].unsqueeze(-1)),
                              vel=data['agent']['velocity'][:, self.num_historical_steps - 1, :2],
                              mask=eval_mask,
                              ptr=ptr,
                              joint=True)
            ignore[torch.arange(ignore.size(0)), best_mode_batch] = 0.0
            cls_mask = cls_mask.unsqueeze(-1) & ~ignore.bool()
            cls_loss = self.cls_loss(pi, cls_target) * cls_mask
            cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
            cls_loss *= 10
        else:
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                     target=gt[:, -1:, :self.output_dim + self.output_head],
                                     prob=pi,
                                     mask=(reg_mask[:, -1] & eval_mask).unsqueeze(-1),
                                     ptr=ptr,
                                     joint=True) * cls_mask
            cls_loss = cls_loss.sum() / (reg_mask[:, -1] & eval_mask).sum().clamp_(min=1)
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        num_eval_nodes_batch = segment_csr(src=eval_mask.long(), indptr=ptr, reduce='sum')
        ptr_eval = num_eval_nodes_batch.new_zeros((num_eval_nodes_batch.size(0) + 1,))
        torch.cumsum(num_eval_nodes_batch, dim=0, out=ptr_eval[1:])
        valid_mask_eval = reg_mask[eval_mask]
        traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
        if not self.output_head:
            traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                     traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
            traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
        if self.dataset == 'waymo':
            pi_eval = gather_csr(src=torch.sigmoid(pi), indptr=ptr_eval)
        else:
            pi_eval = gather_csr(src=F.softmax(pi, dim=-1), indptr=ptr_eval)
        gt_eval = gt[eval_mask]

        self.Brier.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                          valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.minADE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.minAHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.minFDE.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                           valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.minFHE.update(pred=traj_eval, target=gt_eval, prob=pi_eval, valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.MR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval, ptr=ptr_eval)
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval[..., :2] = torch.matmul(traj_eval[..., :2],
                                          rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        gt_eval[..., :2] = torch.bmm(gt_eval[..., :2], rot_mat) + origin_eval[:, :2].unsqueeze(1)
        if self.output_dim == 3:
            traj_eval[..., 2] = traj_eval[..., 2] + origin_eval[:, 2].reshape(-1, 1, 1)
            gt_eval[..., 2] = gt_eval[..., 2] + origin_eval[:, 2].unsqueeze(-1)
        traj_eval[..., -1] = wrap_angle(traj_eval[..., -1] + theta_eval.reshape(-1, 1, 1))
        gt_eval[..., -1] = wrap_angle(gt_eval[..., -1] + theta_eval.unsqueeze(-1))
        self.CR.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                       valid_mask=valid_mask_eval, ptr=ptr_eval)
        self.log('val_Brier', self.Brier, prog_bar=True, on_step=False, on_epoch=True, batch_size=ptr.size(0) - 1)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=ptr.size(0) - 1)
        self.log('val_minAHE', self.minAHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=ptr.size(0) - 1)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=ptr.size(0) - 1)
        self.log('val_minFHE', self.minFHE, prog_bar=True, on_step=False, on_epoch=True, batch_size=ptr.size(0) - 1)
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_CR', self.CR, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        else:
            data['agent']['ptr'] = torch.tensor([0, data['agent']['num_nodes']], device=self.device)
            data['map_polygon']['ptr'] = torch.tensor([0, data['map_polygon']['num_nodes']], device=self.device)
            data['map_point']['ptr'] = torch.tensor([0, data['map_point']['num_nodes']], device=self.device)
        ptr = data['agent']['ptr']
        if self.dataset == 'argoverse_v2':
            eval_mask = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        elif self.dataset == 'waymo':
            eval_mask = data['agent']['target_mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        if self.dataset == 'waymo':
            data['agent']['predict_mask'][~eval_mask, self.num_historical_steps:] = False
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        num_eval_nodes_batch = segment_csr(src=eval_mask.long(), indptr=ptr, reduce='sum')
        ptr_eval = num_eval_nodes_batch.new_zeros((num_eval_nodes_batch.size(0) + 1,))
        torch.cumsum(num_eval_nodes_batch, dim=0, out=ptr_eval[1:])
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.zeros(eval_mask.sum(), self.num_modes, self.num_future_steps, 4, device=self.device)
        traj_eval[..., :2] = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                          rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        if self.output_dim == 3:
            traj_eval[..., 2] = traj_refine[eval_mask, :, :, 2] + origin_eval[:, 2].reshape(-1, 1, 1)
        if self.output_head:
            traj_eval[..., 3] = wrap_angle(traj_refine[eval_mask, :, :, -1] + theta_eval.reshape(-1, 1, 1))
        else:
            traj_2d_with_start_pos_eval = torch.cat(
                [origin_eval[:, :2].reshape(-1, 1, 1, 2).repeat(1, self.num_modes, 1, 1), traj_eval[..., :2]], dim=-2)
            motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
            traj_eval[..., 3] = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
        if self.dataset == 'waymo':
            pi_eval = torch.sigmoid(pi)
        else:
            pi_eval = F.softmax(pi, dim=-1)

        if self.dataset == 'argoverse_v2':
            pi_eval = pi_eval.cpu().numpy()
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                traj_eval_unbatch = unbatch(src=traj_eval, batch=data['agent']['batch'][eval_mask], dim=0)
                for i in range(data.num_graphs):
                    track_predictions = dict()
                    for j in range(traj_eval_unbatch[i].size(0)):
                        track_predictions[eval_id[ptr_eval[i] + j]] = traj_eval_unbatch[i][j][..., :2].cpu().numpy()
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], track_predictions)
            else:
                traj_eval = traj_eval.cpu().numpy()
                track_predictions = dict()
                for i in range(len(eval_id)):
                    track_predictions[eval_id[i]] = traj_eval[i][..., :2]
                self.test_predictions[data['scenario_id']] = (pi_eval[0], track_predictions)
        elif self.dataset == 'waymo':
            if isinstance(data, Batch):
                eval_id_unbatch = unbatch(src=data['agent']['id'][eval_mask], batch=data['agent']['batch'][eval_mask],
                                          dim=0)
                traj_eval_unbatch = unbatch(src=traj_eval, batch=data['agent']['batch'][eval_mask], dim=0)
                vel_eval_unbatch = unbatch(src=data['agent']['velocity'][eval_mask],
                                           batch=data['agent']['batch'][eval_mask], dim=0)
                for i in range(data.num_graphs):
                    track_predictions = dict()
                    filtered_mask = pi_eval[i] > 0.005
                    filtered_traj = traj_eval_unbatch[i][:, filtered_mask, 4::5, :2]
                    filtered_pi = pi_eval[i][filtered_mask]
                    nms(traj=filtered_traj, pi=filtered_pi, nms_thres=None,
                        vel=vel_eval_unbatch[i][:, self.num_historical_steps - 1])
                    for j in range(len(eval_id_unbatch[i])):
                        track_predictions[eval_id_unbatch[i][j].item()] = filtered_traj[j].cpu().numpy()
                    self.test_predictions[data['scenario_id'][i]] = (track_predictions, filtered_pi.cpu().numpy())
            else:
                eval_id = data['agent']['id'][eval_mask]
                vel_eval = data['agent']['velocity'][eval_mask]
                track_predictions = dict()
                filtered_mask = pi_eval[0] > 0.005
                filtered_traj = traj_eval[:, filtered_mask, 4::5, :2]
                filtered_pi = pi_eval[0][filtered_mask]
                nms(traj=filtered_traj, pi=filtered_pi, nms_thres=None, vel=vel_eval[:, self.num_historical_steps - 1])
                for i in range(len(eval_id)):
                    track_predictions[eval_id[i].item()] = filtered_traj[i].cpu().numpy()
                self.test_predictions[data['scenario_id']] = (track_predictions, filtered_pi.cpu().numpy())
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def on_test_end(self):
        if self.dataset == 'argoverse_v2':
            ChallengeSubmission(self.test_predictions).to_parquet(
                Path(self.submission_dir) / f'{self.submission_file_name}.parquet')
        elif self.dataset == 'waymo':
            generate_waymo_motion_prediction_submission(
                predictions=self.test_predictions,
                account_name='nobody@gmail.com',
                method_name='QCNeXt',
                authors=['Nobody'],
                affiliation='Tesla',
                description='',
                method_link='',
                uses_lidar_data=False,
                uses_camera_data=False,
                uses_public_model_pretraining=False,
                public_model_names=[],
                num_model_parameters='8M',
                path=os.path.join(self.submission_dir, f'{self.submission_file_name}.bin'),
                joint=True)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNeXt')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--ego_conditioned', action='store_true')
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--inter_m2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--T_max', type=int, default=50)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        return parent_parser
