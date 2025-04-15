import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph
from torch_scatter import gather_csr

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class QCNeXtDecoder(nn.Module):

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
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 inter_m2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(QCNeXtDecoder, self).__init__()
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
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.inter_m2m_radius = inter_m2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if input_dim != 3 or dataset != 'waymo':
            input_dim_r_t = 4
            input_dim_r_pl2m = 3
            input_dim_r_inter_m2m = 3
        else:
            input_dim_r_t = 5
            input_dim_r_pl2m = 4
            input_dim_r_inter_m2m = 4

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.tgt_agent_emb = nn.Embedding(1, hidden_dim)
        self.ctx_agent_emb = nn.Embedding(1, hidden_dim)
        self.virtual_node_emb = nn.Embedding(1, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_inter_m2m_emb = FourierEmbedding(input_dim=input_dim_r_inter_m2m, hidden_dim=hidden_dim,
                                                num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.inter_m2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.intra_m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                                                           head_dim=head_dim, dropout=dropout, bipartite=False,
                                                           has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.inter_m2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.intra_m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                          dropout=dropout, bipartite=False, has_pos_emb=False)
        self.m2v_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                             dropout=dropout, bipartite=True, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # current step
        # [num_agent, input_dim]
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        # [num_agent]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        # [num_agent, 2]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        # [num_agent * historical_step, hidden_dim]
        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        # [num_modes, num_polygon, hidden_dim]
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        # [num_agent * num_modes, hidden_dim]
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)
        
        if self.ego_conditioned:
            ego_index = data['agent']['av_index']
            # [num_future_step, num_output_dim]
            ego_plan = data['agent']['target'][ego_index, :, :self.output_dim]
            if self.output_head:
                # [num_future_step, num_output_dim + num_head]
                ego_plan = torch.cat([ego_plan, data['agent']['target'][ego_index, :, -1:]], dim=-1)
            # [num_future_step, num_hidden_dim]
            ego_m = self.y_emb(ego_plan.view(-1, ego_plan.size(-1)))
            
            # [num_future_steps, 1, self.hidden_dim]
            ego_m = ego_m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
            
            # [1, self.hidden_dim]
            ego_m = self.traj_emb(ego_m, self.traj_emb_h0.unsqueeze(1).repeat(1, ego_m.size(1), 1))[1].squeeze(0)
            
            # 
            ego_m = gather_csr(src=ego_m, indptr=data['agent']['ptr'])
            
            # [num_modes, self.hidden_dim]
            ego_m = ego_m.repeat_interleave(repeats=self.num_modes, dim=0)
            
            # [num_nodes, num_modes]
            mask_ego = ego_index.new_zeros(data['agent']['num_nodes'], self.num_modes, dtype=torch.bool)
            
            # set ego index as true
            mask_ego[ego_index] = True

        # [num_agent, historical_step]
        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        # ????
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        # [num_agent, num_modes]
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)
        
        if self.dataset == 'argoverse_v2':
            # FOCAL_TRACK， SCORED_TRACK, UNSCORED_TRACK, TRACK_FRAGMENT
            mask_tgt = (data['agent']['category'] == 2) | (data['agent']['category'] == 3)
        elif self.dataset == 'waymo':
            mask_tgt = data['agent']['target_mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        # [num_agent, num_modes]
        mask_tgt = mask_tgt.unsqueeze(-1).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        # index between agent historical step and one mode
        # mask_src: [num_agent, historical_step]
        # mask_dst[:, -1:]: [num_agent, 1]
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        if self.input_dim != 3 or self.dataset != 'waymo':
            # [???, 4]
            r_t2m = torch.stack(
                [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
                 rel_head_t2m,
                 (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        else:
            r_t2m = torch.stack(
                [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
                 rel_pos_t2m[:, -1],
                 rel_head_t2m,
                 (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        # mask_src: [num_agent, historical_step]
        # mask_dst: [num_agent, num_mode]
        # mask_src.unsqueeze(2) & mask_dst.unsqueeze(1): [num_agent, hisitorical_step, num_mode]
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        if self.input_dim != 3 or self.dataset != 'waymo':
            r_pl2m = torch.stack(
                [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
                 rel_orient_pl2m], dim=-1)
        else:
            r_pl2m = torch.stack(
                [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
                 rel_pos_pl2m[:, -1],
                 rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_inter_m2m = radius_graph(
            x=pos_m[:, :2],
            r=self.inter_m2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_inter_m2m = subgraph(subset=mask_dst[:, 0], edge_index=edge_index_inter_m2m)[0]
        rel_pos_inter_m2m = pos_m[edge_index_inter_m2m[0]] - pos_m[edge_index_inter_m2m[1]]
        rel_head_inter_m2m = wrap_angle(head_m[edge_index_inter_m2m[0]] - head_m[edge_index_inter_m2m[1]])
        if self.input_dim != 3 or self.dataset != 'waymo':
            r_inter_m2m = torch.stack(
                [torch.norm(rel_pos_inter_m2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_inter_m2m[1]],
                                          nbr_vector=rel_pos_inter_m2m[:, :2]),
                 rel_head_inter_m2m], dim=-1)
        else:
            r_inter_m2m = torch.stack(
                [torch.norm(rel_pos_inter_m2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_inter_m2m[1]],
                                          nbr_vector=rel_pos_inter_m2m[:, :2]),
                 rel_pos_inter_m2m[:, -1],
                 rel_head_inter_m2m], dim=-1)
        r_inter_m2m = self.r_inter_m2m_emb(continuous_inputs=r_inter_m2m, categorical_embs=None)
        edge_index_inter_m2m = torch.cat(
            [edge_index_inter_m2m + i * edge_index_inter_m2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_inter_m2m = r_inter_m2m.repeat(self.num_modes, 1)

        edge_index_intra_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        m = m + torch.where(mask_tgt.reshape(-1).unsqueeze(-1), self.tgt_agent_emb.weight, self.ctx_agent_emb.weight)
        if self.ego_conditioned:
            m = torch.where(mask_ego.reshape(-1).unsqueeze(-1), ego_m, m)
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                #[num_target_agent, num_mode, num_hidden_dim] = m.reshape(-1, self.hidden_dim)
                # [num_agent * num_mode, hidden_dim]
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                
                # [num_mode * num_agent, hidden_dim]
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                
                # [num_mode * num_agent, hidden_dim]
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                
                # [num_mode * num_agent, hidden_dim]
                m = self.inter_m2m_propose_attn_layers[i](m, r_inter_m2m, edge_index_inter_m2m)
                
                # [num_agent, num_mode, hidden_dim]
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.intra_m2m_propose_attn_layer(m, None, edge_index_intra_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1
        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        m = m + torch.where(mask_tgt.reshape(-1).unsqueeze(-1), self.tgt_agent_emb.weight, self.ctx_agent_emb.weight)
        if self.ego_conditioned:
            m = torch.where(mask_ego.reshape(-1).unsqueeze(-1), ego_m, m)
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.inter_m2m_refine_attn_layers[i](m, r_inter_m2m, edge_index_inter_m2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
        m = self.intra_m2m_refine_attn_layer(m, None, edge_index_intra_m2m)
        # [num_agent, num_modes, num_hidden_dim]
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))

        if isinstance(data, Batch):
            edge_index_m2v = torch.stack([torch.arange(data['agent']['num_nodes'], device=m.device),
                                          data['agent']['batch']], dim=0)
            edge_index_m2v = torch.cat([edge_index_m2v + i * edge_index_m2v.new_tensor(
                [[data['agent']['num_nodes']], [data.num_graphs]]) for i in range(self.num_modes)], dim=1)
        else:
            edge_index_m2v = torch.stack(
                [torch.arange(data['agent']['num_nodes'] * self.num_modes, device=m.device),
                 torch.arange(self.num_modes, device=m.device).repeat_interleave(data['agent']['num_nodes'])], dim=0)
        # [num_mode * num_agent]
        mask_tgt = mask_tgt.reshape(-1, self.num_modes).transpose(0, 1).reshape(-1)
        edge_index_m2v = edge_index_m2v[:, mask_tgt[edge_index_m2v[0]]]
        # [num_mode * num_agent, num_hidden_dim]
        m = m.transpose(0, 1).reshape(-1, self.hidden_dim)
        # [num_mode * num_agent, num_hidden_dim]
        v = self.virtual_node_emb.weight.repeat(self.num_modes * (data['agent']['ptr'].size(0) - 1), 1)
        # [num_mode, num_agent, num_hidden_dim]
        v = self.m2v_attn_layer((m, v), None, edge_index_m2v)
        # [num_target_agent, num_mode, num_hidden_dim]
        v = v.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1)
        # [num_target_agent, num_mode]
        pi = self.to_pi(v).squeeze(-1)

        return {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
        }
