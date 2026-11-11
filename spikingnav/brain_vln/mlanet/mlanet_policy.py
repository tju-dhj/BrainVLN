"""MLANet PPO policy, matching ``mla_best_ppo.pth``.

The published weights are the EnvDrop + DAgger + PPO model. RGB goes
through CLIP ViT-B/32 (the checkpoint uses the last transformer output).
Depth uses the VLN-CE ResNet-50. Sub-instructions are CLIP text tokens.
As in the official config, CLIP and the depth ResNet backbone are frozen.
"""

from __future__ import annotations

from typing import Dict, Tuple

import clip.model as clip_model
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from spikingnav.brain_vln.mlanet.encoders import VlnResnetDepthEncoder
from spikingnav.brain_vln.mlanet.il_inputs import il_observation_space
from spikingnav.brain_vln.mlanet.policy import ILPolicy


class _GRUStateEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.num_recurrent_layers = 1
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, hidden_states, masks):
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = torch.where(
            masks.view(1, -1, 1), hidden_states, hidden_states.new_zeros(())
        )
        x, hidden_states = self.rnn(x.unsqueeze(0), hidden_states)
        return x.squeeze(0), hidden_states.permute(1, 0, 2)


class _InstructionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(2504, 200, padding_idx=0)
        self.encoder_rnn = nn.LSTM(
            input_size=200, hidden_size=128, bidirectional=True
        )

    def forward(self, observations) -> Tensor:
        instruction = observations["instruction"].long()
        lengths = (instruction != 0).long().sum(dim=1).clamp(min=1)
        embedded = self.embedding_layer(instruction)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.encoder_rnn(packed)
        return pad_packed_sequence(output, batch_first=True)[0].permute(0, 2, 1)


class _CLIPEncoder(nn.Module):
    def __init__(self, clip_state: Dict[str, Tensor]) -> None:
        super().__init__()
        original = clip_model.convert_weights
        clip_model.convert_weights = lambda module: None
        try:
            self.model = clip_model.build_model(dict(clip_state))
        finally:
            clip_model.convert_weights = original
        self.model.float()
        # mla_ppo.yaml: CLIP.trainable False. Only ln_rgb / ln_text train.
        self.model.requires_grad_(False)
        self.ln_rgb = nn.LayerNorm(768)
        self.ln_text = nn.LayerNorm(512)
        self.rgb_downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(3), nn.Flatten(start_dim=2)
        )
        self.normalize_mu = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073])
        self.normalize_sigma = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711])
        self.rgb_embedding_seq = None
        self.text_embedding_seq = None
        # MLANet() never forwards CLIP.rgb_level, so the checkpoint was trained
        # with the constructor default: hook the whole transformer (last block).
        # mla_ppo.yaml's rgb_level: -2 is unused and hooks a different layer.
        self.model.visual.transformer.register_forward_hook(self._vit_hook)
        self.model.transformer.register_forward_hook(self._text_hook)

    def _vit_hook(self, _module, _inputs, output):
        self.rgb_embedding_seq = output.float()

    def _text_hook(self, _module, _inputs, output):
        self.text_embedding_seq = output.float()

    def _normalize(self, imgs: Tensor) -> Tensor:
        if self.normalize_sigma.device != imgs.device:
            self.normalize_sigma = self.normalize_sigma.to(imgs.device)
            self.normalize_mu = self.normalize_mu.to(imgs.device)
        imgs = (imgs / 255.0 - self.normalize_mu) / self.normalize_sigma
        return imgs.permute(0, 3, 1, 2)

    def encode_text(self, observations) -> Tensor:
        sub_instruction = observations["sub_instruction"].int()
        batch, steps, _length = sub_instruction.shape
        text_embedding = sub_instruction.new_zeros((batch, steps, 512), dtype=torch.float)
        for index in range(batch):
            pad = sub_instruction.new_zeros((1,))
            count = int(torch.argmin(torch.cat((sub_instruction[index, :, 0], pad))).item())
            if count == 0:
                continue
            _ = self.model.encode_text(sub_instruction[index, :count])
            sequence = self.ln_text(self.text_embedding_seq.float().permute(1, 0, 2))
            positions = sub_instruction[index, :count].argmax(dim=-1)
            text_embedding[index, :count] = sequence[torch.arange(count, device=sequence.device), positions]
        return text_embedding

    def encode_image(self, observations) -> Tuple[Tensor, Tensor]:
        _ = self.model.encode_image(self._normalize(observations["rgb"])).float()
        sequence = self.ln_rgb(self.rgb_embedding_seq.float().permute(1, 0, 2))
        cls = sequence[:, 0, :]
        patches = sequence[:, 1:, :].permute(0, 2, 1).reshape(sequence.shape[0], sequence.shape[2], 7, 7)
        return cls, self.rgb_downsample(patches)


class MLANet(nn.Module):
    def __init__(self, clip_state: Dict[str, Tensor]) -> None:
        super().__init__()
        hidden = 512
        self.instruction_encoder = _InstructionEncoder()
        self.clip_encoder = _CLIPEncoder(clip_state)
        self.depth_encoder = VlnResnetDepthEncoder(
            il_observation_space(), output_size=3072, spatial_output=True
        )
        self.sub_inst_fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 256), nn.ReLU(inplace=True))
        self.rgb_fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 256), nn.ReLU(inplace=True))
        self.depth_fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.0), nn.Linear(3072, 256), nn.ReLU(inplace=True)
        )
        self.prev_action_embedding = nn.Embedding(5, 32)
        self.low_state_encoder = _GRUStateEncoder(544, hidden)
        self.high_state_encoder = _GRUStateEncoder(544, hidden)
        self.low_level_attention = nn.MultiheadAttention(hidden, 8, batch_first=True, kdim=256, vdim=256)
        self.high_level_attention = nn.MultiheadAttention(hidden, 8, batch_first=True, kdim=256, vdim=256)
        self.inst_post = nn.Linear(hidden * 2, 512)
        self.spatial_attention_rgb = nn.MultiheadAttention(512, 8, batch_first=True, kdim=768, vdim=768)
        self.spatial_attention_depth = nn.MultiheadAttention(512, 8, batch_first=True, kdim=192, vdim=192)
        self.visual_post = nn.Linear(1024, 512)
        self.final_input_compress = nn.Sequential(nn.Linear(2080, hidden), nn.ReLU(inplace=True))
        self.falcon_graph = None
        self.action_decoder = _GRUStateEncoder(hidden, hidden)
        self.progress_monitor = nn.Linear(hidden, 1)
        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden // 2) ** 0.5)))
        self.s1 = 1
        self.s2 = 2
        self._output_size = hidden
        self.num_recurrent_layers = 3

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, observations, rnn_states, prev_actions, masks):
        action_ids = prev_actions.long().view(-1)
        instruction_embedding = self.instruction_encoder(observations)
        sub_embedding = self.clip_encoder.encode_text(observations)
        mask_sub = (sub_embedding == 0).all(dim=2)
        depth_embedding = self.depth_encoder(observations).flatten(2)
        rgb_embedding, rgb_seq = self.clip_encoder.encode_image(observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        sub_embedding = self.sub_inst_fc(sub_embedding)
        instruction_embedding = instruction_embedding.permute(0, 2, 1)
        mask_inst = (instruction_embedding == 0).all(dim=2)
        visual = torch.cat(
            [self.rgb_fc(rgb_embedding), self.depth_fc(depth_embedding)], dim=1
        )
        visual_in = torch.cat([visual, prev_actions], dim=1)
        rnn_out = rnn_states.detach().clone()
        low, rnn_out[:, 0:self.s1] = self.low_state_encoder(
            visual_in, rnn_states[:, 0:self.s1], masks
        )
        high, rnn_out[:, self.s1:self.s2] = self.high_state_encoder(
            visual_in, rnn_states[:, self.s1:self.s2], masks
        )
        batch, inst_len = mask_inst.shape
        inst_mask = mask_inst.unsqueeze(1).repeat(1, 8, 1).reshape(batch * 8, 1, inst_len)
        sub_len = mask_sub.shape[1]
        sub_mask = mask_sub.unsqueeze(1).repeat(1, 8, 1).reshape(batch * 8, 1, sub_len)
        attended_low, _ = self.low_level_attention(
            low.unsqueeze(1), instruction_embedding, instruction_embedding, attn_mask=inst_mask
        )
        attended_high, _ = self.high_level_attention(
            high.unsqueeze(1), sub_embedding, sub_embedding, attn_mask=sub_mask
        )
        language = self.inst_post(torch.cat([attended_high.squeeze(1), attended_low.squeeze(1)], dim=1))
        depth_seq = depth_embedding.permute(0, 2, 1)
        rgb_seq = rgb_seq.permute(0, 2, 1)
        attended_rgb, _ = self.spatial_attention_rgb(language.unsqueeze(1), rgb_seq, rgb_seq)
        attended_depth, _ = self.spatial_attention_depth(language.unsqueeze(1), depth_seq, depth_seq)
        vision = self.visual_post(
            torch.cat([attended_rgb.squeeze(1), attended_depth.squeeze(1)], dim=1)
        )
        features = self.final_input_compress(
            torch.cat([high, low, prev_actions, language, vision], dim=1)
        )
        if self.falcon_graph is not None:
            embedding, self._falcon_next = self.falcon_graph(
                observations["falcon_rgb"],
                observations["falcon_depth"],
                masks,
                observations["falcon_state"],
                action_ids,
            )
            features = self.falcon_graph.fusion(
                features,
                embedding,
                features.new_zeros(features.shape[0], 0),
            )
        decoded, rnn_out[:, self.s2:] = self.action_decoder(
            features, rnn_states[:, self.s2:], masks
        )
        return decoded, rnn_out


class _Critic(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)


class MLAPPOPolicy(ILPolicy):
    def __init__(self, clip_state: Dict[str, Tensor]) -> None:
        net = MLANet(clip_state)
        super().__init__(net, 4)
        self.critic = _Critic(net.output_size)


def build_mlanet(state: Dict[str, Tensor]) -> MLAPPOPolicy:
    stripped = {}
    for key, value in state.items():
        if key.startswith("actor_critic."):
            key = key[len("actor_critic."):]
        stripped[key] = value
    prefix = "net.clip_encoder.model."
    clip_state = {
        key[len(prefix):]: value
        for key, value in stripped.items()
        if key.startswith(prefix)
    }
    policy = MLAPPOPolicy(clip_state)
    if any("falcon_graph." in key for key in stripped):
        raise RuntimeError("BrainVLN loads the MLANet RGB, depth, and language towers, not Falcon-STG")
    missing, unexpected = policy.load_state_dict(stripped, strict=False)
    missing = [key for key in missing if "num_batches_tracked" not in key]
    if missing or unexpected:
        raise RuntimeError(
            f"MLANet checkpoint mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return policy
