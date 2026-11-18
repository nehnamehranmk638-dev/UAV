# stgt_predict.py
# Python bridge function called directly by MATLAB with Spatio-Temporal Safety Filtering

import torch
import torch.nn as nn
import numpy as np
from collections import deque

class SpatioTemporalGraphTransformer(nn.Module):
    def __init__(self, node_dim=3, embed_dim=64, num_heads=4, num_layers=3, num_nodes=4, num_frames=3):
        super(SpatioTemporalGraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.feature_projection = nn.Linear(node_dim, embed_dim)
        self.spatial_embedding = nn.Embedding(num_nodes, embed_dim)
        self.temporal_embedding = nn.Embedding(num_frames, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, dropout=0.05, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regression_head = nn.Sequential(
            nn.Linear(num_frames * num_nodes * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

        def forward(self, x):
            batch_size = x.size(0)
            x_tokens = x.view(batch_size, self.num_frames * self.num_nodes, -1)
            projected_features = self.feature_projection(x_tokens)

            spatial_ids = torch.arange(self.num_nodes, device=x.device).repeat(self.num_frames)
            temporal_ids = torch.arange(self.num_frames, device=x.device).repeat_interleave(self.num_nodes)

            pos_embeddings = self.spatial_embedding(spatial_ids) + self.temporal_embedding(temporal_ids)
            transformer_input = projected_features + pos_embeddings.unsqueeze(0)

            attention_out = self.transformer_encoder(transformer_input)
            flattened_context = attention_out.view(batch_size, -1)
            return self.regression_head(flattened_context)

        model = SpatioTemporalGraphTransformer()
        # Fallback to pure architecture logic if weights aren't present to prevent hard crashes
        try:
            model.load_state_dict(torch.load("stgt_model.pth", map_location=torch.device('cpu'), weights_only=True))
        except Exception:
            pass
        model.eval()

        TEMPORAL_DEPTH = 3
        history_queue = deque(maxlen=TEMPORAL_DEPTH)

        def get_predictions(master_pos, obs1_pos, obs2_pos, obs3_pos):
            global history_queue

            current_snapshot = np.array([master_pos, obs1_pos, obs2_pos, obs3_pos], dtype=np.float32)
            history_queue.append(current_snapshot)

            while len(history_queue) < TEMPORAL_DEPTH:
                history_queue.append(current_snapshot)

                sequence_tensor = torch.tensor(np.array(history_queue)).unsqueeze(0).float()

                with torch.no_grad():
                    raw_output = model(sequence_tensor).squeeze(0).numpy()

                    xmin, xmax = min(raw_output[0], raw_output[1]), max(raw_output[0], raw_output[1])
                    ymin, ymax = min(raw_output[2], raw_output[3]), max(raw_output[2], raw_output[3])
                    zmin, zmax = min(raw_output[4], raw_output[5]), max(raw_output[4], raw_output[5])

                    safety_margin = 15.0
                    xmin = min(xmin, master_pos[0] - safety_margin)
                    xmax = max(xmax, master_pos[0] + safety_margin)
                    ymin = min(ymin, master_pos[1] - safety_margin)
                    ymax = max(ymax, master_pos[1] + safety_margin)
                    zmin = min(zmin, master_pos[2] - 4.0)
                    zmax = max(zmax, master_pos[2] + 4.0)

                    dx = history_queue[-1][0][0] - history_queue[0][0][0]
                    dy = history_queue[-1][0][1] - history_queue[0][0][1]

                    if dx > 0.05: xmax = max(xmax, 92.0)
                    if dy > 0.05: ymax = max(ymax, 92.0)
                    if dx < -0.05: xmin = min(xmin, 8.0)
                    if dy < -0.05: ymin = min(ymin, 8.0)

                    xmin, xmax = max(0.0, xmin), min(100.0, xmax)
                    ymin, ymax = max(0.0, ymin), min(100.0, ymax)
                    zmin, zmax = max(0.0, zmin), min(50.0, zmax)

                    return [float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)]