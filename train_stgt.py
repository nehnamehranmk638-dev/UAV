# train_stgt.py
# Authentic Spatio-Temporal Graph Transformer (STGT) Training Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for absolute reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("[STGT CORE] Initializing Spatio-Temporal Graph Transformer Training...")
print("=" * 70)

# 1. AUTHENTIC MULTI-HEAD SPATIO-TEMPORAL GRAPH TRANSFORMER ARCHITECTURE
class SpatioTemporalGraphTransformer(nn.Module):
    def __init__(self, node_dim=3, embed_dim=64, num_heads=4, num_layers=3, num_nodes=4, num_frames=3):
        super(SpatioTemporalGraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # Spatial Node Feature Projection Layer
        self.feature_projection = nn.Linear(node_dim, embed_dim)
        
        # Trainable Spatio-Temporal Positional Embeddings
        self.spatial_embedding = nn.Embedding(num_nodes, embed_dim)
        self.temporal_embedding = nn.Embedding(num_frames, embed_dim)
        
        # The Core Transformer Encoder (Multi-Head Attention Engine)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=128,
            dropout=0.05,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # High-Fidelity Regression Head for Bounding Box Bounds Optimization
        self.regression_head = nn.Sequential(
            nn.Linear(num_frames * num_nodes * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Outputs optimized: [xmin, xmax, ymin, ymax, zmin, zmax]
        )
        
    def forward(self, x):
        # Input shape expected: (Batch, Frames, Nodes, Coordinates) -> (B, 3, 4, 3)
        batch_size = x.size(0)
        
        # Reshape to token sequences for attention mechanism
        # Flattening Frames and Nodes dimensions into a unified token sequence
        x_tokens = x.view(batch_size, self.num_frames * self.num_nodes, -1)
        projected_features = self.feature_projection(x_tokens)
        
        # Compute and inject Spatial and Temporal coordinate identities
        spatial_ids = torch.arange(self.num_nodes, device=x.device).repeat(self.num_frames)
        temporal_ids = torch.arange(self.num_frames, device=x.device).repeat_interleave(self.num_nodes)
        
        pos_embeddings = self.spatial_embedding(spatial_ids) + self.temporal_embedding(temporal_ids)
        transformer_input = projected_features + pos_embeddings.unsqueeze(0)
        
        # Execute Self-Attention mapping across all space-time coordinates
        attention_out = self.transformer_encoder(transformer_input)
        
        # Flatten the contextualized token sequence for regression parsing
        flattened_context = attention_out.view(batch_size, -1)
        return self.regression_head(flattened_context)

# 2. GENERATE HIGH-FIDELITY TRACKING DATASET FOR biRRT OPTIMIZATION
def generate_synthetic_stgt_data(num_samples=2000):
    X_data = []
    Y_data = []
    
    # Define typical mission flight spaces and patrol goals
    goals = [np.array([90.0, 90.0, 20.0]), np.array([10.0, 10.0, 20.0])]
    
    for idx in range(num_samples):
        # Alternating target checkpoints to mimic the active patrol leg
        current_goal = goals[idx % 2]
        
        # Define basic spatial seeds for the master drone and 3 obstacles
        master_start = np.random.uniform(15, 85, size=(3,))
        obs1_start = np.random.uniform(20, 80, size=(3,))
        obs2_start = np.random.uniform(20, 80, size=(3,))
        obs3_start = np.random.uniform(20, 80, size=(3,))
        
        # Simulate trajectories across 3 frames using constant localized velocities
        v_master = (current_goal - master_start) / np.linalg.norm(current_goal - master_start) * np.random.uniform(2, 4)
        v_obs1 = np.random.uniform(-2, 2, size=(3,))
        v_obs2 = np.random.uniform(-2, 2, size=(3,))
        v_obs3 = np.random.uniform(-2, 2, size=(3,))
        
        sequence = []
        for t in range(3): # 3 distinct time steps
            m_t = master_start + v_master * t
            o1_t = obs1_start + v_obs1 * t
            o2_t = obs2_start + v_obs2 * t
            o3_t = obs3_start + v_obs3 * t
            sequence.append([m_t, o1_t, o2_t, o3_t])
            
        sequence = np.array(sequence) # Shape: (3, 4, 3)
        X_data.append(sequence)
        
        # Compute the mathematically optimized search space envelope for biRRT
        latest_master = sequence[-1, 0]
        latest_obstacles = sequence[-1, 1:]
        
        # Find the distance to the closest threatening obstacle
        min_threat_dist = min([np.linalg.norm(latest_master - obs) for obs in latest_obstacles])
        
        # Dynamic search space compression: squeeze box if threat is high, expand to run fast if clear
        margin = 15.0 if min_threat_dist < 25.0 else 30.0
        
        # Formulate optimized boundaries containing both the drone and a corridor toward the goal
        xmin = max(0.0, min(latest_master[0], current_goal[0]) - margin)
        xmax = min(100.0, max(latest_master[0], current_goal[0]) + margin)
        ymin = max(0.0, min(latest_master[1], current_goal[1]) - margin)
        ymax = min(100.0, max(latest_master[1], current_goal[1]) + margin)
        zmin = max(0.0, min(latest_master[2], current_goal[2]) - 8.0)
        zmax = min(50.0, max(latest_master[2], current_goal[2]) + 8.0)
        
        Y_data.append([xmin, xmax, ymin, ymax, zmin, zmax])
        
    return torch.tensor(np.array(X_data), dtype=torch.float32), torch.tensor(np.array(Y_data), dtype=torch.float32)

# 3. TRAINING ROUTINE EXECUTION
X, Y = generate_synthetic_stgt_data()
print(f"[DATASET] Synthesized {X.shape[0]} tracking sequences. Tensor Map Shape: {X.shape}")

model = SpatioTemporalGraphTransformer()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

epochs = 100
loss_history = []

print("\n[TRAINING] Commencing model optimization optimization across 100 epochs...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X)
    loss = criterion(predictions, Y)
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch [{epoch+1:3d}/{epochs}] ---> Graph-Attention Convergence Loss: {loss.item():.4f}")

# 4. SAVE MODEL AND EXPORT METRICS GRAPH
torch.save(model.state_dict(), "stgt_model.pth")
print("\n[SAVED] Model state weights successfully serialized to 'stgt_model.pth'.")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, color='#0072BD', linewidth=2, label='STGT Mean Squared Error')
plt.title('Spatio-Temporal Graph Transformer Training Convergence Curve')
plt.xlabel('Optimization Epochs')
plt.ylabel('Loss Value (MSE)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('stgt_loss_curve.png', dpi=300)
print("[EXPORTED] Training loss metrics graph exported successfully as 'stgt_loss_curve.png'.\n")