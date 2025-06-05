import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.backbones.efficientnet import efficientnet_b0
from solo.backbones.resnet import resnet18

def simple_contrastive_loss(z1, z2, temperature=0.1):
    """Simple contrastive loss to test feature learning."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Positive pairs
    pos_logits = torch.sum(z1 * z2, dim=1) / temperature
    
    # Negative pairs (within batch)
    neg_logits = torch.mm(z1, z2.t()) / temperature
    
    # Simple InfoNCE-like loss
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(neg_logits, labels)
    
    return loss

def test_backbone_learning():
    """Test if backbone features are actually learning during SSL training."""
    
    # Create models
    efficientnet = efficientnet_b0('mocov3', pretrained=True)
    resnet = resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    
    # Get feature dimensions
    eff_dim = efficientnet.num_features  # 1280
    res_dim = 512  # ResNet18 feature dim
    
    # Create simple projectors
    eff_projector = nn.Sequential(
        nn.Linear(eff_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    res_projector = nn.Sequential(
        nn.Linear(res_dim, 256), 
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    # Create dummy data
    batch_size = 32
    x1 = torch.randn(batch_size, 3, 224, 224)
    x2 = torch.randn(batch_size, 3, 224, 224)  # Different augmentation
    
    # Test initial features
    efficientnet.eval()
    resnet.eval()
    with torch.no_grad():
        eff_feats_init = efficientnet(x1)
        res_feats_init = resnet(x1)
    
    print("=== Initial Feature Analysis ===")
    print(f"EfficientNet features - mean: {eff_feats_init.mean().item():.6f}, std: {eff_feats_init.std().item():.6f}")
    print(f"ResNet features - mean: {res_feats_init.mean().item():.6f}, std: {res_feats_init.std().item():.6f}")
    print(f"EfficientNet norm per sample: {torch.norm(eff_feats_init, dim=1).mean().item():.6f}")
    print(f"ResNet norm per sample: {torch.norm(res_feats_init, dim=1).mean().item():.6f}")
    
    # Simulate training
    efficientnet.train()
    resnet.train()
    eff_projector.train()
    res_projector.train()
    
    eff_optimizer = torch.optim.SGD(list(efficientnet.parameters()) + list(eff_projector.parameters()), lr=0.01)
    res_optimizer = torch.optim.SGD(list(resnet.parameters()) + list(res_projector.parameters()), lr=0.01)
    
    print("\n=== Training Simulation ===")
    for step in range(50):
        # EfficientNet forward
        eff_feats1 = efficientnet(x1)
        eff_feats2 = efficientnet(x2)
        eff_z1 = eff_projector(eff_feats1)
        eff_z2 = eff_projector(eff_feats2)
        eff_loss = simple_contrastive_loss(eff_z1, eff_z2)
        
        # ResNet forward
        res_feats1 = resnet(x1)
        res_feats2 = resnet(x2)
        res_z1 = res_projector(res_feats1)
        res_z2 = res_projector(res_feats2)
        res_loss = simple_contrastive_loss(res_z1, res_z2)
        
        # Backward
        eff_optimizer.zero_grad()
        eff_loss.backward()
        eff_optimizer.step()
        
        res_optimizer.zero_grad()
        res_loss.backward()
        res_optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: EfficientNet loss: {eff_loss.item():.4f}, ResNet loss: {res_loss.item():.4f}")
    
    # Test final features
    efficientnet.eval()
    resnet.eval()
    with torch.no_grad():
        eff_feats_final = efficientnet(x1)
        res_feats_final = resnet(x1)
    
    print("\n=== Final Feature Analysis ===")
    print(f"EfficientNet features - mean: {eff_feats_final.mean().item():.6f}, std: {eff_feats_final.std().item():.6f}")
    print(f"ResNet features - mean: {res_feats_final.mean().item():.6f}, std: {res_feats_final.std().item():.6f}")
    
    # Check how much backbone features changed
    eff_change = torch.norm(eff_feats_final - eff_feats_init, dim=1).mean().item()
    res_change = torch.norm(res_feats_final - res_feats_init, dim=1).mean().item()
    
    print(f"\n=== Feature Change Analysis ===")
    print(f"EfficientNet backbone feature change: {eff_change:.6f}")
    print(f"ResNet backbone feature change: {res_change:.6f}")
    
    # Test eval mode difference
    efficientnet.train()
    resnet.train()
    with torch.no_grad():
        eff_feats_train = efficientnet(x1)
        res_feats_train = resnet(x1)
    
    eff_eval_diff = torch.norm(eff_feats_final - eff_feats_train, dim=1).mean().item()
    res_eval_diff = torch.norm(res_feats_final - res_feats_train, dim=1).mean().item()
    
    print(f"\n=== Train/Eval Mode Difference ===")
    print(f"EfficientNet train/eval difference: {eff_eval_diff:.6f}")
    print(f"ResNet train/eval difference: {res_eval_diff:.6f}")

if __name__ == "__main__":
    test_backbone_learning() 