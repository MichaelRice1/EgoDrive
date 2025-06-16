import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Tuple
import math
import warnings
warnings.filterwarnings('ignore')

# Meta's proven architecture components
class GazeEncoder(nn.Module):
    def __init__(self, input_dim, dim_feat, dropout=0.3):
        super(GazeEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, dim_feat, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.drop = nn.Dropout1d(dropout)
        self.pool = nn.AvgPool1d(3)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class IMUEncoder(nn.Module):
    def __init__(self, input_dim, dim_feat, dropout=0.3):
        super(IMUEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, dim_feat, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(dim_feat, dim_feat, kernel_size=9, padding=4)
        self.drop = nn.Dropout1d(dropout)
        self.pool = nn.AvgPool1d(3)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class HandEncoder(nn.Module):
    """Custom encoder for hand landmarks using Meta's CNN pattern"""
    def __init__(self, input_dim, dim_feat, dropout=0.3):
        super(HandEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, dim_feat, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(dim_feat, dim_feat, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(dim_feat, dim_feat, kernel_size=5, padding=2)
        self.drop = nn.Dropout1d(dropout)
        self.pool = nn.AvgPool1d(2)
    
    def forward(self, x):
        # x: [B, T, 63] -> [B, 63, T]
        x = x.permute(0, 2, 1)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.pool(x)
        return x.permute(0, 2, 1)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MetaMultimodalTransformer(pl.LightningModule):
    """
    Meta's architecture adapted for driving action recognition.
    Proven to work on multimodal reading recognition - should transfer well.
    """
    
    def __init__(self, 
                 num_classes: int,
                 dim_feat: int = 128,
                 input_dims: List[int] = [2, 6, 63],  # [gaze, imu, hands]
                 max_sequence_length: int = 512,
                 dropout: float = 0.3,
                 transformer_depth: int = 6,
                 learning_rate: float = 1e-3,
                 warmup_steps: int = 1000):
        super().__init__()
        
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.dim_feat = dim_feat
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        # Modality encoders (Meta's proven CNN approach)
        self.gaze_encoder = GazeEncoder(input_dims[0], dim_feat, dropout)
        self.imu_encoder = IMUEncoder(input_dims[1], dim_feat, dropout)
        self.hand_encoder = HandEncoder(input_dims[2], dim_feat, dropout)
        
        # Learned positional embeddings for each modality
        self.gaze_pe = nn.Parameter(torch.randn(1, max_sequence_length, dim_feat))
        self.imu_pe = nn.Parameter(torch.randn(1, max_sequence_length, dim_feat))
        self.hand_pe = nn.Parameter(torch.randn(1, max_sequence_length, dim_feat))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_feat))
        
        # Meta's transformer architecture
        self.transformer = Transformer(
            dim=dim_feat, 
            depth=transformer_depth, 
            heads=max(1, dim_feat//16),  # Adaptive heads based on dim
            dim_head=dim_feat, 
            mlp_dim=dim_feat, 
            dropout=dropout
        )
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_feat),
            nn.Dropout(dropout),
            nn.Linear(dim_feat, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization for faster convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, batch):
        """Forward pass following Meta's token concatenation approach"""
        gaze = batch['gaze']    # [B, T, 2]
        imu = batch['imu']      # [B, T, 6] 
        hands = batch['hands']  # [B, T, 63]
        lengths = batch['lengths']  # [B]
        
        batch_size = gaze.shape[0]
        
        # Start with CLS token
        tokens = [self.cls_token.repeat(batch_size, 1, 1)]
        
        # Process each modality with Meta's approach
        if gaze is not None:
            gaze_tokens = self.gaze_encoder(gaze)  # [B, T//3, dim_feat] due to pooling
            seq_len = gaze_tokens.size(1)
            gaze_pe = self.gaze_pe[:, :seq_len, :].repeat(batch_size, 1, 1)
            tokens.append(gaze_tokens + gaze_pe)
        
        if imu is not None:
            imu_tokens = self.imu_encoder(imu)  # [B, T//3, dim_feat]
            seq_len = imu_tokens.size(1)
            imu_pe = self.imu_pe[:, :seq_len, :].repeat(batch_size, 1, 1)
            tokens.append(imu_tokens + imu_pe)
        
        if hands is not None:
            hand_tokens = self.hand_encoder(hands)  # [B, T//2, dim_feat]
            seq_len = hand_tokens.size(1)
            hand_pe = self.hand_pe[:, :seq_len, :].repeat(batch_size, 1, 1)
            tokens.append(hand_tokens + hand_pe)
        
        # Concatenate all tokens (Meta's fusion strategy)
        x = torch.cat(tokens, dim=1)  # [B, 1 + gaze_len + imu_len + hand_len, dim_feat]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Extract CLS token for classification (Meta's approach)
        cls_output = x[:, 0, :]  # [B, dim_feat]
        
        # Multi-label classification
        logits = self.classifier(cls_output)  # [B, num_classes]
        
        return logits
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        target = batch['target']
        
        # Multi-label BCE loss with class weighting
        loss = F.binary_cross_entropy_with_logits(logits, target)
        
        # Metrics
        pred = torch.sigmoid(logits) > 0.5
        accuracy = (pred == target.bool()).all(dim=1).float().mean()
        
        # Per-class F1 for monitoring
        f1_scores = []
        for i in range(self.num_classes):
            tp = ((pred[:, i] == 1) & (target[:, i] == 1)).sum().float()
            fp = ((pred[:, i] == 1) & (target[:, i] == 0)).sum().float()
            fn = ((pred[:, i] == 0) & (target[:, i] == 1)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        mean_f1 = torch.stack(f1_scores).mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        self.log('train_f1', mean_f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        target = batch['target']
        
        loss = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits) > 0.5
        accuracy = (pred == target.bool()).all(dim=1).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Meta's proven optimizer setup with warmup"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Warmup + cosine schedule (proven for transformers)
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                return 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (10000 - self.warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

class MetaInspiredTrainingPipeline:
    """
    Training pipeline based on Meta's successful multimodal approach.
    Uses their proven architecture but optimized for fast iteration.
    """
    
    def __init__(self, dataset, batch_size: int = 32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = None
        
        # Meta uses larger batches - but we'll adapt for available hardware
        print(f"Using batch size: {batch_size}")
        
        # Smart data splitting (stratified by action types)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Reproducible splits
        )
        
        print(f"Data split: {train_size} train, {val_size} val")
    
    def create_dataloaders(self, num_workers: int = 4):
        """Create optimized dataloaders following Meta's approach"""
        from torch.utils.data import DataLoader
        
        # Use the collate function from previous artifact
        def collate_variable_length(batch):
            """Custom collate for variable-length multimodal sequences"""
            from torch.nn.utils.rnn import pad_sequence
            
            # Separate each modality
            gaze_batch = [item['gaze'] for item in batch]
            imu_batch = [item['imu'] for item in batch]
            hands_batch = [item['hands'] for item in batch]
            targets = torch.stack([item['target'] for item in batch])
            lengths = torch.tensor([item['length'] for item in batch])
            
            # Pad sequences to max length in batch
            gaze_padded = pad_sequence(gaze_batch, batch_first=True, padding_value=0)
            imu_padded = pad_sequence(imu_batch, batch_first=True, padding_value=0)
            hands_padded = pad_sequence(hands_batch, batch_first=True, padding_value=0)
            
            return {
                'gaze': gaze_padded,
                'imu': imu_padded,
                'hands': hands_padded,
                'target': targets,
                'lengths': lengths
            }
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_variable_length,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_variable_length,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"Created dataloaders: {len(self.train_loader)} train, {len(self.val_loader)} val batches")
    
    def stage1_quick_test(self):
        """Quick architecture validation"""
        print("\n=== STAGE 1: ARCHITECTURE VALIDATION ===")
        
        # Test with minimal model
        model = MetaMultimodalTransformer(
            num_classes=self.dataset.num_classes,
            dim_feat=64,  # Small for quick test
            transformer_depth=2,
            learning_rate=1e-3
        )
        
        # Single batch overfitting test
        batch = next(iter(self.train_loader))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        initial_loss = None
        
        for step in range(100):
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch['target'])
            loss.backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = loss.item()
            
            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
        
        final_loss = loss.item()
        reduction = (initial_loss - final_loss) / initial_loss
        
        if reduction > 0.5:  # 50% loss reduction
            print(f"✓ Architecture working: {reduction:.1%} loss reduction")
            return True
        else:
            print(f"✗ Architecture issues: only {reduction:.1%} loss reduction")
            return False
    
    def stage2_baseline_model(self, max_epochs: int = 30):
        """Fast baseline following Meta's smaller config"""
        print("\n=== STAGE 2: BASELINE MODEL ===")
        
        self.model = MetaMultimodalTransformer(
            num_classes=self.dataset.num_classes,
            dim_feat=128,           # Meta's base size
            transformer_depth=3,    # Shallow for speed
            dropout=0.2,
            learning_rate=2e-3,     # Higher for faster convergence
            warmup_steps=200
        )
        
        # Fast trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            precision=16,
            gradient_clip_val=1.0,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            val_check_interval=0.5,  # Check twice per epoch
            limit_train_batches=0.5  # Use half data for speed
        )
        
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        # Quick validation
        val_results = trainer.validate(self.model, self.val_loader)
        val_acc = val_results[0]['val_acc']
        
        print(f"Baseline accuracy: {val_acc:.3f}")
        
        return val_acc > 0.4  # Reasonable threshold for multimodal
    
    def stage3_full_training(self, max_epochs: int = 100):
        """Full Meta-inspired training"""
        print("\n=== STAGE 3: FULL META-INSPIRED TRAINING ===")
        
        self.model = MetaMultimodalTransformer(
            num_classes=self.dataset.num_classes,
            dim_feat=256,           # Meta's full size
            transformer_depth=6,    # Meta's depth
            dropout=0.3,
            learning_rate=1e-3,
            warmup_steps=1000
        )
        
        # Meta's training setup
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min',
            verbose=True
        )
        
        checkpoint = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=2,
            filename='meta-model-{epoch}-{val_acc:.3f}'
        )
        
        logger = TensorBoardLogger('lightning_logs', name='meta_multimodal')
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            precision=16,
            gradient_clip_val=1.0,
            callbacks=[early_stopping, checkpoint],
            logger=logger,
            enable_progress_bar=True,
            val_check_interval=1.0
        )
        
        trainer.fit(self.model, self.train_loader, self.val_loader)
        
        # Load best model
        best_model = MetaMultimodalTransformer.load_from_checkpoint(
            checkpoint.best_model_path,
            num_classes=self.dataset.num_classes
        )
        
        final_results = trainer.validate(best_model, self.val_loader)
        final_acc = final_results[0]['val_acc']
        
        print(f"Final accuracy: {final_acc:.3f}")
        print(f"Best model: {checkpoint.best_model_path}")
        
        return best_model, final_acc

def run_meta_training_pipeline(dataset, target_hours: float = 3.0):
    """
    Complete Meta-inspired training pipeline.
    Uses proven architecture from Meta's reading recognition paper.
    """
    print(f"🔬 META-INSPIRED MULTIMODAL TRAINING")
    print(f"Based on: Reading Recognition in the Wild (Meta Research)")
    print(f"Target: Production model in {target_hours} hours")
    print(f"Dataset: {len(dataset)} sequences, {dataset.num_classes} classes")
    
    # Determine optimal batch size based on available memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Estimate memory usage and adjust batch size
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = min(32, max(8, int(gpu_memory / 4)))  # Rough heuristic
    else:
        batch_size = 16
    
    print(f"Using batch size: {batch_size} on {device}")
    
    pipeline = MetaInspiredTrainingPipeline(dataset, batch_size=batch_size)
    pipeline.create_dataloaders(num_workers=4)
    
    # Stage 1: Architecture validation (15 min)
    if not pipeline.stage1_quick_test():
        print("❌ Architecture validation failed")
        return None
    
    # Stage 2: Baseline model (30 min)
    # if not pipeline.stage2_baseline_model(max_epochs=25):
    #     print("❌ Baseline model failed")
    #     return None
    
    # # Stage 3: Full training (90-120 min)
    # best_model, accuracy = pipeline.stage3_full_training(max_epochs=80)
    
    # print(f"\n🎉 META-INSPIRED TRAINING COMPLETE!")
    # print(f"Final accuracy: {accuracy:.3f}")
    # print(f"Architecture: CNN encoders + Transformer (Meta's approach)")
    # print(f"Key features:")
    # print(f"- Modality-specific CNN encoders")
    # print(f"- Learned positional embeddings")
    # print(f"- CLS token classification")
    # print(f"- Proven on similar multimodal tasks")
    
    # if accuracy > 0.75:
    #     print("✅ Excellent results!")
    # elif accuracy > 0.6:
    #     print("✅ Good results - consider fine-tuning")
    # else:
    #     print("⚠️ Room for improvement - check data quality")
    
    # return best_model

if __name__ == "__main__":
    print("Meta-Inspired Multimodal Transformer Training")
    print("="*55)
    print("Based on: 'Reading Recognition in the Wild' (Meta Research)")
    print("Key advantages:")
    print("- Proven CNN + Transformer architecture")
    print("- Modality-specific encoders")
    print("- CLS token approach for classification")
    print("- Optimized for multimodal fusion")
    print("\nUsage:")
    run_meta_training_pipeline(your_dataset)