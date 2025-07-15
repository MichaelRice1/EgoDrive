import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import os
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


class EgoDriveClassifier(L.LightningModule):
    def __init__(
        self, 
        model, 
        num_classes=6, 
        lr=1e-4, 
        use_loss_weight=False, 
        loss_weight=None,
        training_phase='multimodal',  # 'multimodal', 'hallucination', 'rgb_only'
        use_multi_task=False,
        warmup_steps=500,
        weight_decay=1e-5
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.training_phase = training_phase
        self.use_multi_task = use_multi_task
        self.task = "multiclass" if num_classes > 2 else "binary"
        self.num_classes = num_classes
        
        # Loss functions
        if use_loss_weight and loss_weight is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        # Metrics
        self.accuracy = Accuracy(num_classes=num_classes, average='micro', task=self.task)
        self.precision = Precision(num_classes=num_classes, average='macro', task=self.task)
        self.recall = Recall(num_classes=num_classes, average='macro', task=self.task)
        self.f1score = F1Score(num_classes=num_classes, average='macro', task=self.task)
        self.auroc = AUROC(num_classes=num_classes, average='macro', task=self.task)
        self.confmat = ConfusionMatrix(num_classes=num_classes, task=self.task)
        
        # Per-class metrics
        self.per_class_accuracy = Accuracy(num_classes=num_classes, average='none', task=self.task)
        self.per_class_precision = Precision(num_classes=num_classes, average='none', task=self.task)
        self.per_class_recall = Recall(num_classes=num_classes, average='none', task=self.task)
        
        # Validation metrics
        self.val_loss = torchmetrics.MeanMetric()
        self.val_pred = torchmetrics.CatMetric()
        self.val_label = torchmetrics.CatMetric()
        
        # Test metrics
        self.test_pred = torchmetrics.CatMetric()
        self.test_label = torchmetrics.CatMetric()
        
        # Behavior class names
        self.class_names = [
            'left wing mirror check',
            'rear view mirror check',
            'right wing mirror check',
            'driving',
            'idle',
            'mobile phone usage'
        ]
        
        # For tracking best model
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass handling different training phases"""
        # Extract labels if present
        labels = batch.get('label', None)
        
        # Remove label from input dict if present
        input_data = {k: v for k, v in batch.items() if k != 'label'}
        
        # Forward through model with appropriate training phase
        outputs = self.model(input_data, labels=labels, training_phase=self.training_phase)
        
        return outputs
    
    def extract_logits(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract logits from model output dictionary"""
        if self.training_phase == 'multimodal':
            return outputs.get('teacher_logits', outputs.get('logits'))
        else:
            return outputs['logits']
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute loss based on training phase"""
        if 'loss' in outputs:
            # Model already computed the loss
            return outputs['loss']
        
        # Compute cross-entropy loss
        logits = self.extract_logits(outputs)
        ce_loss = self.loss_fn(logits, labels)
        
        # Add auxiliary losses if available
        total_loss = ce_loss
        
        if self.training_phase == 'hallucination' and 'hallucination_losses' in outputs:
            for loss_name, loss_val in outputs['hallucination_losses'].items():
                total_loss += 0.1 * loss_val  # Weight hallucination losses
                
        if 'kd_loss' in outputs:
            total_loss += outputs['kd_loss']
            
        return total_loss
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        target = batch['label']
        
        loss = self.compute_loss(outputs, target)
        logits = self.extract_logits(outputs)
        pred = torch.argmax(logits, -1)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        acc = self.accuracy(pred, target)
        self.log("train_acc", acc, prog_bar=True)
        
        # Log additional metrics for different phases
        if self.training_phase == 'hallucination' and 'hallucination_losses' in outputs:
            for loss_name, loss_val in outputs['hallucination_losses'].items():
                self.log(f"train_{loss_name}", loss_val)
                
        if 'modality_weights' in outputs:
            for i, weight in enumerate(outputs['modality_weights']):
                modality_names = ['rgb', 'gaze', 'imu', 'objects', 'hands']
                if i < len(modality_names):
                    self.log(f"modality_weight_{modality_names[i]}", weight)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        target = batch['label']
        
        loss = self.compute_loss(outputs, target)
        logits = self.extract_logits(outputs)
        
        # Update metrics
        pred_probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, -1)
        
        self.val_loss.update(value=loss)
        self.val_pred.update(value=pred)
        self.val_label.update(value=target)
        self.auroc.update(preds=pred_probs, target=target)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        target = batch['label']
        
        logits = self.extract_logits(outputs)
        pred_probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(logits, -1)
        
        self.test_pred.update(value=pred)
        self.test_label.update(value=target)
        self.auroc.update(preds=pred_probs, target=target)
        
    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        logits = self.extract_logits(outputs)
        return F.softmax(logits, dim=-1)
    
    def log_confusion_matrix(self, confmat, stage='val'):
        """Log confusion matrix as both table and heatmap"""
        # Normalize confusion matrix
        confmat_norm = (confmat.float() / confmat.sum(dim=1, keepdim=True)).cpu()
        confmat_norm[torch.isnan(confmat_norm)] = 0
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confmat_norm.cpu().numpy(),
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'{stage.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Log to wandb
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(confmat_norm, annot=True, fmt="g", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        
        self.logger.experiment.log({
            "val_confusion_matrix": wandb.Image(fig),
            "global_step": self.global_step
        })


        plt.close()
        
        # Also log as table
        data = []
        columns = ["True\\Pred"] + self.class_names
        for i, true_class in enumerate(self.class_names):
            row = [true_class]
            for j in range(len(self.class_names)):
                row.append(f"{confmat_norm[i, j]:.3f}")
            data.append(row)
        
        # self.log(key=f"{stage}_conf_mat_table", columns=columns, data=data)
    
    def on_validation_epoch_end(self):
        avg_loss = self.val_loss.compute()
        preds = self.val_pred.compute()
        targets = self.val_label.compute()
        
        # Compute metrics
        val_acc = self.accuracy(preds, targets)
        val_auc = self.auroc.compute()
        val_recall = self.recall(preds, targets)
        val_precision = self.precision(preds, targets)
        val_f1score = self.f1score(preds, targets)
        confmat = self.confmat(preds, targets)
        
        # Per-class metrics
        per_class_acc = self.per_class_accuracy(preds, targets)
        per_class_prec = self.per_class_precision(preds, targets)
        per_class_rec = self.per_class_recall(preds, targets)
        
        # Log main metrics
        values = {
            "val_loss": avg_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1_score": val_f1score,
        }
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log per-class metrics
        for i, class_name in enumerate(self.class_names):
            acc_i = per_class_acc[i].item()
            prec_i = per_class_prec[i].item()
            rec_i = per_class_rec[i].item()
            
            self.log(f"val_acc_{class_name}", acc_i)
            self.log(f"val_prec_{class_name}", prec_i)
            self.log(f"val_rec_{class_name}", rec_i)

            # Debugging rear view mirror precision anomaly
            # if class_name == "idle":
            #     print(f"[DEBUG] Class '{class_name}' - idx {i}")
            #     print(f"[DEBUG] Accuracy: {acc_i:.4f}, Precision: {prec_i:.4f}, Recall: {rec_i:.4f}")
            #     cm = self.confmat.compute()
            #     print(f"[DEBUG] Confusion matrix row (true={i}): {cm[i].tolist()}")
            #     print(f"[DEBUG] Confusion matrix col (pred={i}): {[cm[j, i].item() for j in range(len(self.class_names))]}")
        
        # Log confusion matrix
        self.log_confusion_matrix(confmat, stage='val')
        
        # Track best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.log("best_val_acc", self.best_val_acc)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.log("best_val_loss", self.best_val_loss)
        
        # Reset metrics
        self.val_loss.reset()
        self.val_pred.reset()
        self.val_label.reset()
        self.auroc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1score.reset()
        self.confmat.reset()
        self.per_class_accuracy.reset()
        self.per_class_precision.reset()
        self.per_class_recall.reset()
    
    def on_test_epoch_end(self):
        preds = self.test_pred.compute()
        targets = self.test_label.compute()
        
        # Save predictions and targets for verification
        predictions_np = preds.cpu().numpy().astype(int)
        targets_np = targets.cpu().numpy().astype(int)
        
        # Create comprehensive results dictionary
        test_results = {
            'predictions': predictions_np,
            'targets': targets_np,
            'raw_predictions': preds.cpu(),
            'raw_targets': targets.cpu()
        }
        
        # Add logits and probabilities if available
        if hasattr(self, 'all_logits') and len(self.all_logits) > 0:
            test_results['logits'] = np.array(self.all_logits)
            test_results['probabilities'] = np.array(self.all_probs)
            test_results['batch_info'] = self.batch_info
        
        # Compute metrics
        test_acc = self.accuracy(preds, targets)
        test_auc = self.auroc.compute()
        test_recall = self.recall(preds, targets)
        test_precision = self.precision(preds, targets)
        test_f1score = self.f1score(preds, targets)
        confmat = self.confmat(preds, targets)
        
        # Per-class metrics
        per_class_acc = self.per_class_accuracy(preds, targets)
        per_class_prec = self.per_class_precision(preds, targets)
        per_class_rec = self.per_class_recall(preds, targets)
        
        # Add metrics to results
        test_results['metrics'] = {
            "test_auc": test_auc.item() if hasattr(test_auc, 'item') else test_auc,
            "test_precision": test_precision.item() if hasattr(test_precision, 'item') else test_precision,
            "test_recall": test_recall.item() if hasattr(test_recall, 'item') else test_recall,
            "test_f1_score": test_f1score.item() if hasattr(test_f1score, 'item') else test_f1score,
            "test_accuracy": test_acc.item() if hasattr(test_acc, 'item') else test_acc,
            "confusion_matrix": confmat.cpu().numpy(),
            "per_class_accuracy": per_class_acc.cpu().numpy(),
            "per_class_precision": per_class_prec.cpu().numpy(),
            "per_class_recall": per_class_rec.cpu().numpy()
        }
        
        # Add class names and distributions
        test_results['class_names'] = self.class_names
        test_results['distributions'] = {
            'predictions': np.bincount(predictions_np, minlength=len(self.class_names)),
            'targets': np.bincount(targets_np, minlength=len(self.class_names))
        }
        
        # Check for potential issues
        unique_preds = np.unique(predictions_np)
        unique_targets = np.unique(targets_np)
        
        test_results['diagnostics'] = {
            'unique_predictions': unique_preds,
            'unique_targets': unique_targets,
            'num_samples': len(predictions_np),
            'single_class_prediction': len(unique_preds) == 1,
            'predicted_class_if_single': unique_preds[0] if len(unique_preds) == 1 else None
        }
        
        # Check for NaN values if logits are available
        if 'logits' in test_results:
            nan_count = np.isnan(test_results['logits']).sum()
            test_results['diagnostics']['nan_count'] = nan_count
            test_results['diagnostics']['has_nan'] = nan_count > 0
        
        # Save results to file
        import pickle
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'test_results_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write(f"=== COMPREHENSIVE TEST RESULTS ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total samples: {len(predictions_np)}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test AUC: {test_auc:.4f}\n")
            f.write(f"Test Precision: {test_precision:.4f}\n")
            f.write(f"Test Recall: {test_recall:.4f}\n")
            f.write(f"Test F1 Score: {test_f1score:.4f}\n\n")
            
            # Diagnostic information
            f.write(f"=== DIAGNOSTICS ===\n")
            if test_results['diagnostics']['single_class_prediction']:
                f.write(f"WARNING: Model only predicts class {test_results['diagnostics']['predicted_class_if_single']} ({self.class_names[test_results['diagnostics']['predicted_class_if_single']]})\n")
            
            if 'has_nan' in test_results['diagnostics'] and test_results['diagnostics']['has_nan']:
                f.write(f"WARNING: Found {test_results['diagnostics']['nan_count']} NaN values in logits\n")
            
            f.write(f"Unique predictions: {test_results['diagnostics']['unique_predictions']}\n")
            f.write(f"Unique targets: {test_results['diagnostics']['unique_targets']}\n")
            f.write(f"Prediction distribution: {test_results['distributions']['predictions']}\n")
            f.write(f"Target distribution: {test_results['distributions']['targets']}\n\n")
            
            # Per-class metrics
            f.write(f"=== PER-CLASS METRICS ===\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Accuracy: {per_class_acc[i]:.4f}\n")
                f.write(f"  Precision: {per_class_prec[i]:.4f}\n")
                f.write(f"  Recall: {per_class_rec[i]:.4f}\n\n")
            
            # Confusion matrix
            f.write(f"=== CONFUSION MATRIX ===\n")
            f.write(f"True\\Pred\t" + "\t".join(self.class_names) + "\n")
            for i, true_class in enumerate(self.class_names):
                row = [true_class]
                for j in range(len(self.class_names)):
                    row.append(str(confmat[i, j].item()))
                f.write("\t".join(row) + "\n")
            
            f.write(f"\n=== NORMALIZED CONFUSION MATRIX ===\n")
            confmat_norm = (confmat.float() / confmat.sum(dim=1, keepdim=True)).cpu()
            confmat_norm[torch.isnan(confmat_norm)] = 0
            f.write(f"True\\Pred\t" + "\t".join(self.class_names) + "\n")
            for i, true_class in enumerate(self.class_names):
                row = [true_class]
                for j in range(len(self.class_names)):
                    row.append(f"{confmat_norm[i, j]:.3f}")
                f.write("\t".join(row) + "\n")
        
        # Log to wandb (your existing logging)
        values = {
            "test_auc": test_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1score,
            "test_accuracy": test_acc,
        }
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log confusion matrix
        self.log_confusion_matrix(confmat, stage='test')
        
        # Create detailed test report
        report_data = []
        for i, class_name in enumerate(self.class_names):
            report_data.append([
                class_name,
                f"{per_class_acc[i]:.3f}",
                f"{per_class_prec[i]:.3f}",
                f"{per_class_rec[i]:.3f}",
                f"{2 * per_class_prec[i] * per_class_rec[i] / (per_class_prec[i] + per_class_rec[i] + 1e-8):.3f}"
            ])
        
        columns = ["Class", "Accuracy", "Precision", "Recall", "F1-Score"]
        table = wandb.Table(columns=columns, data=report_data)
        self.logger.experiment.log({"test/per_class_metrics": table})
        
        # Log predictions vs targets table to wandb
        pred_target_data = []
        for i in range(min(100, len(predictions_np))):  # Log first 100 samples
            pred_target_data.append([
                i,
                int(targets_np[i]),
                self.class_names[int(targets_np[i])],
                int(predictions_np[i]),
                self.class_names[int(predictions_np[i])],
                "✓" if predictions_np[i] == targets_np[i] else "✗"
            ])
        
        pred_columns = ["Sample", "True_Label_ID", "True_Label", "Pred_Label_ID", "Pred_Label", "Correct"]
        pred_table = wandb.Table(columns=pred_columns, data=pred_target_data)
        self.logger.experiment.log({"test/predictions_vs_targets": pred_table})
        
        # Log prediction distribution
        pred_dist_data = []
        for i, class_name in enumerate(self.class_names):
            pred_count = test_results['distributions']['predictions'][i]
            target_count = test_results['distributions']['targets'][i]
            pred_dist_data.append([class_name, target_count, pred_count, pred_count - target_count])
        
        dist_columns = ["Class", "True_Count", "Pred_Count", "Difference"]
        dist_table = wandb.Table(columns=dist_columns, data=pred_dist_data)
        self.logger.experiment.log({"test/class_distributions": dist_table})
        
        # Print comprehensive summary
        print(f"\n=== COMPREHENSIVE TEST RESULTS ===")
        print(f"Results saved to: {filename}")
        print(f"Total samples: {len(predictions_np)}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Print diagnostic information
        if test_results['diagnostics']['single_class_prediction']:
            print(f"⚠️ WARNING: Model only predicts class {test_results['diagnostics']['predicted_class_if_single']} ({self.class_names[test_results['diagnostics']['predicted_class_if_single']]})")
        
        if 'has_nan' in test_results['diagnostics'] and test_results['diagnostics']['has_nan']:
            print(f"⚠️ WARNING: Found {test_results['diagnostics']['nan_count']} NaN values in logits")
        
        print(f"Unique predictions: {unique_preds}")
        print(f"Unique targets: {unique_targets}")
        print(f"Prediction distribution: {test_results['distributions']['predictions']}")
        print(f"Target distribution: {test_results['distributions']['targets']}")
        
        # Log additional insights for driving behaviors (your commented code)
        # phone_recall = per_class_rec[1] # Mobile phone use
        # self.log("test_phone_detection_recall", phone_recall)
        # 
        # # Log mirror check performance
        # mirror_recalls = [per_class_rec[2], per_class_rec[3], per_class_rec[4]]
        # avg_mirror_recall = sum(mirror_recalls) / len(mirror_recalls)
        # self.log("test_mirror_check_avg_recall", avg_mirror_recall)
        
        # Reset metrics
        self.test_pred.reset()
        self.test_label.reset()
        self.auroc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1score.reset()
        self.confmat.reset()
        
        # Clear additional logging data if present
        if hasattr(self, 'all_logits'):
            self.all_logits.clear()
        if hasattr(self, 'all_probs'):
            self.all_probs.clear()
        if hasattr(self, 'batch_info'):
            self.batch_info.clear()
    
    def configure_optimizers(self):
        # Different optimizer configs for different training phases
        # Start with very low LR to prevent early collapse

        if self.training_phase == 'multimodal':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.training_phase == 'hallucination':
            # Only optimize hallucinator and classifier
            params = []
            for name, param in self.model.named_parameters():
                if 'hallucinator' in name or 'classifier' in name:
                    params.append(param)
            optimizer = optim.AdamW(params, lr=self.lr * 0.5, weight_decay=self.weight_decay)
        else:  # rgb_only
            # Only optimize classifier
            params = []
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    params.append(param)
            optimizer = optim.AdamW(params, lr=self.lr * 0.1, weight_decay=self.weight_decay)
        
        # Scheduler
        scheduler = WarmupScheduler(optimizer, warmup_steps=self.warmup_steps)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def on_train_epoch_end(self):
        """Log training phase info"""
        self.log("training_phase", 
                 {"multimodal": 0, "hallucination": 1, "rgb_only": 2}[self.training_phase])


