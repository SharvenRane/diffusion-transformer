"""
Training loop with MLflow tracking.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import mlflow
from tqdm import tqdm


class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model.cuda()
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs
        )
        self.scaler = GradScaler()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.cuda(), labels.cuda()
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.val_loader, desc="Evaluating"):
            images, labels = images.cuda(), labels.cuda()
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def train(self):
        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run():
            mlflow.log_params(dict(self.config))
            best_acc = 0
            for epoch in range(self.config.epochs):
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.evaluate()
                self.scheduler.step()

                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, step=epoch)

                print(f"Epoch {epoch+1}/{self.config.epochs} | "
                      f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                      f"Val: {val_loss:.4f}/{val_acc:.4f}")

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save({"model": self.model.state_dict(), "epoch": epoch},
                               "checkpoints/best.pth")
                    mlflow.log_artifact("checkpoints/best.pth")
