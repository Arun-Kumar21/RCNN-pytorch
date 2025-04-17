import torch
import torch.nn as nn
import torch.optim as optim

from config.config import Config
from models.RCNN import RCNN
from data.data_loader import dataloader

class ModelTrainer:
    def __init__(self, model, device=Config.DEVICE, lr=Config.LEARNING_RATE, epochs=Config.EPOCHS):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = self.custom_criterion
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.epochs = epochs

    def custom_criterion(self, cls_pred, bbox_pred, cls_targets, bbox_targets):
        cls_loss = nn.CrossEntropyLoss()(cls_pred, cls_targets)
        reg_loss = nn.SmoothL1Loss()(bbox_pred, bbox_targets)
        return cls_loss + reg_loss, cls_loss, reg_loss

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, cls_targets, bbox_targets, proposal_boxes) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            cls_targets = cls_targets.to(self.device)
            bbox_targets = bbox_targets.to(self.device)

            self.optimizer.zero_grad()
            cls_pred, bbox_pred = self.model(inputs)
            loss, cls_loss, reg_loss = self.criterion(cls_pred, bbox_pred, cls_targets, bbox_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()

            _, predicted = cls_pred.max(1)
            total += cls_targets.size(0)
            correct += predicted.eq(cls_targets).sum().item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, '
                      f'Cls Loss: {cls_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')

        return total_loss / len(dataloader), 100. * correct / total

    def train(self, dataloader, save_path=None):
        try:
            best_acc = 0
            for epoch in range(self.epochs):
                epoch_loss, epoch_acc = self.train_one_epoch(dataloader)
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f'Model saved with accuracy: {best_acc:.2f}%')

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
            if save_path:
                torch.save(self.model.state_dict(), save_path)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            if save_path:
                torch.save(self.model.state_dict(), save_path)
            raise e

        finally:
            if save_path:
                print(f"Successfully saved model at {save_path}")


if __name__ == '__main__':
    model = RCNN(Config.NUM_CLASSES)
    trainer = ModelTrainer(model)
    trainer.train(dataloader, './weights/rcnn_model.pth')