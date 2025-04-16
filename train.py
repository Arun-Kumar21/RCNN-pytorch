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
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    self.epochs = epochs
  
  def train(self, train_loader, save_path=None):
    try:
      best_acc = 0
      for epoch in range(self.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
          inputs, targets = inputs.to(self.device), targets.to(self.device)

          self.optimizer.zero_grad()
          outputs = self.model(inputs)
          
          loss = self.criterion(outputs, targets)
          loss.backward()
          self.optimizer.step()

          total_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item() 
         
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
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