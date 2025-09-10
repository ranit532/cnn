import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.models.lenet import LeNet
from src.models.get_models import get_model
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def train(args):
    """
    Main training loop.
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))

        # Get data loaders
        train_loader, test_loader, label_map = get_dataloaders(
            "data/processed/train.csv", 
            "data/processed/test.csv", 
            batch_size=args.batch_size
        )
        
        # Get the model
        if args.model_name == 'lenet':
            model = LeNet(num_classes=len(label_map))
        else:
            model = get_model(args.model_name, num_classes=len(label_map), pretrained=True)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # Log training loss
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)

            # Evaluation
            model.eval()
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.numpy())
                    all_preds.extend(predicted.numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

        # Create and log confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_map.keys(), yticklabels=label_map.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Log the model
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--model_name', type=str, default='lenet', help='Model to train (lenet, resnet18, efficientnet_b0, densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()
    train(args)