import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn

import os
import zipfile

from pathlib import Path
import requests


def iterate_dir(dir_path):
  """
  Walks through the target directory
  :param dir_path: target dir
  :return:
  - number of subdirectories
  - number of images
  - name of each subdirectory
  """

  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")


def accuracy_fn(y_true, y_pred):
  """
  Calculate accuracy between truth labels and prediction labels.
  :param y_true: Truth labels
  :param y_pred: Prediction labels
  :return:
  """
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc


def print_train_time(start: float,
                     end: float,
                     device=None):
  """
  Prints difference between start and end time.
  :param start: Start time of computation
  :param end: End time of computation
  :param device: The device the code works
  :return:
  """
  total_time = end - start
  return total_time


def set_seeds(seed: int = 42):
  """
  Sets random seed for torch opearations.
  :param seed (int, optional): Random seed to set.
  """
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


def eval_model(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               accuracy_fn):
  """
  :param model: The target model
  :param data_loader: Test data loader
  :param loss_fn: Selected loss function
  :param accuracy_fn: Custom accuracy function
  :return: A dict of model data.
  """

  loss, acc = 0, 0
  model.eval()

  with torch.inference_mode():
    for X, y in data_loader:
      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y, y_pred)

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__,
          "model_loss": loss.item(),
          "model_acc": acc}


def train_step_with_batch(model: nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          loss_fn: nn.Module,
                          optimizer: torch.optim,
                          accuracy_fn,
                          device: torch.device):
  """
  :param model (torch.nn.Module): The target model
  :param data_loader (torch.utils.data.DataLoader): The train data loader
  :param loss_fn (torch.nn.Module): Selected loss function
  :param optimizer (torch.optim): Selected optimizer
  :param accuracy_fn: Custom accuracy function
  :param device (torch.device): The device the model runs
  :return: Training the target model in the data loader.
  """

  train_loss, train_acc = 0, 0

  model.train()

  for batch, (X, y) in enumerate(data_loader):
    X, y = X.to(device), y.to(device)

    y_pred = model(X)  # logits

    loss = loss_fn(y_pred, y)
    train_loss += loss

    acc = accuracy_fn(y, y_pred.argmax(dim=1))
    train_acc += acc

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)

  print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}")


def test_step_with_batches(model: nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           loss_fn: torch.nn.Module,
                           accuracy_fn,
                           device: torch.device):
  """
  :param model (torch.nn.Module): The target model
  :param data_loader (torch.utils.data.DataLoader): The test data loader
  :param loss_fn (torch.nn.Module): Selected loss function
  :param optimizer (torch.optim): Selected optimizer
  :param accuracy_fn: Custom accuracy function
  :param device (torch.device): The device the model runs
  :return: Testing the target model in the data loader.
  """

  test_loss, test_acc = 0, 0

  model.eval()

  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      test_pred = model(X)

      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}")
