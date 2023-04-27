import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

def match_num(pred: torch.Tensor, labels: torch.Tensor):
  pred = pred.argmax(dim = 1)
  matches = (pred == labels).sum()
  return matches

def acc_and_loss(net: nn.Module, dataset: Dataset, batch_size: int):
  """Returns the accuracy and loss of the model `net` iterating through `dataset` with an allocated `batch_size`"""

  loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)
  matches = 0
  loss_sum = 0
  criterion = nn.CrossEntropyLoss()
  for b_data, b_target in loader:
    b_data = b_data.cuda()
    b_target = b_target.cuda()
    b_pred = net(b_data)
    matches += match_num(b_pred, b_target)

    loss_sum += criterion(b_pred, b_target)
    accuracy = matches/len(dataset)

    num_batch = len(dataset)/batch_size + 1
    loss_aver = loss_sum/num_batch
  
  return accuracy, loss_aver

def sketch_and_save(root: str, train_acc: list, val_acc: list, train_loss: list, val_loss: list):
  path_acc = os.path.join(root, '/accuracy_train_val.png')
  path_loss = os.path.join(root, '/loss_train_val.png')

  plt.plot(train_acc)
  plt.plot(val_acc)
  plt.legend(['Train acc', 'Val acc'])
  plt.title('Accuracy')
  plt.savefig(path_acc)

  plt.plot(train_loss)
  plt.plot(val_loss)
  plt.legend(['Train loss', 'Val loss'])
  plt.title('Loss')
  plt.savefig(path_loss)

def Train(model: nn.Module, trainset: Dataset, valset: Dataset, use_subset: bool = True, epoch_num: int = 20, bs: int = 64, lr: float = 0.01, momentum: float = 0.8, weight_decay: float = 5e-4, show_progress: bool = True, save_fig: bool = True):
  """Execute the training session with custom model, trainset, valset, hyperparameters and others
  Parameters description:
  1. `model` is an instance in `nn.Module` abstract class and will be considered during this training session
  2. `trainset` is the dataset used for training
  3. `valset` is the dataset used for validating
  4. `use_subset` indicates whether only a subset or the entire `trainset` and `valset` will be used to acquire the loss and accuracy in the (sub)training set and (sub)validation set
  5. `epoch_num`, `bs`, `lr`, `momentum`, `weight_decay` are hyperparameters
  6. `show_progress` if set to `True`, then the training accuracy will be displayed to the screen after each completed epoch
  7. `save_fig` if set to `True`, plot and save the accuracy and loss graph in the `acc_and_loss` folder
  """
  
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

  subset_indices = [i for i in range(1000)]
  subtrainset = Subset(trainset, indices = subset_indices)
  subvalset = Subset(valset, indices = subset_indices)


  train_acc_rec = []
  val_acc_rec = []
  train_loss_rec = []
  val_loss_rec = []

  for epoch in range(1, epoch_num + 1):
    trainloader = DataLoader(dataset = trainset, batch_size = bs, shuffle = True)
    for data, target in trainloader:
      optimizer.zero_grad()
      data = data.cuda()
      target = target.cuda()

      pred = model(data)
      loss = criterion(pred, target)
      loss.backward()
      optimizer.step()

    with torch.no_grad():
      if not use_subset:
        acc_train, loss_train = acc_and_loss(net = model, dataset = trainset, batch_size = bs)
        acc_val, loss_val = acc_and_loss(model, dataset = valset, batch_size = bs)
      else:
        acc_train, loss_train = acc_and_loss(net = model, dataset = subtrainset, batch_size = bs)
        acc_val, loss_val = acc_and_loss(model, dataset = subvalset, batch_size = bs)
  
    
    train_acc_rec.append(acc_train.cpu())
    train_loss_rec.append(loss_train.item().cpu())
    val_acc_rec.append(acc_val.cpu())
    val_loss_rec.append(loss_val.item().cpu())

    if show_progress:
        print(f'Train acc epoch {epoch}: {acc_train}')
        print(f'Val acc epoch {epoch}: {acc_val}')
  
  if save_fig:
    root = './acc_and_loss'
    if not os.path.isdir(root):
      os.makedirs(root)
    sketch_and_save(root, train_acc_rec, val_acc_rec, train_loss_rec, val_loss_rec)
  

def Test(model: nn.Module, testset: Dataset, save_result_to_txt: bool = True):
  """Execute an weighted model on an testset to get the accuracy and loss"""
  with torch.no_grad():
    acc_test, loss_test = acc_and_loss(net = model, dataset = testset, batch_size = 64)

  result_txt_path = './acc_and_loss/test_result.txt'
  if save_result_to_txt:
    with open(file = result_txt_path, mode = 'a') as f:
      f.write(f'Accuracy on test set: {acc_test};\tLoss on test set: {loss_test}\n')
  else:
    print(f'Accuracy on test set: {acc_test};\tLoss on test set: {loss_test}')
