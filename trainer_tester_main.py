import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from imagenet_dataset import get_imagenet_datasets
import os

from resnet18 import ResNet18
from resnet34 import ResNet34
from resnet50 import ResNet50
from resnet101 import ResNet101
from resnet152 import ResNet152

def plot_results(image_batch, predictions, truth, image_name = "plot"):

    plot_columns = 4
    plot_rows = image_batch.shape[0] // plot_columns

    img_count = plot_columns * plot_rows

    fig, axes = plt.subplots(plot_rows, plot_columns, figsize=(plot_columns * 4, plot_rows * 3))
    plt.subplots_adjust(hspace = 0.4)

    for img_idx in range(img_count):
        row = img_idx // plot_columns
        col = img_idx % plot_columns

        img = image_batch[img_idx]
        img = np.transpose(img,(1,2,0))

        axes[row, col].imshow(img)

        predicted_class = dataset_test.get_class_name(predictions[img_idx])
        actual_class = dataset_test.get_class_name(truth[img_idx])
        axes[row, col].set_title(f"Predicted class {predicted_class} \n but actually {actual_class}")

    plt.savefig(f"{image_name}.jpg")
    plt.close()

NUM_CLASSES = None
NUM_CLASSES = 1000

data_path = "/Users/martinsf/ai/deep_learning_projects/data/imagenet_images"
dataset_train, dataset_test = get_imagenet_datasets(data_path, num_classes = NUM_CLASSES)

if NUM_CLASSES == None:
    NUM_CLASSES = dataset_train.get_number_of_classes()

NUM_TRAIN_SAMPLES = dataset_train.get_number_of_samples()
NUM_TEST_SAMPLES= dataset_test.get_number_of_samples()

print(f"train_samples  {NUM_TRAIN_SAMPLES} test_samples {NUM_TEST_SAMPLES}")

print(f'num_classes {NUM_CLASSES}')
NUM_EPOCHS = 10000
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
#DEVICE = 'cuda'
DEVICE = 'cpu'

model_resnet = ResNet50(class_num = NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(params = model_resnet.parameters(), lr = LEARNING_RATE)

def layers_debug(optim):
    layer_count = 0
    for var_name in optim.state_dict():
        shape = optim.state_dict()[var_name].shape
        if len(optim.state_dict()[var_name].shape)>1:
            layer_count += 1

        print(f"{var_name}\t\t{optim.state_dict()[var_name].shape}")
    print(layer_count)


layers_debug(model_resnet)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of resnet params {count_parameters(model_resnet)}")

trained_models_path = "./trained_models"

if not os.path.exists(trained_models_path):
    os.mkdir(trained_models_path)

last_model_path = os.path.join(trained_models_path, "last.pt")
best_model_path = os.path.join(trained_models_path, "best.pt")
best_test_accuracy = 0

data_loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle = True)
data_loader_test = DataLoader(dataset_test, BATCH_SIZE, shuffle = True)

for epoch in range(NUM_EPOCHS):

    print(f"Starting epoch {epoch}")

    #TRAINING

    model_resnet = model_resnet.train()

    epoch_train_losses = []
    epoch_train_true_positives = 0

    for batch_idx, batch in enumerate(data_loader_train):


        x = batch['image'].to(DEVICE)
        y = batch['cls'].to(DEVICE)

        y_one_hot = torch.zeros(x.shape[0], NUM_CLASSES).to(DEVICE)
        y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

        labels = batch['class_name']

        y_prim = model_resnet.forward(x)

        loss = torch.sum(-y_one_hot * torch.log(y_prim))
        epoch_train_losses.append(loss.detach().to('cpu').numpy())

        epoch_train_true_positives += torch.sum(y_prim.argmax(dim=1) == y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    epoch_accuracy = float(epoch_train_true_positives) / float(NUM_TRAIN_SAMPLES)
    print(f"true_positives {epoch_train_true_positives} from {NUM_TRAIN_SAMPLES} samples")
    print(f"Epoch {epoch} train mean loss is {np.mean(epoch_train_losses)}")
    print(f"Epoch {epoch} train accuracy is {epoch_accuracy*100} %")

    #TEST
    with torch.no_grad():

        model_resnet = model_resnet.eval()

        epoch_test_losses = []
        epoch_test_true_positives = 0

        for batch_idx, batch in enumerate(data_loader_test):

            x = batch['image'].to(DEVICE)


            y = batch['cls'].to(DEVICE)

            y_one_hot = torch.zeros(x.shape[0], NUM_CLASSES).to(DEVICE)
            y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

            labels = batch['class_name']

            y_prim = model_resnet.forward(x)
            loss = torch.sum(-y_one_hot * torch.log(y_prim))
            epoch_test_losses.append(loss.detach().to('cpu').numpy())

            epoch_test_true_positives += torch.sum(y_prim.argmax(dim=1) == y)

            if batch_idx == 0:
                plot_results(x.detach().to('cpu').numpy(),
                             y_prim.argmax(dim=1).detach().to('cpu').numpy(),
                             y.detach().to('cpu').numpy(),
                             image_name=f"epoch_{epoch}")


        epoch_test_accuracy = float(epoch_test_true_positives) / float(NUM_TEST_SAMPLES)
        print(f"true_positives {epoch_test_true_positives} from {NUM_TEST_SAMPLES} samples")
        print(f"Epoch {epoch} test mean loss is {np.mean(epoch_test_losses)}")
        print(f"Epoch {epoch} test accuracy is {epoch_test_accuracy * 100}")

        torch.save(model_resnet, last_model_path)
        if epoch_test_accuracy > best_test_accuracy:
            best_test_accuracy = epoch_test_accuracy
            torch.save(model_resnet, best_model_path)