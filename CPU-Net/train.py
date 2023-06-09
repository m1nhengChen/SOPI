import os
import time
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
import numpy as np
import projection
import cv2
# first train run this code
import Loss
import data_loader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Device configuration
import regnet

os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# WORK_DIR = './train_data/train_data_rol'
NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

MODEL_PATH = './model'
MODEL_NAME = 'Inception_v3.pth'

# Create model

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

dataset = np.fromfile("./data/final_pro.raw", dtype="float32").reshape((112500, 64, 64))
labelset = np.loadtxt("./label_test.txt").astype("int64")
ctset = np.fromfile("./data/final.raw", dtype="float32").reshape((45, 250, 128, 128))
dataset = torch.from_numpy(dataset)
labelset = torch.from_numpy(labelset)

torch.manual_seed(3)
# train_data, vali_data = torch.utils.data.random_split(dataset, [450000, 111152])
train_data, vali_data = torch.utils.data.random_split(dataset, [90000, 22500])
torch.manual_seed(3)
# train_lable, vali_lable = torch.utils.data.random_split(labelset, [450000, 111152])
train_lable, vali_lable = torch.utils.data.random_split(labelset, [90000, 22500])

dataset = data_loader.Dataset(train_data, train_lable)
va_dataset = data_loader.Dataset(vali_data, vali_lable)

DL = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
va_DL = torch.utils.data.DataLoader(va_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

print("dataloading is over！")


def main():
    print(f"Train numbers: 90000")

    # writer = SummaryWriter("./logs")  # 存放log文件的目录

    # first train run this line
    model = regnet.RegNet().to(device)
    # model = torch.nn.DataParallel(model)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1], output_device=[0, 1])
    print(model)

    # cast = torch.nn.CrossEntropyLoss().to(device)
    cast = Loss.NCC().to(device)

    # Optimization
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=1e-4)
    step = 1
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # cal one epoch time
        start = time.time()

        for images, label_set in DL:
            images = images.reshape((BATCH_SIZE, 1, 64, 64)).to(device)
            ct_train_data = np.zeros((BATCH_SIZE, 250, 128, 128), dtype="float32")
            for index in range(BATCH_SIZE):
                ct_train_data[index] = ctset[label_set[index]]
            ct_train_data = torch.from_numpy(ct_train_data.reshape((BATCH_SIZE, 1, 250, 128, 128))).to(device)
            # Forward pass
            outputs = model(ct_train_data, images)
            image_project = projection.project(outputs, label_set[index])
            loss = cast(image_project, images.reshape(64, 64))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{step * BATCH_SIZE}/{NUM_EPOCHS * 90000}], "
                  f"Loss: {loss.item():.8f}.")
            step += 1
            # writer.add_scalar("batch_loss", loss.item(), step)
        ###################
        ''' va_total_loss = 0
        for va_images, va_label_set in va_DL:
            va_images = va_images.reshape((BATCH_SIZE, 1, 64, 64)).to(device)

            ct_vali_data = np.zeros((BATCH_SIZE, 250, 128, 128), dtype="float32")
            for index in range(BATCH_SIZE):
                ct_vali_data[index] = ctset[va_label_set[index][0]]

            va_label_set = va_label_set[:, 1:4].to(device)
            ct_vali_data = torch.from_numpy(ct_vali_data.reshape((BATCH_SIZE, 1, 250, 128, 128))).to(device)

            # print prediction
            with torch.no_grad():
                vali_outputs = model(ct_vali_data, va_images)

            va_total_loss += cast(vali_outputs, va_label_set).item()
            print(vali_outputs)
            print(va_label_set)
        # equal prediction and acc
        print(va_total_loss)
        writer.add_scalar("validation", va_total_loss, step)'''
        #####################################################

        # cal train one epoch time
        end = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}], "
              f"time: {end - start} sec!")

        # Save the model checkpoint
        torch.save(model, MODEL_PATH + '/' + str(epoch) + "_" + MODEL_NAME)
    print(f"Model save to {MODEL_PATH + '/' + MODEL_NAME}.")
    # writer.close()


if __name__ == '__main__':
    main()
