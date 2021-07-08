import torch
import torchvision
from torch import optim
import os
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn
import utils.train_utils as tu
from  utils.dataset import create_dataloader, create_test_dataloader
import cv2 as cv
import numpy as np

if __name__ == "__main__":

    #Create directory for storing results
    output_dir = os.path.join("./", "output")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    #some cfgs, some cfg will be used in the future
    #TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    ckpt = None
    num_epochs = 10
    ckpt_save_interval = 2

    #Prepare data
    print("Prepare data")
    data_path = "./data"
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    trainloader= create_dataloader(data_path, batch_size=4, num_workers=2)

    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # images = images / 2 + 0.5
    # images = np.transpose(torchvision.utils.make_grid(images).numpy(), (1 ,2, 0))
    # cv.imshow("Sample image", images)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #Prepare model
    print("Prepare model")
    from model.cnn import CNN
    model = CNN()
    model.cuda()

    #Prepare training
    print("Prepare training")
    optimizer = optim.SGD(model.parameters(), lr=tu.lr_scheduler(), momentum=0.9)

    #Define starting iteration/epochs.
    #Will use checkpoints in the future when running on clusters
    if ckpt is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=ckpt)
    elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    else:
        starting_iteration, starting_epoch = 0, 0

    #Training
    print("Start training")
    trainer = Trainer(model=model,
                      model_fn=model_fn,
                      optimizer=optimizer,
                      ckpt_dir=ckpt_dir)

    trainer.train(num_epochs=num_epochs,
                  train_loader=trainloader,
                  ckpt_save_interval=ckpt_save_interval,
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    #Testing
    print("Start testing")
    testloader = create_test_dataloader(data_path, batch_size=4, num_workers=2)
    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e%d" % (trainer._epoch + 1)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels.cpu()).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    #Finalizing
    print("Finished")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future