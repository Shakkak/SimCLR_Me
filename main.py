import os
import numpy as np
import torch
import torchvision
import argparse
from PIL import Image

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
#Define the split ratio
split_ratio = 0.6

#Dataset function called
class Birddataset(Dataset):
    def __init__(self, image_dir, allowed_classes, transform=None, dataset_type = None):
        """
        Args:
            image_dir (str): Directory path containing input images.
            mask_dir (str): Directory path containing corresponding segmentation masks.
            transform (callable): Optional transformation to be applied to both the image and the mask. . Use ToTensorV2()
            dataset_type (str, optional): Type of dataset, e.g., 'Train' or 'Test'. Defaults to 'Train'.
        """
        # Initialize paths and transformation
        self.allowed_classes=allowed_classes
        self.image_dir = image_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.classes = [item for item in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, item))]
        self.samples=[]
        for class_name in self.classes:
                if class_name in allowed_classes:

                    self.images = os.listdir(os.path.join(self.image_dir, class_name))
                    for img in self.images:
                        self.samples.append([img,class_name])

        random.seed(87)
        random.shuffle(self.samples)

        # print(self.samples)

        if dataset_type == 'Train':
            self.images = self.samples[:int(len(self.samples)*split_ratio)]
        elif dataset_type == 'Test':
            self.images = self.samples[int(len(self.samples)*split_ratio):]
        else:
            self.images = self.samples

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of image-mask pairs in the designated dataset split.
        """
        # Return the length of the dataset (number of images)
        return len(self.images)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index of the image-mask pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image and its corresponding one-hot encoded mask.
                - image (torch.Tensor): Transformed image tensor.
                - onehot_mask (torch.Tensor): One-hot encoded mask tensor for segmentation.
        """
        # Load the image and mask
        image_path = os.path.join(self.image_dir,self.images[index][1],self.images[index][0])



        # Load image and mask as grayscale
        image = Image.open(image_path)
        if self.transform:
            transformed = self.transform(image)

        class_id = self.allowed_classes.index(self.images[index][1])

        return transformed, class_id




def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # if args.dataset == "STL10":
    #     train_dataset = torchvision.datasets.STL10(
    #         args.dataset_dir,
    #         split="unlabeled",
    #         download=True,
    #         transform=TransformsSimCLR(size=args.image_size),
    #     )
    if args.dataset == "ME":
        # print()
        train_dataset = Birddataset(
            image_dir="/kaggle/working//Noisy_birds",
            allowed_classes=["unlabeled"],
            transform=TransformsSimCLR(size=128),
            )
    # elif args.dataset == "CIFAR10":
    #     train_dataset = torchvision.datasets.CIFAR10(
    #         args.dataset_dir,
    #         download=True,
    #         transform=TransformsSimCLR(size=args.image_size),
    #     )
    # else:
    #     raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
