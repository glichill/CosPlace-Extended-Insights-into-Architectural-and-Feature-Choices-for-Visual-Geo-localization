import torch.nn as nn
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import matplotlib.pyplot as plt

import itertools
import test
import util
import parser
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from cosplace_model import mixVPRcosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.targetdomain_dataset import TargetDataset

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")


#### Model
if args.aggregation_type=='MixVPR':
  resnet_backbone = mixVPRcosplace_network.ResNet(model_name="resnet18", pretrained=True, layers_to_freeze=2, layers_to_crop=[4])
  model = mixVPRcosplace_network.GeoLocalizationNet(resnet_backbone, resnet_backbone.out_channels)
else:
  model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim)



# Initialize the domain discriminator and its optimizer if domain adaptation is enabled
if args.enable_domain_adaptation:
    from domain_discriminator import DomainDiscriminator
    feature_dim = args.fc_output_dim  # Assuming this is the feature dimensionality
    domain_discriminator = DomainDiscriminator(input_dim=feature_dim).to(args.device)
    optimizer_discriminator = torch.optim.Adam(domain_discriminator.parameters(), lr=0.0001)
    adversarial_loss_fn = nn.BCELoss()

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

from kneed import KneeLocator

def find_elbow(ssds):
    kn = KneeLocator(range(1, len(ssds) + 1), ssds, curve='convex', direction='decreasing')
    return kn.elbow - 1  # -1 because the index starts from 0


#### Optimizer
criterion = torch.nn.CrossEntropyLoss()

#Adam Optimizer
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#AdamW Optimizer
#model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

#ASGD Optimizer
#model_optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.wd, alpha=args.al)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

### NetVLAD inizialization
if args.use_netvlad:
    # Access the NetVLAD module within the nn.Sequential object
    netvlad_module = model.aggregation[0]
    
    if args.choose_num_cluster:
       # Range of K values to test
       k_values = list(range(5, 31, 5))  # 5, 10, 15, ..., 30
    
       # Call the modified initialize_netvlad_layer method on the NetVLAD module
       ssds, k_values = netvlad_module.initialize_netvlad_layer_with_elbow(args, groups[0], model.backbone, k_values)
    
       print(f"ssds before find_elbow: {ssds}")

       # Find the knee of the elbow plot
       idx_of_knee = find_elbow(ssds)
       best_k = k_values[idx_of_knee]
    
       print(f"The best number of clusters according to the elbow method is {best_k}")
    
       # Plot the SSDs for each k
       plt.figure()
       plt.plot(k_values, ssds, 'bx-')
       plt.xlabel('k')
       plt.ylabel('Sum of Squared Distances')
       plt.title('Elbow Method for Optimal k')
       plt.annotate(f'Best k={best_k}', xy=(best_k, ssds[idx_of_knee]), xytext=(best_k, ssds[idx_of_knee]*1.1),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12)
    
       # Save the plot
       plt.savefig('elbow_plot.png')
    
       # Re-initialize the NetVLAD layer with the best number of clusters
       netvlad_module.clusters_num = best_k
    else:
       netvlad_module.clusters_num = args.netvlad_clusters  
    print(model.aggregation)
   
    netvlad_module.initialize_netvlad_layer(args, groups[0], model.backbone)
    


# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

#classifiers_optimizers = [torch.optim.AdamW(classifier.parameters(), lr=args.classifiers_lr, weight_decay=args.classifiers_wd) for classifier in classifiers]

#classifiers_optimizers = [torch.optim.ASGD(classifier.parameters(), lr=args.classifiers_lr, lambd=args.classifiers_lambd, alpha=args.classifiers_al) for classifier in classifiers]


logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
test_ds_tokyo = TestDataset("/content/drive/MyDrive/tokyo_xs/test/",positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Test set Tokyo: {test_ds_tokyo}")                      
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                          scale=[1-args.random_resized_crop, 1]),
            augmentations.DeviceAgnosticRandomHorizontalFlip(p=0.5), 
                                                                     
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)

    if args.enable_domain_adaptation:
            target_dataset = TargetDataset(args, args.target_dataset_folder)
            target_loader = commons.InfiniteDataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=multiprocessing.cpu_count(), pin_memory=True)
    model = model.train()
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _ = next(dataloader_iterator)
        images, targets = images.to(args.device), targets.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        if not args.use_amp16:
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            epoch_losses = np.append(epoch_losses, loss.item())
            del output, images
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            epoch_losses = np.append(epoch_losses, loss.item())
            del output, images

        # Domain adaptation: adversarial training
        if args.enable_domain_adaptation:
            target_images, _ = next(iter(target_loader))  # Assuming the target_loader provides images without labels
            target_images = target_images.to(args.device)
            target_descriptors = model(target_images)

            # Compute domain labels
            source_domain_labels = torch.ones(descriptors.size(0), 1).to(args.device)
            target_domain_labels = torch.zeros(target_descriptors.size(0), 1).to(args.device)

            # Compute adversarial loss
            source_domain_loss = adversarial_loss_fn(domain_discriminator(descriptors), source_domain_labels)
            target_domain_loss = adversarial_loss_fn(domain_discriminator(target_descriptors), target_domain_labels)
            total_domain_loss = source_domain_loss + target_domain_loss

            # Update domain discriminator
            optimizer_discriminator.zero_grad()
            total_domain_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            # Combine the main task loss with the adversarial loss
            loss += args.lambda_adversarial * total_domain_loss

        # Backpropagation
        if not args.use_amp16:
            loss.backward()
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            scaler.scale(loss).backward()
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
            
        del loss

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    
    #### Evaluation
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

if args.enable_test_tokyo:
   logging.info(f"Now testing on the test set: {test_ds_tokyo}")
   recalls_tokyo, recalls_str_tokyo = test.test(args, test_ds_tokyo, model, args.num_preds_to_save)
   logging.info(f"{test_ds_tokyo}: {recalls_str_tokyo}")


logging.info("Experiment finished (without any errors)")
