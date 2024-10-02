from hg import *
from optimizer import *
from loader import *
import torch
import sys
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm
from utils import *
from MAID import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Convex models')))
from TV import *
from FoE import *
from ICNN import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

problem = "denoising" # "semi-blind deconvolution" "deconvolution" "denoising"
training_Mode = "MAID" #"ISGD" "MAID" "IAdam" "IASGD"
opt_alg = 'MAID'

channels = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
regularizer = FoE(channels, 10, 7, learnable_smoothing = True, learnable_weights = True).to(device)
regularizer.init_weights()
# regularizer.load_state_dict(torch.load(os.getcwd()+ "/Logs/regularizer_0.1_0.0001_MAID.pt"))
# regularizer = TV(channels, learnable_smoothing = True).to(device)
# regularizer = ICNN()
# regularizer.initialize_weights()
# regularizer.zero_clip_weights()
# regularizer = regularizer.to(device)


class Lower_Level(nn.Module):
    def __init__(self, measurement, regularizer, forward_operator = lambda x: x):
        super(Lower_Level, self).__init__()
        self.regularizer = regularizer
        self.measurement = measurement
        self.A = forward_operator
    def data_fidelity(self, x):
        # return 0.5 *torch.mean(torch.norm(self.A(x) - self.measurement, dim = (2, 3))**2)
        return 0.5 * torch.norm(self.A(x) - self.measurement)**2/ (self.measurement.shape[0])
        # return torch.nn.MSELoss()(self.A(x), self.measurement)
    def forward(self, x):
        return self.data_fidelity(x) + self.regularizer(x)
class Upper_Level(nn.Module):
    def __init__(self, x):
        super(Upper_Level, self).__init__()
        self.x = x
    def forward(self, x):
        return  torch.norm(x - self.x)**2/ (self.x.shape[0] * self.x.shape[1])
        # return torch.nn.MSELoss()(x, self.x)

    # def forward(self, x):
    #     return torch.nn.MSELoss()(x, self.x)

# Load data
batch_size = 64
noise_level = 25
train_size = 25
img_size_x, img_size_y = 96, 96
budget = 3e5

if problem == "semi-blind deconvolution" or problem == "deconvolution":
    blur_kernel_original = gaussian_kernel(5, 2, 1)
    kernel_4d = blur_kernel_original.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel_4d.repeat(3, 1, 1, 1)
    blur_original = nn.Conv2d(channels, channels, 5, padding = 2, bias = False, groups= channels)
    blur_original.weight.data = kernel_4d
    class blur (nn.Module):
        def __init__(self, kernel):
            super(blur, self).__init__()
            self.kernel = kernel
            self.conv = nn.Conv2d(channels, channels, 5, padding = 2, bias = False, groups= channels, padding_mode= 'reflect')
            self.conv.weight.requires_grad = False
        def forward(self, x):
            if x.device != self.kernel.device:
                self.kernel = self.kernel.to(x.device)
                self.conv = self.conv.to(x.device)
            self.conv.weight.data = self.kernel
            return self.conv(x)
    blur_obj = blur(kernel_4d)
    operator = blur_obj
else:
    operator = lambda x: x

if channels == 1:
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((img_size_x,img_size_y)), transforms.ToTensor()])
else:
    #CenterCrop or Resize
    transform = transforms.Compose([transforms.Resize((img_size_x, img_size_y)), transforms.ToTensor()])
dataset = CustomDataset("STL10", pytorch_dataset=True, transform=transform)
# test_size = 100
# test_dataset = Subset(dataset, range(train_size, train_size + test_size))
# dataset = CustomDataset("BSDS300", pytorch_dataset=False, transform=transform)
dataset = Subset(dataset, range(train_size))
if training_Mode == "ISGD" or training_Mode == "IAdam" or training_Mode == "IASGD" or training_Mode == "SF_SGD":
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False, drop_last=True)
    noisy_dataset = NoisyDataset(dataset, noise_level=noise_level, forward_operator = operator)
    noisy_loader = DataLoader(noisy_dataset, batch_size=batch_size, shuffle = False)
    temp_init = torch.empty(batch_size, channels, img_size_x, img_size_y)
elif training_Mode == "MAID" or training_Mode == "IGD":
    dataloader = DataLoader(dataset, batch_size=train_size, shuffle = False, drop_last=True)
    noisy_dataset = NoisyDataset(dataset, noise_level=noise_level, forward_operator = operator)
    noisy_loader = DataLoader(noisy_dataset, batch_size=train_size, shuffle = False, drop_last=True)
    temp_init = torch.empty(train_size, channels, img_size_x, img_size_y)
    
init_loader = []
init_test = []
for data in (noisy_loader):
    # init_loader.append(torch.zeros_like(data[0]))
    init_loader.append(data[0])
# test data 
test_size = 50
test_dataset = CustomDataset("STL10", pytorch_dataset=True, transform=transform, type= "test")
test_dataset = Subset(test_dataset, range(test_size))
test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle = False)
noisy_dataset_test = NoisyDataset(test_dataset, noise_level=noise_level, forward_operator = operator)
noisy_loader_test = DataLoader(noisy_dataset_test, batch_size=test_size, shuffle = False)
for data in (noisy_loader_test):
    # init_test.append(torch.zeros_like(data[0]))
    init_test.append(data[0])
# Single test image   
    # test_image = Image.open(os.getcwd() + "/nessy.jpg").convert('RGB')
    # test_image = transform(test_image).unsqueeze(0)
    # test_image = test_image.to(device)
    # measurement = operator(test_image) + noise_level/255 * torch.randn_like(test_image)
    # plt.imshow(measurement[0].cpu().detach().permute(1, 2, 0).numpy())
    # plt.title("PSNR: " + str(psnr(test_image[0], measurement[0]).mean().item()))
    # plt.show()
    # init_test = torch.zeros_like(test_image)


if problem == "semi-blind deconvolution":
    blur_kernel = gaussian_kernel(5, 5 // 2, 0.1)
    kernel_4d = blur_kernel.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel_4d.repeat(3, 1, 1, 1) 
    blur_conv = nn.Conv2d(channels, channels, 5, padding = 2, bias = False, groups= channels)
    blur_conv = blur_conv.to(device)
    blur_conv.weight.data = kernel_4d.to(device)
    operator = blur_conv
    regularizer.forward_operator = operator
lower_level = Lower_Level(noisy_loader, regularizer, forward_operator= operator)#.to(device)
upper_level = Upper_Level(temp_init).to(device)
x_init = temp_init.to(device)
hypergrad = Hypergrad_Calculator(lower_level, upper_level, x_init = x_init, verbose= True)
test_hypergrad = Hypergrad_Calculator(lower_level, upper_level, x_init = x_init, verbose= False)
print ("number of parameters: ", sum(p.numel() for p in hypergrad.parameters()), "number of mini-batches: ", len(dataloader))

number_epochs = 1000
number_batches = len(dataloader)
eps0 = 1e1
p = 0.0
q = 0.0  # should be greater than 0.5 and less than 1
alpha = 1e-4
setting= "poly-poly" # "constant" "poly-log" "poly-poly" "log-log"


# optimizer = Bilevel_Optimizer(hypergrad = hypergrad, verbose = True, optimizer= 'SGD', lr = alpha).to(device)
# logs_dict = {"loss_batch": [], "loss_epoch": [], "eps": [], "step": [], "lower_iter": [], "cg_iter": [],
#              "psnr_batch" : [], "psnr_epoch" : [], "setting": setting, "p": p, "q": q, "train_mode": training_Mode}
# # initialising the logs
# for i, (data, noisy_data, init_data) in enumerate(zip(dataloader, noisy_loader, init_loader)):
#     hypergrad = data_update(hypergrad, init_data.to(device), noisy_data[0].to(device), data[0].to(device))
#     logs_dict["loss_batch"].append(hypergrad.upper_level_obj(hypergrad.x_init).item())
# logs_dict["loss_epoch"].append(torch.mean(torch.tensor(logs_dict["loss_batch"])))
# logs_dict["psnr_epoch"].append(psnr(data[0], init_loader[i].cpu()).mean().item())
# logs_dict["eps"].append(eps0)
# logs_dict["step"].append(alpha)
# logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
# logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
# loss_batch = []
# psnr_batch = []
# directory = os.getcwd()
# if training_Mode == "ISGD" or training_Mode == "IAdam" or training_Mode == "IASGD" or training_Mode == "SF_SGD":
#     if training_Mode == "SF_SGD":
#         hypergrad.train()
#         optimizer.optimizer.train()
#     for epoch in range(number_epochs):
#         progress_bar = tqdm(enumerate(zip(dataloader, noisy_loader, init_loader)), total = len(dataloader), desc = f"Epoch {epoch}/{number_epochs}")
#         # init_loader = DataLoader(ReplaceableDataset(init_loader.dataset), batch_size=32, shuffle = False)
#         for i, (data, noisy_data, init_data) in progress_bar:
#             hypergrad = data_update(hypergrad, init_data.to(device), noisy_data[0].to(device), data[0].to(device))
#             optimizer.hypergrad = hypergrad
#             optimizer.step()
#             # hypergrad.lower_level_obj.regularizer.zero_mean()
#             # updating the initalisation
#             # init_loader[i] = hypergrad.FISTA(hypergrad.x_init, hypergrad.lower_tol, hypergrad.max_lower_iter)
#             init_loader[i] = hypergrad.x_init.detach().cpu().clamp(0, 1)
#             if setting == "poly-poly":
#                 # print(total_iter(i, epoch, number_batches) + 1)
#                 optimizer.lr = alpha * (total_iter(i, epoch, number_batches) + 1)**(-q)
#                 hypergrad.lower_tol = eps0 * (total_iter(i, epoch, number_batches) + 1)**(-p)
#             elif setting == "poly-log":
#                 optimizer.lr = alpha * (total_iter(i, epoch, number_batches) + 1)**(-q)
#                 if i == 0:
#                     hypergrad.lower_tol = eps0
#                 else:
#                     hypergrad.lower_tol = eps0 * (torch.log(total_iter(i, epoch, number_batches) + 1))**(-p)
#             elif setting == "log-log":
#                 if i == 0:
#                     optimizer.lr = alpha
#                     hypergrad.lower_tol = eps0
#                 else:
#                     optimizer.lr = alpha * (torch.log(total_iter(i, epoch, number_batches) + 1))**(-q)
#                     hypergrad.lower_tol = eps0 * (torch.log(total_iter(i, epoch, number_batches) + 1))**(-p) 
#             elif setting == "constant":
#                 optimizer.lr = alpha
#                 hypergrad.lower_tol = eps0       
#             print("PSNR: ", psnr(data[0][0].unsqueeze(0), init_loader[i][0].unsqueeze(0).cpu()).item())
#             # plt.imshow(hypergrad.x_init[0].cpu().detach().permute(1, 2, 0).numpy())
#             # plt.show()
#             # logging
#             if hypergrad.upper_level_obj(hypergrad.x_init) is None or torch.isnan(hypergrad.upper_level_obj(hypergrad.x_init)) or torch.isinf(hypergrad.upper_level_obj(hypergrad.x_init)):
#                 print("Loss is nan or inf")
#                 break
#             logs_dict["loss_batch"].append(hypergrad.upper_level_obj(hypergrad.x_init).item())
#             loss_batch.append(logs_dict["loss_batch"][-1])
#             psnr_batch.append(psnr(data[0], init_loader[i].cpu()).mean().item())
#             logs_dict["eps"].append(hypergrad.lower_tol)
#             logs_dict["step"].append(optimizer.lr)
#             logs_dict["psnr_batch"].append(psnr_batch[-1])
#             if hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"] > budget:
#                 print("Budget exhausted")
#                 break
            
#         # test
#         for i, (data, noisy_data) in enumerate(zip(test_loader, noisy_loader_test)):
#             init_test = torch.zeros_like(data[0])
#             hypergrad = data_update(hypergrad, init_test.to(device), noisy_data[0].to(device), data[0].to(device))
#             out = hypergrad.FISTA(hypergrad.x_init, 1e-3, hypergrad.max_lower_iter)
#             print("PSNR: ", psnr(data[0][0], out[0].cpu()).mean().item())
#         logs_dict["psnr_epoch"].append(torch.mean(torch.tensor(logs_dict["psnr_batch"])))
#         logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
#         logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])            
#         # plt.imshow(hypergrad.x_init[0].cpu().detach().numpy().squeeze(), cmap = "gray")
#         # plt.show()
#         logs_dict["loss_epoch"].append(torch.mean(torch.tensor(loss_batch)))
#         loss_batch = []
#         psnr_batch = []
#         torch.save(logs_dict, directory + f'/Logs/logs_dict_{eps0}_{alpha}_{setting}_{p}_{q}.pt')
#         if setting == "constant":
#             torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), directory + f'/Logs/regularizer_{eps0}_{alpha}_{setting}_{opt_alg}.pt')
#         else:
#             torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), directory + f'/Logs/regularizer_{eps0}_{alpha}_{setting}_{p}_{q}_{opt_alg}.pt')
        
#         # early stopping due to maximum budget
#         if hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"] > budget:
#             print("Budget exhausted")
#             break
# elif training_Mode == "MAID" or training_Mode == "IGD":
#     hypergrad.verbose = False
#     hypergrad.warm_start = False
#     optimizer = MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr = alpha, hypergrad_module = hypergrad, eps = eps0)
#     if training_Mode == "IGD":
#         optimizer.fixed_eps = True
#         optimizer.fixed_lr = True
#     # MAID full batch training
#     for epoch in range(number_epochs):
#         progress_bar = tqdm(enumerate(zip(dataloader, noisy_loader, init_loader)), total = len(dataloader), desc = f"Epoch {epoch}/{number_epochs}")
#         for i, (data, noisy_data, init_data) in progress_bar:
#             hypergrad = data_update(hypergrad, init_data.to(device), noisy_data[0].to(device), data[0].to(device))
#             optimizer.hypergrad = hypergrad
#             optimizer.zero_grad()
#             optimizer.hypergrad.hypergrad()
#             hypergrad.lower_skip = True
#             loss_val, init_loader[i] = optimizer.step()
#             hypergrad.x_init = init_loader[i]
#             # hypergrad.lower_level_obj.regularizer.zero_mean()
#             # updating the initalisation
#             # if optimizer.state['successful']:
#             #     hypergrad.x_init = init_loader[i]
#             # else:
#             #     init_loader[i] = optimizer.hypergrad.x_init.detach().cpu().clamp(0, 1)
#             print("PSNR: ", psnr(data[0][0].unsqueeze(0), init_loader[i][0].unsqueeze(0).cpu()).item(), "loss: ", loss_val.item())
#             hypergrad.x_init.detach().cpu()
#             init_data.detach().cpu()
#             noisy_data[0].detach().cpu()
#             data[0].detach().cpu()
            
#             if epoch % 99 == 0:
#                 plt.imshow(hypergrad.x_init[10].cpu().detach().permute(1, 2, 0).numpy())
#                 plt.title("PSNR: " + str(psnr(data[0][10], hypergrad.x_init[10].cpu()).mean().item()))
#                 plt.show()
#                 # test image
                
#                 # hypergrad.lower_level_obj.measurement = measurement
#                 # out = hypergrad.FISTA(init_test, hypergrad.lower_tol, 200)
#                 # plt.imshow(out[0].cpu().detach().permute(1, 2, 0).numpy())
#                 # plt.title("PSNR: " + str(psnr(test_image[0], out[0]).mean().item()))
#                 # plt.show()
#                 # measurement = out
#                 # plot kernel
#                 # plt.imshow(hypergrad.lower_level_obj.A.kernel[0].cpu().detach().numpy().squeeze())
#                 # plt.colorbar()
#                 # plt.show()
#             # logging
#             if optimizer.state['successful']:
#                 logs_dict["loss_batch"].append(loss_val.item())
#                 loss_batch.append(logs_dict["loss_batch"][-1])
#                 logs_dict["eps"].append(hypergrad.lower_tol)
#                 logs_dict["step"].append(optimizer.lr)
#                 logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
#                 logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
#                 logs_dict["psnr_epoch"].append(psnr(data[0], init_loader[i].cpu()).mean().item())
#         logs_dict["loss_epoch"] = logs_dict["loss_batch"]
#         torch.save(logs_dict, directory + f'/Logs/logs_dict_{eps0}_{alpha}_{opt_alg}_2.pt')
#         torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), directory + f'/Logs/regularizer_{eps0}_{alpha}_{opt_alg}_2.pt')
#         # early stopping due to maximum budget
#         print("cost: ", hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"])
#         if hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"] > budget:
#             print("Budget exhausted")
#             if not optimizer.state['successful']:
#                 logs_dict["loss_batch"].append(loss_val.item())
#                 loss_batch.append(logs_dict["loss_batch"][-1])
#                 logs_dict["eps"].append(hypergrad.lower_tol)
#                 logs_dict["step"].append(optimizer.lr)
#                 logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
#                 logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
#                 logs_dict["psnr_epoch"].append(psnr(data[0], init_loader[i].cpu()).mean().item())
#             break

def initialize_optimizer(hypergrad, device, alpha, training_mode, optimizer_type='SGD'):
    if training_mode in ["MAID", "IGD"]:
        hypergrad.verbose = False
        hypergrad.warm_start = False
        return MAID(hypergrad.lower_level_obj.regularizer.parameters(), lr=alpha, hypergrad_module=hypergrad, eps=eps0)
    return Bilevel_Optimizer(hypergrad=hypergrad, verbose=True, optimizer=optimizer_type, lr=alpha).to(device)

def initialize_logs(setting, p, q, training_mode):
    return {
        "loss_batch": [], "loss_epoch": [], "eps": [], "step": [],
        "lower_iter": [], "cg_iter": [], "psnr_batch": [], "psnr_epoch": [],
        "setting": setting, "p": p, "q": q, "train_mode": training_mode, 
        "loss_test": [], "psnr_test": []
    }
def initialize_log_values(dataloader, noisy_loader, init_loader, hypergrad, logs_dict, device, eps0, alpha, psnr_fn):
    # Initialize logs by processing one pass through the data
    for i, (data, noisy_data, init_data) in enumerate(zip(dataloader, noisy_loader, init_loader)):
        # Update hypergrad with data and save the loss
        hypergrad = data_update(hypergrad, init_data.to(device), noisy_data[0].to(device), data[0].to(device))
        logs_dict["loss_batch"].append(hypergrad.upper_level_obj(hypergrad.x_init).item())
        logs_dict["psnr_batch"].append(psnr_fn(data[0], init_loader[i].cpu()).mean().item())
        

    # Calculate mean loss for the epoch and append it to loss_epoch
    logs_dict["loss_epoch"].append(torch.mean(torch.tensor(logs_dict["loss_batch"])))

    # Log constant values for `eps`, `step`, `lower_iter`, `cg_iter`
    logs_dict["eps"].append(eps0)
    logs_dict["step"].append(alpha)
    logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
    logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
    # Log the PSNR for the batch
    logs_dict["psnr_epoch"].append(torch.mean(torch.tensor(logs_dict["psnr_batch"])))

    return logs_dict
def update_data(hypergrad, init_data, noisy_data, data, device):
    return data_update(hypergrad, init_data.to(device), noisy_data.to(device), data.to(device))

def update_lr_tol(setting, optimizer, hypergrad, total_iter, i, epoch, num_batch, alpha, eps0, p, q):
    if setting == "poly-poly":
        optimizer.lr = alpha * (total_iter(i, epoch, num_batch) + 1) ** (-q)
        hypergrad.lower_tol = eps0 * (total_iter(i, epoch, num_batch) + 1) ** (-p)
    elif setting == "poly-log":
        optimizer.lr = alpha * (total_iter(i, epoch,num_batch) + 1) ** (-q)
        hypergrad.lower_tol = eps0 if i == 0 else eps0 * (torch.log(total_iter(i, epoch,num_batch) + 1)) ** (-p)
    elif setting == "log-log":
        optimizer.lr = alpha * (torch.log(total_iter(i, epoch,num_batch) + 1)) ** (-q)
        hypergrad.lower_tol = eps0 * (torch.log(total_iter(i, epoch,num_batch) + 1)) ** (-p)
    elif setting == "constant":
        optimizer.lr = alpha
        hypergrad.lower_tol = eps0

def log_metrics(logs_dict, hypergrad, optimizer, data, psnr_fn, init_loader, loss_batch, psnr_batch, i, epoch, directory):
    logs_dict["loss_batch"].append(hypergrad.upper_level_obj(hypergrad.x_init).item())
    loss_batch.append(logs_dict["loss_batch"][-1])
    psnr_batch.append(psnr_fn(data, init_loader[i].cpu()).mean().item())
    logs_dict["eps"].append(hypergrad.lower_tol)
    logs_dict["step"].append(optimizer.lr)
    logs_dict["psnr_batch"].append(psnr_batch[-1])
    save_logs(logs_dict, hypergrad, directory, eps0, alpha, setting, p, q, opt_alg, epoch)

def save_logs(logs_dict, hypergrad, directory, eps0, alpha, setting, p, q, opt_alg, epoch):
    # Define file paths based on the setting
    log_file = f'{directory}/Logs/logs_dict_{eps0}_{alpha}_{setting}_{p}_{q}.pt'
    
    if setting == "constant":
        regularizer_file = f'{directory}/Logs/regularizer_{eps0}_{alpha}_{setting}_{opt_alg}.pt'
    else:
        regularizer_file = f'{directory}/Logs/regularizer_{eps0}_{alpha}_{setting}_{p}_{q}_{opt_alg}.pt'
    # Save logs and regularizer state
    torch.save(logs_dict, log_file)
    torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), regularizer_file)
    
def train_epoch(epoch, dataloader, noisy_loader, init_loader, hypergrad, optimizer, logs_dict, number_batches, psnr_fn, alpha, eps0, setting, p, q, total_iter, budget, device):
    progress_bar = tqdm(enumerate(zip(dataloader, noisy_loader, init_loader)), total=len(dataloader), desc=f"Epoch {epoch}")
    loss_batch, psnr_batch = [], []
    directory = os.getcwd()

    for i, (data, noisy_data, init_data) in progress_bar:
        hypergrad = update_data(hypergrad, init_data, noisy_data[0], data[0], device)
        optimizer.hypergrad = hypergrad
        optimizer.step()
        # hypergrad.lower_level_obj.regularizer.zero_mean()
        init_loader[i] = hypergrad.x_init.detach().cpu().clamp(0, 1)
        update_lr_tol(setting, optimizer, hypergrad, total_iter, i, epoch, number_batches, alpha, eps0, p, q)

        if hypergrad.upper_level_obj(hypergrad.x_init) is None or torch.isnan(hypergrad.upper_level_obj(hypergrad.x_init)):
            print("Loss is nan or inf")
            break

        log_metrics(logs_dict, hypergrad, optimizer, data[0], psnr_fn, init_loader, loss_batch, psnr_batch, i, epoch, directory)
        if hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"] > budget:
            logs_dict["psnr_epoch"].append(torch.mean(torch.tensor(psnr_batch)))
            logs_dict["loss_epoch"].append(torch.mean(torch.tensor(loss_batch)))
            logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
            logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
            return False
    logs_dict["psnr_epoch"].append(torch.mean(torch.tensor(psnr_batch)))
    logs_dict["loss_epoch"].append(torch.mean(torch.tensor(loss_batch)))
    logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
    logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
    return True

def train_igd_maid_epoch(epoch, dataloader, noisy_loader, init_loader, hypergrad, optimizer, logs_dict, psnr_fn, device, number_epochs, budget):
    progress_bar = tqdm(enumerate(zip(dataloader, noisy_loader, init_loader)), total=len(dataloader), desc=f"Epoch {epoch}/{number_epochs}")
    loss_batch, psnr_batch = [], []
    directory = os.getcwd()

    for i, (data, noisy_data, init_data) in progress_bar:
        # Update hypergrad with new data
        hypergrad = update_data(hypergrad, init_data, noisy_data[0], data[0], device)
        optimizer.hypergrad = hypergrad

        # Optimizer step: zero gradient, compute hypergrad and step
        optimizer.zero_grad()
        optimizer.hypergrad.hypergrad()
        hypergrad.lower_skip = True

        # Perform optimization step
        loss_val, init_loader[i] = optimizer.step()
        hypergrad.x_init = init_loader[i]
        # hypergrad.lower_level_obj.regularizer.zero_mean()
        hypergrad.lower_level_obj.regularizer.load_state_dict(optimizer.hypergrad.lower_level_obj.regularizer.state_dict())
        # Log PSNR and loss
        psnr_value = psnr_fn(data[0][0].unsqueeze(0), init_loader[i][0].unsqueeze(0).cpu()).item()
        print("PSNR: ", psnr_value, "Loss: ", loss_val.item())

        # Detach tensors to free memory
        hypergrad.x_init.detach().cpu()
        init_data.detach().cpu()
        noisy_data[0].detach().cpu()
        data[0].detach().cpu()

        # Optional: Visualization every 20 epochs
        # if epoch % 20 == 0:
        #     plt.imshow(hypergrad.x_init[10].cpu().detach().permute(1, 2, 0).numpy())
        #     plt.title(f"PSNR: {psnr_fn(data[0][10], hypergrad.x_init[10].cpu()).mean().item()}")
        #     plt.show()
        init_test[0] = test_model(test_loader, noisy_loader_test, test_hypergrad, init_test[0], psnr_fn, device, logs_dict)
        
        # Logging if optimizer step was successful
        if optimizer.state['successful']:
            logs_dict["loss_batch"].append(loss_val.item())
            loss_batch.append(logs_dict["loss_batch"][-1])
            logs_dict["eps"].append(hypergrad.lower_tol)
            logs_dict["step"].append(optimizer.lr)
            logs_dict["lower_iter"].append(hypergrad.logs["lower_counter"])
            logs_dict["cg_iter"].append(hypergrad.logs["cg_counter"])
            logs_dict["psnr_epoch"].append(psnr_fn(data[0], init_loader[i].cpu()).mean().item())

        # Check if budget is exhausted
        if hypergrad.logs["lower_counter"] + hypergrad.logs["cg_counter"] > budget:
            print("Budget exhausted")
            return False  # Stop training

    logs_dict["loss_epoch"] = logs_dict["loss_batch"]
    torch.save(logs_dict, directory + f'/Logs/logs_dict_{eps0}_{alpha}_{opt_alg}_2.pt')
    torch.save(hypergrad.lower_level_obj.regularizer.state_dict(), directory + f'/Logs/regularizer_{eps0}_{alpha}_{opt_alg}_2.pt')

    return True  # Continue training
def test_model(test_loader, noisy_loader_test, hypergrad, init, psnr_fn, device ,logs_dict):        
    for i, (data, noisy_data) in enumerate(zip(test_loader, noisy_loader_test)):
        init_test = init.clone().detach().to(device)
        # init_test = noisy_data[0].clone().detach()
        # hypergrad.warm_start = False
        # hypergrad.lower_skip = False
        hypergrad = update_data(hypergrad, init_test, noisy_data[0], data[0], device)
        out = hypergrad.FISTA(init_test, 1e-6, 10)
        logs_dict["loss_test"].append(hypergrad.upper_level_obj(out).item())
        logs_dict["psnr_test"].append(psnr_fn(data[0], out.cpu()).mean().item())
        print("PSNR: ", psnr_fn(data[0], out.cpu()).mean().item())
        # plt.imshow(out[0].cpu().detach().permute(1, 2, 0).numpy())
        # plt.show()
        return out.clone().detach()

def main_training_loop(training_mode, hypergrad, test_hypergrad, dataloader, noisy_loader, init_loader, test_loader, noisy_loader_test, device, psnr_fn, number_epochs, total_iter, alpha, eps0, setting, p, q, budget):
    optimizer = initialize_optimizer(hypergrad, device, alpha, training_mode)
    logs_dict = initialize_logs(setting, p, q, training_mode)
    logs_dict = initialize_log_values(dataloader, noisy_loader, init_loader, hypergrad, logs_dict, device, eps0, alpha, psnr_fn)
    directory = os.getcwd()

    for epoch in range(number_epochs):
        if training_mode in ["IGD", "MAID"]:
            budget_control = train_igd_maid_epoch(epoch, dataloader, noisy_loader, init_loader, hypergrad, optimizer, logs_dict, psnr_fn, device, number_epochs, budget)
        else:
            budget_control = train_epoch(epoch, dataloader, noisy_loader, init_loader, hypergrad, optimizer, logs_dict, len(dataloader), psnr_fn, alpha, eps0, setting, p, q, total_iter, budget, device)
            test_hypergrad.lower_level_obj.regularizer.load_state_dict(hypergrad.lower_level_obj.regularizer.state_dict())
            init_test[0] = test_model(test_loader, noisy_loader_test, test_hypergrad, init_test[0], psnr_fn, device, logs_dict)
          
        if not budget_control:
            print("Budget exhausted")
            break
    
    # Save the final logs
    torch.save(logs_dict, f'{directory}/Logs/final_logs_dict.pt')

# Example of how you'd call it
main_training_loop(training_Mode, hypergrad, test_hypergrad, dataloader, noisy_loader, init_loader, test_loader, noisy_loader_test, device, psnr, number_epochs, total_iter, alpha, eps0, setting, p, q, budget)