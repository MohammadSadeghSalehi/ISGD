import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image , ImageOps
import os , sys, math
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Convex models')))
from TV import *
from FoE import *
from ICNN import *

def compute_psnr(img1, img2, max_pixel_value=1):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): First input image. Shape can be [H, W] for grayscale or [3, H, W] for colored.
        img2 (torch.Tensor): Second input image. Same shape as img1.
        max_pixel_value (float): Maximum possible pixel value (default is 255 for 8-bit images).

    Returns:
        float: The PSNR value between the two images.
    """
    
    # Ensure img1 and img2 are torch tensors
    assert isinstance(img1, torch.Tensor), "img1 should be a torch tensor"
    assert isinstance(img2, torch.Tensor), "img2 should be a torch tensor"
    
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "The input images must have the same dimensions"
    C, H, W = img1.shape
    # Determine if the image is colored or grayscale
    if img1.ndim == 3 and img1.shape[0] == 3:
        # If the image is colored, calculate MSE across all channels
        mse = torch.norm(img1 - img2)**2/ (C * H * W)
    elif img1.ndim == 3 and img1.shape[0] == 1:
        mse = torch.norm(img1 - img2)**2 / (H * W)
    else:
        raise ValueError("Image must be either grayscale [1, H, W] or colored [3, H, W]")
    
    # If MSE is zero, PSNR is infinite (identical images)
    if mse == 0:
        return float('inf')

    # Compute PSNR
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse.item()))
    
    return psnr



device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class Inference:
    def __init__(self, func=None, x0=None, gt = None):
        self.func = func
        self.x = x0
        self.gt = gt
    def power_method(self, x0, max_iter):
        """
        Power method to compute the largest eigenvalue of matrix A applied as a function.
        """
        A = lambda u: self.hessian_vector_product(self.x, u)
        x = x0
        for i in range(max_iter):
            x = A(x)
            x = x / torch.norm(x)
        return torch.dot(A(x).view(-1), x.view(-1)) / torch.dot(x.view(-1), x.view(-1)) , x

    def grad_lower_level(self, x):
        """
        Compute the gradient of the given function with respect to x.
        """
        grad = torch.autograd.grad(self.func(x), x, create_graph=True, retain_graph=True)[0]
        return grad

    def hessian_vector_product(self, x, v):
        """
        Compute the Hessian-vector product for the given function at x.
        """
        return torch.autograd.functional.vhp(self.func, x, v)[1].detach()

    def FISTA(self, x, tol, max_iter, lipschitz_estimate=True, verbose=False):
        """
        FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for optimization.
        """
        # Initialize variables
        x = x.clone().detach()
        y = x.clone().detach()
        t = 1
        if lipschitz_estimate:
            L,_ = self.power_method(torch.randn_like(x), 50)
            L = L.item()
        else:
            L = 1000.0
        for i in range(max_iter):
            y.requires_grad_(True)  # Ensure y has gradients enabled for each iteration
            # Compute gradient
            grad = self.grad_lower_level(y)
            with torch.no_grad():
                x_new = y - (1 / L) * grad  # Gradient step

                # Nesterov acceleration step
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y = x_new + ((t - 1) / t_new) * (x_new - x)

                # Update x and t
                x.copy_(x_new)
                t = t_new

            # Check convergence
            norm_inner = torch.norm(grad, dim=(2, 3))
            norm_channel = torch.mean(norm_inner, dim=1)
            if norm_channel.max() < tol:
                if verbose:
                    print(f"Converged at iteration {i + 1}, gradient norm: {torch.norm(grad).item()}")
                break

            if verbose:
                loss = self.func(x).item()
                grad_norm = torch.norm(grad).item()
                print(f"Iteration {i + 1}, loss: {loss}, gradient norm: {grad_norm}")
        return x.detach()
    def LBFGS(self, x, tol, max_iter, verbose=False, lipschitz_estimate=False):
        """
        L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) for optimization.
        """
        x = x.clone().detach()
        x = x.requires_grad_(True)
        if lipschitz_estimate:
            L,_ = self.power_method(torch.randn_like(x), 100)
            L = L.item()
        else:
            L = 1000.0
        print("L: ", L)
        optimizer = torch.optim.LBFGS([x], lr=1/L, max_iter=1000, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = self.func(x)
            x.grad = self.grad_lower_level(x)
            print("loss: ", loss.item(), "gradient norm: ", torch.norm(x.grad).item())
            return loss
        optimizer.step(closure)
        return x.detach()
    def adam(self, x, tol, max_iter, verbose=False):
        """
        Adam optimization algorithm.
        """
        x = x.clone().detach()
        x = x.requires_grad_(True)
        optimizer = torch.optim.AdamW([x], lr=1e-1)
        for i in range(max_iter):
            optimizer.zero_grad()
            loss = self.func(x)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Iteration {i + 1}, loss: {loss.item()}")
        return x.detach()
    def psnr_ssim(self, img1, img2):
        # check shapes 
        # img2 = torch.clamp(img2, 0, 1)
        if img1.shape[0] == 1:
            img1 = img1.squeeze(0)
            if img1.shape[0] == 1:
                img1 = img1.squeeze(0)
            else:
                img1 = img1.permute(1, 2, 0)
        if img2.shape[0] == 1:
            img2 = img2.squeeze(0)
            if img2.shape[0] == 1:
                img2 = img2.squeeze(0)
            else:
                img2 = img2.permute(1, 2, 0)
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().detach().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().detach().numpy()
        return peak_signal_noise_ratio(img1, img2, data_range=1), structural_similarity(img1, img2, multichannel=True, full= True, data_range=1, channel_axis= 2)[0]
        
    def plot(self):
        """
        Plot the function.
        """
        psnr , ssim = self.psnr_ssim(self.x, self.gt)
        if self.x.shape[1] == 1: 
            image = self.x.squeeze(0).squeeze(0).cpu().detach().numpy()
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(f'PSNR: {psnr:.2f}, SSIM: {ssim:.2f}')
            plt.show()
        else:
            image = self.x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            image = np.clip(image, 0, 1)
            plt.imshow(image, vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f'PSNR: {psnr:.2f}, SSIM: {ssim:.2f}')
            plt.show()
    def run(self, tol=1e-3, max_iter=100, lipschitz_estimate=True, verbose=False):
        """
        Run the optimization algorithm.
        """
        self.x = self.FISTA(self.x, tol, max_iter, lipschitz_estimate, verbose)
        # self.x = self.LBFGS(self.x, tol, max_iter, verbose)
        # self.x = self.adam(self.x, tol, max_iter, verbose)
        self.plot()
        return self.x
    
class Lower_Level(nn.Module):
    def __init__(self, measurement, regularizer, forward_operator = lambda x: x):
        super(Lower_Level, self).__init__()
        self.regularizer = regularizer
        self.measurement = measurement
        self.A = forward_operator
    def data_fidelity(self, x):
        # return 0.5 * torch.norm(self.A(x) - self.measurement)**2/ self.measurement.shape[0] 
        return 0.5 *torch.nn.MSELoss()(self.A(x), self.measurement)
    def forward(self, x):
        return self.data_fidelity(x) + 0.007 *self.regularizer(x)

dir = os.getcwd()
img_size_x, img_size_y = 128, 128
channel = 3
# define model
regularizer = FoE(channel, 10, 7, learnable_smoothing = True, learnable_weights = True).to(device)
# regularizer.init_weights()
# load model weights
regularizer.load_state_dict(torch.load('regularizer.pt'))
if channel == 1:
    input = Image.open(dir + '/nessy.jpg').convert('L')
    transform = transforms.Compose([transforms.Lambda(lambda x: ImageOps.exif_transpose(x)),transforms.Resize((img_size_x,img_size_y)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])
else:
    input = Image.open(dir + '/nessy.jpg').convert('RGB')
    transform = transforms.Compose([transforms.Lambda(lambda x: ImageOps.exif_transpose(x)),transforms.Resize((img_size_x,img_size_y)),transforms.ToTensor()])
img = transform(input)
img = img.view(1,channel,img_size_x,img_size_y).to(device)
plt.imshow(img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
plt.axis('off')
plt.show()

# noisy image
noise_level = 25
noise_factor = noise_level/255
operator = lambda x: x
noisy_image = operator(img) + noise_factor * torch.randn_like(img)
plt.imshow(noisy_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), vmin=0, vmax=1)
plt.axis('off')
plt.title(f'Noisy image, PSNR: {peak_signal_noise_ratio(img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), noisy_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), data_range=1):.2f}')
plt.show()

func = Lower_Level(noisy_image, regularizer, forward_operator = lambda x: x).forward
inference = Inference(func, (noisy_image), img)
out = inference.run(tol=1e-8, max_iter=500, lipschitz_estimate=True, verbose=True)





