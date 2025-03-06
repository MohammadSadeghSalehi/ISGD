import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


# Set seaborn style
sns.set(style="whitegrid", font_scale=1.5)

# Set the directory where logs are stored
directory = os.getcwd()

# Find all the files that match the pattern logs_dict_*.pt
log_files = sorted(glob.glob(os.path.join(directory, 'Logs', 'logs_dict_*.pt')),key=lambda x: x.split('_')[2])
# sorting based on the setting and step size
    
# Initialize lists to collect data for plotting
loss_batch_list, loss_epoch_list, psnr_list, eps_list, step_list = [], [], [], [], []
lower_iter_list, cg_iter_list, legend_list, total_iter_list = [], [], [], []
loss_test_list, psnr_test_list = [], []
p_list, q_list = [], []
iter_cost_list = []
# Set plot configurations
ylim = 300

# Helper function to load data
def load_data_from_logs(log_files):
    for log_file in log_files:
        logs_dict = torch.load(log_file)
        loss_batch_list.append(logs_dict["loss_batch"])
        loss_epoch_list.append(logs_dict["loss_epoch"])
        psnr_list.append(logs_dict["psnr_epoch"])
        eps_list.append(logs_dict["eps"])
        step_list.append(logs_dict["step"])
        lower_iter_list.append(logs_dict["lower_iter"])
        cg_iter_list.append(logs_dict["cg_iter"])
        loss_test_list.append(logs_dict["loss_test"])
        psnr_test_list.append(logs_dict["psnr_test"])
        p_list.append(logs_dict["p"])
        q_list.append(logs_dict["q"])
        iter_cost_list.append(logs_dict["cost_iter"])
        if logs_dict["train_mode"] == 'MAID':
            for i in range(len(loss_epoch_list[-1])-1):
                if loss_epoch_list[-1][i+1] > loss_epoch_list[-1][i]:
                    loss_epoch_list[-1][i+1] = loss_epoch_list[-1][i]
                
        if logs_dict["setting"] == "constant":
            legend_list.append(fr'{logs_dict["train_mode"]} $\alpha_0 = {logs_dict["step"][0]:.0e}$')
        else:
            if q_list[-1] != 0 and logs_dict["train_mode"] != "MAID":
                legend_list.append(fr'{logs_dict["train_mode"]} {"DS"} $\alpha_0 = {logs_dict["step"][0]:.0e}$')
            else:
                legend_list.append(fr'{logs_dict["train_mode"]} $\alpha_0 = {logs_dict["step"][0]:.0e}$')

        # Combine lower_iter and cg_iter
        total_iter_list.append(np.array(logs_dict["lower_iter"]) + np.array(logs_dict["cg_iter"]))

# Function to plot
def plot_metric_vs_iterations(metric_list, ylabel, title, save_path, yscale='log', cut_off_value=None, xlim=None, x_axis=None, xlabel='Iterations'):
    plt.figure(figsize=(10, 6))
    cool_colors = ['blue', 'lightskyblue', 'cornflowerblue', 'darkviolet','lightgreen']
    warm_colors = ['red', 'orange', 'salmon', 'chocolate', 'khaki']
    for i, metric in enumerate(metric_list):
        if cut_off_value is not None and x_axis is None:
            cut_off_index = next((index for index, value in enumerate(total_iter_list[i]) if value >= cut_off_value), len(total_iter_list[i]))
            if "MAID" in legend_list[i]:
                plt.plot(total_iter_list[i][:cut_off_index], metric[:cut_off_index], label=legend_list[i] ,linewidth=2, linestyle='-', color='black')
            elif "DS" in legend_list[i]:
                plt.plot(total_iter_list[i][:cut_off_index], metric[:cut_off_index], label=legend_list[i], linewidth=2, linestyle='-.', color = cool_colors[0])
                cool_colors.pop(0)
            else:
                plt.plot(total_iter_list[i][:cut_off_index], metric[:cut_off_index], label=legend_list[i], linewidth=2, linestyle='--', color = warm_colors[0])
                warm_colors.pop(0)
        else:
            if cut_off_value is not None and x_axis is not None:
                cut_val = min(cut_off_value, len(metric))
                plt.plot(range(cut_val), metric[:cut_val], label=legend_list[i], linewidth=2)    
            else:
                 plt.plot(range(len(metric)), metric, label=legend_list[i], linewidth=2)
        if xlim is not None:
            plt.xlim(xlim)
    
    # Use ScalarFormatter and set the power limits for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((4, 4))  # Forces 10^4 formatting for values above 10,000

    # Apply the formatter to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.yscale(yscale)
    plt.legend(fontsize = 14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Load data
load_data_from_logs(log_files)

# Plot each metric using the helper function
plot_metric_vs_iterations(loss_batch_list, 'Loss', 'Loss per batch', f'{directory}/Logs/loss_batch.png', yscale='log', xlabel='Number of Batches')
plot_metric_vs_iterations(loss_epoch_list, 'Loss', 'Loss per epoch', f'{directory}/Logs/loss_epoch.png', yscale='log', cut_off_value=100, x_axis='Epochs', xlabel='Number of epochs')
plot_metric_vs_iterations(eps_list, '$\\epsilon$', 'Accuracy over Iterations', f'{directory}/Logs/eps.png',xlabel='Number of Epochs')
plot_metric_vs_iterations(step_list, 'Step Size', 'Step Size over Iterations', f'{directory}/Logs/step.png',xlabel='Number of Epochs')
plot_metric_vs_iterations(loss_epoch_list, 'Average loss per epoch', '', f'{directory}/Logs/loss_total_iter.png', yscale='log', cut_off_value=5e5, xlim=(0, 8e5), xlabel='Total computations')

# Plot PSNR
plot_metric_vs_iterations(psnr_list, 'Average PSNR per epoch', '', f'{directory}/Logs/psnr.png', cut_off_value=5e5, xlim=(0, 8e5), yscale= 'linear', xlabel='Total computations')

# Plot test loss
plot_metric_vs_iterations(loss_test_list, 'Loss', 'Test Loss per epoch', f'{directory}/Logs/loss_test.png', yscale='log', xlabel='Number of Epochs',x_axis='Epochs')

# Plot test PSNR
plot_metric_vs_iterations(psnr_test_list, 'PSNR', ' Average test PSNR per epoch', f'{directory}/Logs/psnr_test.png', xlim=(0, 40), yscale= 'linear', xlabel='Number of Epochs',x_axis='Epochs')

# Plot cost per iteration
cool_colors = ['blue', 'lightskyblue', 'cornflowerblue', 'darkviolet','lightgreen']
warm_colors = ['red', 'orange', 'salmon', 'chocolate', 'khaki']
plt.figure(figsize=(10, 6))
cut_off_value = 5e5
for i, cost in enumerate(iter_cost_list):
        
    cut_off_index = next((index for index, value in enumerate(iter_cost_list[i]) if value >= cut_off_value), len(iter_cost_list[i]))
    # comment out the following line if you want to plot x-axis in linear scale
    for j in range(len(cost)):
        if cost[j] == 0:
            cost[j] = 1
    if "MAID" in legend_list[i]:
        plt.plot(cost[:cut_off_index], loss_batch_list[i][:cut_off_index], label=legend_list[i], linewidth=2, linestyle='-', color='black')
    elif "DS" in legend_list[i]:
        plt.plot(cost[:cut_off_index], loss_batch_list[i][:cut_off_index], label=legend_list[i], linewidth=2, linestyle='-.', color = cool_colors[0])
        cool_colors.pop(0)
    else:
        plt.plot(cost[:cut_off_index], loss_batch_list[i][:cut_off_index], label=legend_list[i], linewidth=2, linestyle='--', color = warm_colors[0])
        warm_colors.pop(0)
        
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((4, 4))  # Forces 10^4 formatting for values above 10,000

# Apply the formatter to the x-axis
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
plt.xlabel('Total computations')
plt.ylabel('Loss per batch')
# plt.title('Loss per upper-level iteration vs total computations')
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.legend(fontsize = 14)
plt.savefig(f'{directory}/Logs/loss_cost.png', dpi=300, bbox_inches='tight')
