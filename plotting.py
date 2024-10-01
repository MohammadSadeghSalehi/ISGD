import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import glob

# Set the directory where logs are stored
directory = os.getcwd()

# Find all the files that match the pattern logs_dict_*.pt
log_files = glob.glob(os.path.join(directory, 'Logs', 'logs_dict_*.pt'))
log_files = sorted(log_files, key=lambda x: x.split('_')[-2])

# Initialize lists to collect data for plotting
loss_batch_list = []
loss_epoch_list = []
psnr_list = []
eps_list = []
step_list = []
lower_iter_list = []
cg_iter_list = []
legend_list = []
total_iter_list = []
loss_test_list = []
psnr_test_list = []

# plotting options
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'figure.figsize': (10, 8)})
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'legend.fontsize': 16})
ylim = 300

# Loop over all log files and load them
for log_file in log_files:
    logs_dict = torch.load(log_file)
    # Extend data from the logs_dict to the respective lists
    loss_batch_list.append(logs_dict["loss_batch"])
    loss_epoch_list.append(logs_dict["loss_epoch"])
    psnr_list.append(logs_dict["psnr_epoch"])
    eps_list.append(logs_dict["eps"])
    step_list.append(logs_dict["step"])
    lower_iter_list.append(logs_dict["lower_iter"])
    cg_iter_list.append(logs_dict["cg_iter"]) 
    loss_test_list.append(logs_dict["loss_test"])
    psnr_test_list.append(logs_dict["psnr_test"])
    if logs_dict["setting"] == "constant":
        legend_list.append(logs_dict["train_mode"] + rf' $\epsilon_0 = ${logs_dict["eps"][0]:.1f}, $\alpha_0 = ${logs_dict["step"][0]:.1e}')
    else:
        if logs_dict["train_mode"] != "ISGD":
            # decaying eps and step size
            # legend_list.append(logs_dict["train_mode"] + f' $\epsilon_0$ = {logs_dict["eps"][0]}, p = {logs_dict["p"]}, q = {logs_dict["q"]}')
            # fixed eps and step size
            legend_list.append(logs_dict["train_mode"] + rf' $\epsilon_0$ = {logs_dict["eps"][0]:.1f}, $\alpha_0$ = {logs_dict["step"][0]:.1e}')
        else:
            # decaying eps and step size
            # legend_list.append(logs_dict["train_mode"] + f' $\epsilon_0$ = {logs_dict["eps"][0]}, p = {logs_dict["p"]}, q = {logs_dict["q"]}')
            # fixed eps and step size
            legend_list.append(logs_dict["train_mode"] + rf' $\epsilon_0$ = {logs_dict["eps"][0]:.1f}, $\alpha_0$ = {logs_dict["step"][0]:.1e}')

    # Compute the cumulative lower_iter + cg_iter
    total_iter_list.append(np.array(logs_dict["lower_iter"]) + np.array(logs_dict["cg_iter"]))

for i, loss_batch in enumerate(loss_batch_list):
    plt.plot(range(0, len(loss_batch)), loss_batch[:], label= legend_list[i])
    plt.xlabel('Number of Batches')
    plt.ylabel('Loss')
    plt.title('Loss per Batch')
    plt.grid(True)
    plt.yscale('log')
plt.ylim(None, ylim)
plt.legend()
plt.savefig(directory + '/Logs/loss_batch.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot loss_epoch against number of epochs
plt.figure(figsize=(10, 6))
for i, loss_epoch in enumerate(loss_epoch_list):
    plt.plot(range(0,len(loss_epoch)), loss_epoch[:], label=legend_list[i])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid(True)
    plt.yscale('log')
plt.ylim(None, ylim)
plt.legend()
plt.savefig(directory + '/Logs/loss_epoch.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot loss_epoch against lower_iter + cg_iter
plt.figure(figsize=(10, 6))
for i , (total_iter, loss_epoch_list) in enumerate(zip(total_iter_list, loss_epoch_list)):
    plt.plot(total_iter[:], loss_epoch_list[:], label=legend_list[i])
    plt.xlabel('Lower Iterations + CG Iterations')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch vs Total Iterations')
    plt.grid(True)
    plt.yscale('log')
plt.ylim(None, ylim)

plt.legend()
plt.savefig(directory + '/Logs/loss_total_iter.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot eps
plt.figure(figsize=(10, 6))
for i, eps in enumerate(eps_list):
    plt.plot(range(0, len(eps)), eps[:], label=legend_list[i])
    plt.xlabel('Iterations')
    plt.ylabel(f'$\epsilon$')
    plt.title('Accuracy over Iterations')
    plt.grid(True)
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.savefig(directory + '/Logs/eps.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot step size
plt.figure(figsize=(10, 6))
for i, step in enumerate(step_list):
    plt.plot(range(0,len(step)), step[:], label=legend_list[i])
    plt.xlabel('Iterations')
    plt.ylabel('Step Size')
    plt.title('Step Size over Iterations')
    plt.grid(True)
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.savefig(directory + '/Logs/step.png', dpi=300, bbox_inches='tight')
plt.show()

# plot psnr
plt.figure(figsize=(10, 6))
for i, (total_iter, psnr_list) in enumerate(zip(total_iter_list, psnr_list)):
    # plt.plot(range(0,len(psnr)), psnr[:], label=legend_list[i])
    # plt.xlabel('Number of Epochs')
    plt.plot(total_iter[:], psnr_list[:], label=legend_list[i])
    plt.xlabel('Total lower level computations')
    plt.ylabel('PSNR')
    plt.title('PSNR per Epoch')
    plt.grid(True)
plt.legend()
plt.savefig(directory + '/Logs/psnr.png', dpi=300, bbox_inches='tight')
plt.show()

# plot test loss
plt.figure(figsize=(10, 6))
for i, loss_test in enumerate(loss_test_list):
    plt.plot(range(0,len(loss_test)), loss_test[:], label=legend_list[i])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss per Epoch')
    plt.grid(True)
    plt.yscale('log')
plt.ylim(None, ylim)
plt.legend()
plt.savefig(directory + '/Logs/loss_test.png', dpi=300, bbox_inches='tight')
plt.show()

# plot test psnr
plt.figure(figsize=(10, 6))
for i, psnr_test in enumerate(psnr_test_list):
    plt.plot(range(0,len(psnr_test)), psnr_test[:], label=legend_list[i])
    plt.xlabel('Number of Epochs')
    plt.ylabel('PSNR')
    plt.title('Test PSNR per Epoch')
    plt.grid(True)
plt.legend()
plt.savefig(directory + '/Logs/psnr_test.png', dpi=300, bbox_inches='tight')
plt.show()