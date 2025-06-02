"""
This module contains model architectures and utility functions for hurricane intensity estimation.

The module includes:
- Neural network architectures (CNN, UNet, GraphGNN, VisionGNN) for intensity prediction
- Functions for data processing and normalization
- Functions for calculating statistics and plotting results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import torch
import xarray as xr

import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

import torch.nn.functional as F


class BiasCorrectionModel(nn.Module):
    """
    A CNN model for bias correction of hurricane intensity predictions.
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels):
        super(BiasCorrectionModel, self).__init__()

        self.name = 'CNN'

        # Use the provided in_channels for the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Determine the flattened size after the convolutional layers
        self._get_flattened_size((in_channels, 32, 32))

        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 2)  # Output MSLP and maximum wind speed

    def _get_flattened_size(self, shape):
        """Calculate size of flattened features after convolutions"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = self.convs(dummy_input)
            self._to_linear = np.prod(dummy_output.shape[1:])

    def forward(self, x):
        """Forward pass through the network"""
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class UNet(nn.Module):
    """
    A UNet architecture for hurricane intensity prediction.
    
    Args:
        in_channels (int): Number of input channels (default: 38)
        out_channels (int): Number of output channels (default: 2)
    """
    def __init__(self, in_channels=38, out_channels=2):
        super(UNet, self).__init__()
        self.name = 'UNet'

        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        
        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  

    def conv_block(self, in_channels, out_channels):
        """Create a convolutional block with ReLU activation"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        """Create an upsampling convolutional block"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """Forward pass implementing the U-Net architecture"""
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool(e4)
        
        b = self.bottleneck(p4)
        
        u4 = self.upconv4(b)
        u4 = self.center_crop(u4, e4.size())
        c4 = torch.cat([u4, e4], dim=1)
        d4 = self.decoder4(c4)
        
        u3 = self.upconv3(d4)
        u3 = self.center_crop(u3, e3.size())
        c3 = torch.cat([u3, e3], dim=1)
        d3 = self.decoder3(c3)
        
        u2 = self.upconv2(d3)
        u2 = self.center_crop(u2, e2.size())
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.decoder2(c2)
        
        u1 = self.upconv1(d2)
        u1 = self.center_crop(u1, e1.size())
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.decoder1(c1)
        
        out = self.final_conv(d1)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1) 
        
        return out

    def center_crop(self, layer, target_size):
        """
        Crop the layer to match the target size.
        
        Args:
            layer (Tensor): Layer to be cropped
            target_size (tuple): Desired output size
        """
        _, _, H, W = layer.size()
        th, tw = target_size[2], target_size[3]
        ch = (H - th) // 2
        cw = (W - tw) // 2
        return layer[:, :, ch:H-ch, cw:W-cw]


class GraphGNN(nn.Module):
    """
    A Graph Neural Network for processing graph-structured data.
    
    Args:
        in_channels (int): Number of input features
        out_channels (int): Number of output features
        hidden_channels (int): Number of hidden features (default: 128)
        num_layers (int): Number of GNN layers (default: 3)
    """
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=3):
        super(GraphGNN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index, batch):
        """Forward pass through the GNN"""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return x


class VisionGNN(nn.Module):
    """
    A Vision-GNN hybrid model that processes image patches using a GNN.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        img_size (int): Size of input image (default: 32)
        patch_size (int): Size of image patches (default: 8)
        hidden_channels (int): Number of hidden features (default: 128)
        num_layers (int): Number of GNN layers (default: 3)
    """
    def __init__(self, in_channels, out_channels, img_size=32, patch_size=8, hidden_channels=128, num_layers=3):
        super(VisionGNN, self).__init__()
        self.name = 'GraphGNN'

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.node_dim = in_channels * patch_size * patch_size
        
        # Linear embedding for each patch (node feature)
        self.node_embedding = nn.Linear(self.node_dim, hidden_channels)
        
        # Graph Neural Network
        self.gnn = GraphGNN(hidden_channels, out_channels, hidden_channels=hidden_channels, num_layers=num_layers)

    def forward(self, x):
        """Forward pass converting image to patches and processing with GNN"""
        b, c, h, w = x.shape
        
        # Reshape input into patches (nodes)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(b, -1, c * self.patch_size * self.patch_size)
        
        # Embed patches (node features)
        x = self.node_embedding(patches)
        
        # Create a fully connected graph (example edge_index for simplicity)
        num_nodes = x.size(1)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous().to(x.device)
        edge_index = edge_index.repeat(b, 1, 1).view(-1, 2).t().contiguous()
        
        # Prepare node feature matrix and batch index
        x = x.view(-1, x.size(-1))  # (batch_size * num_nodes, hidden_channels)
        batch = torch.arange(b).repeat_interleave(num_nodes).to(x.device)
        
        # Apply GNN
        out = self.gnn(x, edge_index, batch)
        
        return out
    


def calculate_means_stds_track_files(track_data_files):
    """
    Calculate means and standard deviations from track files.
    
    Args:
        track_data_files (list): List of track data file paths
        
    Returns:
        dict: Dictionary containing means and standard deviations
    """
    track_data = []
    for files in track_data_files:
        track_data.append(pd.read_csv(files)[['mslp', 'vmax']])

    track_data = np.array(track_data).reshape(-1, 1, 2).squeeze()
    track_data_mean = np.nanmean(track_data, axis=0)
    track_data_std = np.nanstd(track_data, axis=0)
    track_means_stds = {'means': track_data_mean, 'stds': track_data_std}
    return track_means_stds

def get_means_stds_from_hurdat_tracks(track_data_files=None):
    """
    Get means and standard deviations from HURDAT track files.
    
    Args:
        track_data_files (list, optional): List of track files. If None, uses default path
        
    Returns:
        dict: Dictionary containing means and standard deviations
    """
    if track_data_files is None:
        track_data_folder = '/nas/rstor/akumar/USA/PhD/Objective04_ERA5/Results_v04/Hurricane_tracks/'
        track_data_files = sorted(glob.glob(track_data_folder + '*HURDAT.csv'))
    means_track_files = calculate_means_stds_track_files(track_data_files)
    return means_track_files


def calculate_mean_differences(output_track, mean=True):
    """
    Calculate differences between predicted and actual hurricane metrics.
    
    Args:
        output_track (dict): Dictionary containing track data
        mean (bool): Whether to return mean differences (default: True)
        
    Returns:
        dict: Dictionary containing differences or mean differences
    """
    differences = {}

    differences['merra_mslp'] = output_track['merra']['mslp'] - output_track['hurdat']['mslp']
    differences['model_mslp'] = output_track['model']['mslp'] - output_track['hurdat']['mslp']
    differences['merra_ws'] = output_track['merra']['ws'] - output_track['hurdat']['vmax']
    differences['model_ws'] = output_track['model']['ws'] - output_track['hurdat']['vmax']

    if mean:
        # Calculate the mean of the differences
        mean_differences = {key: value.mean() for key, value in differences.items()}
        return mean_differences
    else:
        # Return the raw differences
        return differences


def plot_combined_results(output_tracks_era, output_tracks_merra, diffs_era, diffs_merra, fhours, prefix=None):
    """
    Plot combined results comparing ERA and MERRA predictions.
    
    Args:
        output_tracks_era (list): List of ERA track predictions
        output_tracks_merra (list): List of MERRA track predictions
        diffs_era (list): List of ERA differences
        diffs_merra (list): List of MERRA differences
        fhours (array): Forecast hours
        prefix (str, optional): Plot title prefix
        
    Returns:
        array: Array of plot axes
    """

    # colors = ('#6B8E23', 'r', '#FF8000', '#FFC125')
    colors = ('b', 'r', 'g', 'brown')

    fig, axs = plt.subplots(2, 2, figsize=(16.5, 9.5))
    
    # ERA plots
    output_track = output_tracks_era[0]
    axs[0,0].plot(fhours, output_track['hurdat']['vmax'], color='k', marker='o', label='HURDAT')
    axs[0,0].plot(fhours, output_track['merra']['ws'], color=colors[0], marker='o', label=f'FCN | {diffs_era[0]["merra_ws"]:.2f} m/s')
    axs[0,0].plot(fhours, output_tracks_era[0]['model']['ws'], color=colors[1], marker='o', label=f'UNet | {diffs_era[0]["model_ws"]:.2f} m/s')
    axs[0,0].plot(fhours, output_tracks_era[1]['model']['ws'], color=colors[2], marker='o', label=f'CNN | {diffs_era[1]["model_ws"]:.2f} m/s')
    axs[0,0].plot(fhours, output_tracks_era[2]['model']['ws'], color=colors[3], marker='o', label=f'GNN | {diffs_era[2]["model_ws"]:.2f} m/s')
    axs[0,0].set_ylabel('Wind Speed (m/s)')
    axs[0,0].set_title(f'(a) ERA-FCN Wind Speed ')#| IC: {prefix}')
    axs[0,0].legend(fontsize=11, frameon=False, )

    axs[0,1].plot(fhours, output_track['hurdat']['mslp'], color='k', marker='o', label='HURDAT')
    axs[0,1].plot(fhours, output_track['merra']['mslp'], color=colors[0], marker='o', label=f'FCN | {diffs_era[0]["merra_mslp"]:.2f} hPa')
    axs[0,1].plot(fhours, output_tracks_era[0]['model']['mslp'], color=colors[1], marker='o', label=f'UNet | {diffs_era[0]["model_mslp"]:.2f} hPa')
    axs[0,1].plot(fhours, output_tracks_era[1]['model']['mslp'], color=colors[2], marker='o', label=f'CNN | {diffs_era[1]["model_mslp"]:.2f} hPa')
    axs[0,1].plot(fhours, output_tracks_era[2]['model']['mslp'], color=colors[3], marker='o', label=f'GNN | {diffs_era[2]["model_mslp"]:.2f} hPa')
    axs[0,1].set_ylabel('MSLP (hPa)')
    axs[0,1].set_title(f'(b) ERA-FCN MSLP ')#| IC: {prefix}')
    axs[0,1].legend(fontsize=11, frameon=False, )

    # MERRA plots  
    output_track = output_tracks_merra[0]
    axs[1,0].plot(fhours, output_track['hurdat']['vmax'], color='k', marker='o', label='HURDAT')
    axs[1,0].plot(fhours, output_track['merra']['ws'], color=colors[0], marker='o', label=f'FCN | {diffs_merra[0]["merra_ws"]:.2f} m/s')
    axs[1,0].plot(fhours, output_tracks_merra[0]['model']['ws'], color=colors[1], marker='o', label=f'UNet | {diffs_merra[0]["model_ws"]:.2f} m/s')
    axs[1,0].plot(fhours, output_tracks_merra[1]['model']['ws'], color=colors[2], marker='o', label=f'CNN | {diffs_merra[1]["model_ws"]:.2f} m/s')
    axs[1,0].plot(fhours, output_tracks_merra[2]['model']['ws'], color=colors[3], marker='o', label=f'GNN | {diffs_merra[2]["model_ws"]:.2f} m/s')
    axs[1,0].set_ylabel('Wind Speed (m/s)')
    axs[1,0].set_title(f'(c) MERRA-FCN Wind Speed ')#| IC: {prefix}')
    axs[1,0].legend(fontsize=11, frameon=False,)
    axs[1,0].set_xlabel('Forecast Hours')

    axs[1,1].plot(fhours, output_track['hurdat']['mslp'], color='k', marker='o', label='HURDAT')
    axs[1,1].plot(fhours, output_track['merra']['mslp'], color=colors[0], marker='o', label=f'FCN | {diffs_merra[0]["merra_mslp"]:.2f} hPa')
    axs[1,1].plot(fhours, output_tracks_merra[0]['model']['mslp'], color=colors[1], marker='o', label=f'UNet | {diffs_merra[0]["model_mslp"]:.2f} hPa')
    axs[1,1].plot(fhours, output_tracks_merra[1]['model']['mslp'], color=colors[2], marker='o', label=f'CNN | {diffs_merra[1]["model_mslp"]:.2f} hPa')
    axs[1,1].plot(fhours, output_tracks_merra[2]['model']['mslp'], color=colors[3], marker='o', label=f'GNN | {diffs_merra[2]["model_mslp"]:.2f} hPa')
    axs[1,1].set_ylabel('MSLP (hPa)')
    axs[1,1].set_title(f'(d) MERRA-FCN MSLP ')#| IC: {prefix}')
    axs[1,1].legend(fontsize=11, frameon=False, )
    axs[1,1].set_xlabel('Forecast Hours')

    [ax.set_xticks(np.arange(0, 120, 12)) for ax in axs.flatten()]
    plt.tight_layout()
    return axs

def get_suptitle(input_str):
    """
    Generate a plot super title from input string.
    
    Args:
        input_str (str): Input string containing storm ID and init time
        
    Returns:
        str: Formatted super title
    """
    # Split into parts
    storm_id, init_time = input_str.split('_')
    year = storm_id[:4]
    basin_number = storm_id[4:6]
    name = storm_id[6:]

    init_datetime = f"{init_time[:4]}-{init_time[4:6]}-{init_time[6:8]} {init_time[8:]} UTC"

    return f"{name} ({basin_number}) | Initialized at {init_datetime}"


def scale_dataset_domain(filename, means_folder):
    """
    Scale dataset using global means and standard deviations.
    
    Args:
        filename (str): Path to dataset file
        means_folder (str): Path to folder containing means and stds
        
    Returns:
        Dataset: Scaled dataset
    """
    global_means = xr.open_dataset(f'{means_folder}/global_means.npy')
    global_stds = xr.open_dataset(f'{means_folder}/global_stds.npy')
    return (xr.open_dataset(filename) - global_means) / global_stds

def unscale_data(predictions, means_track_files):
    """
    Unscale predictions using track means and standard deviations.
    
    Args:
        predictions (array): Scaled predictions
        means_track_files (dict): Dictionary containing means and stds
        
    Returns:
        array: Unscaled predictions
    """
    predictions = torch.tensor(predictions, dtype=torch.float32)
    
    means = torch.tensor(means_track_files['means'], dtype=torch.float32)
    stds = torch.tensor(means_track_files['stds'], dtype=torch.float32)
    
    unscaled_predictions = predictions * stds + means
    
    return unscaled_predictions.numpy()


def run_inference(hurricane_prefix=None, files=None, means_track_files=None, model=None, model_weights=None, cropped_data_meanstd_folder=None):
    """
    Run inference on hurricane data using specified model.
    
    Args:
        hurricane_prefix (str): Prefix identifying hurricane data
        files (dict): Dictionary containing file paths
        means_track_files (dict): Dictionary containing means and stds
        model (nn.Module): Model to use for inference
        model_weights (str): Path to model weights
        cropped_data_meanstd_folder (str): Path to means/stds folder
        
    Returns:
        dict: Dictionary containing predictions and actual values
        
    Raises:
        ValueError: If required arguments are missing
        FileNotFoundError: If files cannot be found
        RuntimeError: If errors occur during processing
    """
    if not hurricane_prefix or not files or not means_track_files or not model or not model_weights:
        raise ValueError("All arguments must be provided")

    try:
        

        data_file = [f for f in files['cropped_data_files'] if hurricane_prefix in f][0]
        track_file = [f for f in files['hurdat_track_files'] if hurricane_prefix in f][0]
        merra_file = [f for f in files['merra_track_files'] if hurricane_prefix in f][0]

    except StopIteration:
        raise FileNotFoundError("One or more files with the specified hurricane_prefix were not found")


    try:
        nc_data = scale_dataset_domain(data_file, cropped_data_meanstd_folder)['predicted']
        # print(nc_data.shape)
        data_variable = torch.from_numpy(nc_data[:, :, 4:-4, 4:-4].values).to(dtype=torch.float32)
        # print('After cropping', data_variable.shape)
    except Exception as e:
        raise RuntimeError(f"Error processing data files: {e}")

    # Set device and load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    try:
        model_weights = torch.load(model_weights, map_location=device)
        model.load_state_dict(model_weights)
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")

    model.eval()

    # Run inference
    with torch.no_grad():
        try:
            predictions = model(data_variable.to(device))
            unscaled_predictions = unscale_data(predictions.cpu().numpy(), means_track_files)
            bias_corrected_tracks = pd.DataFrame(unscaled_predictions, columns=['mslp', 'ws'])
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}")

    # Read additional data
    try:
        hurdat_data = pd.read_csv(track_file)[['mslp', 'vmax']]
        merra_data = pd.read_csv(merra_file)[['mslp', 'ws']]
        merra_data['mslp'] = merra_data['mslp'] / 100
    except Exception as e:
        raise RuntimeError(f"Error reading track or MERRA files: {e}")

    return {'hurdat': hurdat_data, 'model': bias_corrected_tracks, 'merra': merra_data}
