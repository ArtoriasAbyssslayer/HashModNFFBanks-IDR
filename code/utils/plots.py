import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
from utils import rend_util

def plot(model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, max_depth, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)

    depth = torch.ones(batch_size * num_samples).cuda().float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size,-1)

    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)

    # plot rendered images
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

    data = []

    # plot surface
    surface_traces = get_surface_trace(path=path,
                                       epoch=epoch,
                                       sdf=lambda x: model.implicit_network(x)[:, 0],
                                       resolution=resolution
                                       )
    data.append(surface_traces[0])

    # plot cameras locations
    for i, loc, dir in zip(indices, cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    for i, p, m in zip(indices, points, network_object_mask):
        p = p[m]
        sampling_idx = torch.randperm(p.shape[0])[:2048]
        p = p[sampling_idx, :]

        val = model.implicit_network(p)
        caption = ["sdf: {0} ".format(v[0].item()) for v in val]

        data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace

def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 10000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        # FIXED: Corrected the volume reshaping and transposition
        # The original had: .transpose([1, 0, 2]) which was causing incorrect orientation
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(resolution, resolution, resolution),  # Keep original XYZ order
            level=0,
            spacing=(grid['xyz'][0][1] - grid['xyz'][0][0],
                     grid['xyz'][1][1] - grid['xyz'][1][0],
                     grid['xyz'][2][1] - grid['xyz'][2][0]))
    
        # FIXED: Apply proper offset to align with grid
        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        
        # FIXED: Optional rotation to align with expected orientation
        # Uncomment one of these if you need specific orientation adjustments:
        
        # For skull facing up (if currently sideways):
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        verts = np.dot(verts, rotation_matrix.T)
   
        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=normals)  # FIXED: Removed negative sign
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

import torch

def sdf_chunked_gpu(points, model, model_input=None, pose_vecs=None, indices=None, eval_cameras=False, chunk_size=2**16):
    """
    Evaluate SDF in chunks to avoid GPU memory OOM.
    
    Args:
        points (torch.Tensor): [N, 3] points where SDF should be evaluated
        model (nn.Module): model with implicit_network(x) -> [N, 1]
        model_input (dict, optional): canonical input dict (intrinsics, uv, mask, etc.)
        pose_vecs (nn.Embedding, optional): camera poses if eval_cameras=True
        indices (torch.Tensor, optional): indices for pose_vecs
        eval_cameras (bool): whether to evaluate in camera space
        chunk_size (int): max number of points per chunk
    Returns:
        torch.Tensor: SDF values [N]
    """
    device = next(model.parameters()).device
    points = points.to(device)
    sdf_vals = []

    # Split points into chunks
    chunks = torch.split(points, chunk_size, dim=0)
    for pts_chunk in chunks:
        if model_input is not None:
            # Copy inputs for this chunk
            input_chunk = {k: v.clone() for k, v in model_input.items()}
        else:
            input_chunk = {}

        if eval_cameras and pose_vecs is not None:
            input_chunk['pose'] = pose_vecs(indices.to(device))

        # Evaluate the SDF
        with torch.no_grad():
            sdf_chunk = model.implicit_network(pts_chunk)[:, 0]  # [chunk_size]
            sdf_vals.append(sdf_chunk)

    # Concatenate all chunks
    return torch.cat(sdf_vals, dim=0)




def get_surface_high_res_mesh(sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(resolution)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 10000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    # FIXED: Same correction as above for marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(resolution, resolution, resolution),
        level=0,
        spacing=(grid['xyz'][0][1] - grid['xyz'][0][0],
                 grid['xyz'][1][1] - grid['xyz'][1][0],
                 grid['xyz'][2][1] - grid['xyz'][2][0]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
    mesh_low_res = trimesh.Trimesh(verts, faces, vertex_normals=normals)  # FIXED: Removed negative sign
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    eigenvalues, eigenvectors = torch.linalg.eig(s_cov)
    vecs = eigenvectors.real.transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']
    g = []
    
    for i, pnts in enumerate(torch.split(grid_points, 10000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                        pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)
        
    # MC to new grid
    points = grid_points
    z = []
    
    for i, pnts in enumerate(torch.split(points, 10000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        # FIXED: Same correction for high-res mesh
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                            grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][1] - grid_aligned['xyz'][0][0],
                    grid_aligned['xyz'][1][1] - grid_aligned['xyz'][1][0],
                    grid_aligned['xyz'][2][1] - grid_aligned['xyz'][2][0]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=normals)  # FIXED: Removed negative sign

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = (ground_true.cuda() + 1.) / 2.
    rgb_points = (rgb_points + 1. ) / 2.

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

# def lin2img(tensor, img_res):
#     batch_size, num_samples, channels = tensor.shape
#     return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
def lin2img(tensor, img_res):
    """
    Convert linear tensor to image format, handling dynamic tensor sizes.
    
    Args:
        tensor: Input tensor of shape [batch_size, num_samples, channels]
        img_res: Target image resolution [height, width]
    
    Returns:
        Reshaped tensor of shape [batch_size, channels, height, width]
    """
    batch_size, num_samples, channels = tensor.shape
    target_pixels = img_res[0] * img_res[1]
    
    if num_samples == target_pixels:
        # Perfect match - use original logic
        return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
    
    elif num_samples < target_pixels:
        # Not enough pixels - pad with zeros
        print(f"Warning: Padding tensor from {num_samples} to {target_pixels} pixels")
        padding_needed = target_pixels - num_samples
        padding = torch.zeros(batch_size, padding_needed, channels, device=tensor.device, dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, padding], dim=1)
        return padded_tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
    
    else:
        # Too many pixels - truncate
        print(f"Warning: Truncating tensor from {num_samples} to {target_pixels} pixels")
        truncated_tensor = tensor[:, :target_pixels, :]
        return truncated_tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def lin2img_adaptive(tensor, img_res):
    """
    Alternative version that adapts the image resolution to fit the actual data.
    Use this if you want to see the actual data without padding/truncating.
    """
    batch_size, num_samples, channels = tensor.shape
    target_pixels = img_res[0] * img_res[1]
    
    if num_samples == target_pixels:
        # Perfect match
        return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
    
    # Calculate closest image dimensions that fit the actual samples
    aspect_ratio = img_res[1] / img_res[0]  # width / height
    
    # Find dimensions that best fit our sample count
    height = int(np.sqrt(num_samples / aspect_ratio))
    width = int(height * aspect_ratio)
    
    # Adjust to fit exactly
    actual_pixels = height * width
    if actual_pixels > num_samples:
        # Try square dimensions
        side = int(np.sqrt(num_samples))
        height = width = side
        actual_pixels = side * side
    
    if actual_pixels > num_samples:
        # Final fallback - use whatever fits
        height = 1
        width = num_samples
        actual_pixels = num_samples
    
    print(f"Adapting resolution from {img_res} to [{height}, {width}] to fit {num_samples} samples")
    
    # Truncate to fit the calculated dimensions
    tensor_resized = tensor[:, :actual_pixels, :]
    return tensor_resized.permute(0, 2, 1).view(batch_size, channels, height, width)