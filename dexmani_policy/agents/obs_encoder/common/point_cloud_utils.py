import torch
import torch.nn as nn
import pytorch3d.ops as torch3d_ops


def fps(points, num_points=1024):
    B, _, C = points.shape
    if C == 3:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points, K=num_points)
        sampled_points = sampled_points.contiguous()
    elif C > 3:
        _, indices = torch3d_ops.sample_farthest_points(points=points[..., :3], K=num_points)
        sampled_points = points[torch.arange(B).unsqueeze(1), indices]
    return sampled_points, indices


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class FPS_kNN(nn.Module):

    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, feat):
        """
        Input:
            xyz: point cloud coordinates, [B, N, C]
            feat: point features, [B, N, D]
            group_num: number of sampled centroids
            k_neighbors: max number of kNN points for each centroid
        Return:
            lc_xyz: local centroid coordinates, [B, group_num, C]
            lc_feat: local centroid features, [B, group_num, D]
            knn_xyz: kNN coordinates for each local centroid, [B, group_num, k_neighbors, C]
            knn_feat: kNN features for each local centroid, [B, group_num, k_neighbors, D]
            fps_idx: indices of sampled centroids, [B, group_num]
            knn_idx: indices of kNN points for each centroid, [B, group_num, k_neighbors]
        """
        B, N, _ = xyz.shape
        _, fps_idx = torch3d_ops.sample_farthest_points(points=xyz, K=self.group_num)

        lc_xyz = index_points(xyz, fps_idx)
        lc_feat = index_points(feat, fps_idx)

        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_feat = index_points(feat, knn_idx)
       
        return lc_xyz, lc_feat, knn_xyz, knn_feat


class FPS_kNN_pytorch3d(nn.Module):

    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, feat):
        lc_xyz, fps_idx = torch3d_ops.sample_farthest_points(
            points=xyz, K=self.group_num, random_start_point=False
        )
        lc_feat = feat.gather(1, fps_idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1]))  # [B, G, C]
        knn = torch3d_ops.knn_points(p1=lc_xyz, p2=xyz, K=self.k_neighbors, return_nn=False)
        knn_idx = knn.idx

        knn_xyz = torch3d_ops.knn_gather(xyz, knn_idx)
        knn_feat   = torch3d_ops.knn_gather(feat, knn_idx)

        return lc_xyz, lc_feat, knn_xyz, knn_feat