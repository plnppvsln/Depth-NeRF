# losses/sparse_depth_loss.py
import torch

def local_depth_ranking_loss(rendered_depth, dpt_depth, coords, margin=1e-4, n_pairs=128):
    """
    rendered_depth: [N,] — NeRF depth for N sampled rays
    dpt_depth: [H, W] — numpy array from DPT (inverse depth)
    coords: [N, 2] — pixel coordinates (y, x) for each ray
    """
    device = rendered_depth.device
    H, W = dpt_depth.shape
    dpt = torch.from_numpy(dpt_depth).to(device)

    loss = 0.0
    count = 0

    for _ in range(n_pairs):
        i, j = torch.randint(0, rendered_depth.shape[0], (2,))
        yi, xi = coords[i]
        yj, xj = coords[j]

        if not (0 <= yi < H and 0 <= xi < W and 0 <= yj < H and 0 <= xj < W):
            continue

        dpt_i = dpt[yi, xi]
        dpt_j = dpt[yj, xj]

        if dpt_i >= dpt_j:  # DPT: i is closer (inverse depth: larger = closer)
            diff = rendered_depth[i] - rendered_depth[j] + margin
            if diff > 0:
                loss += diff
                count += 1

    return loss / max(count, 1)


def spatial_continuity_loss(rendered_depth, dpt_depth, coords, k=4, margin=1e-4):
    """
    rendered_depth: [N,] — NeRF depth for sampled rays
    dpt_depth: [H, W] — numpy array from DPT
    coords: [N, 2] — pixel coordinates (y, x) corresponding to each ray
    """
    device = rendered_depth.device
    H, W = dpt_depth.shape
    dpt = torch.from_numpy(dpt_depth).to(device)

    loss = 0.0
    count = 0

    # Build a mapping from (y, x) to depth index for fast lookup
    coord_to_idx = {}
    for idx, (y, x) in enumerate(coords):
        y, x = int(y), int(x)
        if 0 <= y < H and 0 <= x < W:
            coord_to_idx[(y, x)] = idx

    for idx, (y, x) in enumerate(coords):
        y, x = int(y), int(x)
        if not (0 <= y < H and 0 <= x < W):
            continue

        center_val = dpt[y, x]
        center_rd = rendered_depth[idx]

        # Find k nearest neighbors by DPT value (within a local region)
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if (ny, nx) in coord_to_idx:
                    neighbors.append(coord_to_idx[(ny, nx)])

        if len(neighbors) < 2:
            continue

        # Sort neighbors by DPT value difference
        neighbor_vals = [dpt[int(coords[n][0]), int(coords[n][1])] for n in neighbors]
        sorted_pairs = sorted(zip(neighbor_vals, neighbors), key=lambda x: abs(x[0] - center_val))
        selected_neighbors = [n for _, n in sorted_pairs[:k]]

        for n in selected_neighbors:
            diff = torch.abs(center_rd - rendered_depth[n]) - margin
            if diff > 0:
                loss += diff
                count += 1

    return loss / max(count, 1)