import math
from typing import List, Union
import torch.distributed as dist
if dist.is_available():
    import torch.distributed.distributed_c10d as c10d
else:
    ft_c = None
    c10d = None


def get_group(group=None):
    if group is None:
        group = c10d._get_default_group()

    if isinstance(group, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = group
    else:
        pg = group.get_group()

    return pg


def get_world_size(group=None):
    pg = get_group(group)
    return dist.get_world_size(pg)


def get_rank(group=None):
    pg = get_group(group)
    return dist.get_rank(pg)


def init_parallel_vae_mesh(device_type=None, *, mesh=None):
    if mesh is not None:
        return mesh

    assert device_type is not None, "device must be provided if mesh is not provided"

    world_size = get_world_size()

    return dist.init_device_mesh(device_type, (world_size,))


def init_context_parallel_mesh(
    device_type=None, *, mesh=None, max_batch_dim_size=None, max_ring_dim_size=None, max_ulysses_dim_size=None
):
    if mesh is not None:
        return mesh

    assert device_type is not None, "device must be provided if mesh is not provided"

    world_size = get_world_size()
    if max_batch_dim_size is None:
        batch_dim_size = 1
    else:
        batch_dim_size = math.gcd(world_size, max_batch_dim_size)

    attn_world_size = world_size // batch_dim_size

    assert not (
        max_ring_dim_size is not None and max_ulysses_dim_size is not None
    ), "Only one of max_ulysses_dim_size and max_ring_dim_size can be set"

    if max_ulysses_dim_size is None:
        if max_ring_dim_size is None:
            ring_dim_size = 1
        else:
            ring_dim_size = math.gcd(attn_world_size, max_ring_dim_size)
        ulysses_dim_size = attn_world_size // ring_dim_size
    else:
        ulysses_dim_size = math.gcd(attn_world_size, max_ulysses_dim_size)
        ring_dim_size = attn_world_size // ulysses_dim_size

    mesh_shape = (batch_dim_size, ring_dim_size, ulysses_dim_size)
    return dist.init_device_mesh(device_type, mesh_shape, mesh_dim_names=("batch", "ring", "ulysses"))
