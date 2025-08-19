import torch
import os
import functools
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DecoderOutput
from .utils import (
    get_group,
    get_world_size,
    get_rank,
    init_parallel_vae_mesh,
)
from xfuser.logger import init_logger
logger = init_logger(__name__)


# def parallelize_vae(vae: AutoencoderKLWan, rank: int = 0):
def parallelize_vae(vae: AutoencoderKLWan):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    @functools.wraps(vae.__class__._decode)
    @torch.inference_mode
    def new__decode(
            self,
            z: torch.Tensor,
            *args,
            return_dict: bool = True,
            **kwargs,
    ):
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12
        self.spatial_compression_ratio = 8
        self.temporal_compression_ratio = 4

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        count = 0
        rows = []
        requests = []
        for j in range(0, height, tile_latent_stride_height):
            row = []
            for k in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                iter_ = num_frames
                if count % world_size == rank:
                    for i in range(iter_):
                        self._conv_idx = [0]
                        tile = z[:, :, i: i + 1, j: j + tile_latent_min_height, k: k + tile_latent_min_width]
                        tile = self.post_quant_conv(tile)
                        decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                        time.append(decoded)
                    decoded = torch.cat(time, dim=2)
                    # self.clear_cache()
                    if rank != 0:
                        request = dist.isend(decoded, 0)
                        requests.append(request)
                else:
                    decoded = None
                    if rank == 0:
                        cur_height = min(tile_latent_min_height, (height - j)) * self.spatial_compression_ratio
                        cur_width = min(tile_latent_min_width, (width - k)) * self.spatial_compression_ratio
                        cur_num_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
                        shape = (batch_size, 3, cur_num_frames, cur_height, cur_width)
                        decoded = torch.empty(shape, device=z.device, dtype=z.dtype)
                        request = dist.irecv(decoded, count % world_size)
                        jj, kk = j // tile_latent_stride_height, k // tile_latent_stride_width
                        requests.append((request, decoded, jj, kk))
                row.append(decoded)
                count += 1
            rows.append(row)
        self.clear_cache()

        for request in requests:
            if rank == 0:
                req, tensor, i, j = request
                req.wait()
                rows[i][j] = tensor
            else:
                request.wait()

        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))
            dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        else:
            dec = torch.zeros(batch_size, 3, num_frames, sample_height, sample_width, device=z.device, dtype=z.dtype)

        if not return_dict:
            return (dec,)
        return DecoderOutput(dec, dec)

    vae._decode = new__decode.__get__(vae)



    @functools.wraps(vae.__class__._encode)
    @torch.inference_mode
    def new__encode(
            self,
            x: torch.Tensor,
            *args,
            **kwargs,
    ):
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.spatial_compression_ratio = 8
        self.temporal_compression_ratio = 4

        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        if hasattr(self, "tile_sample_min_height"):
            tile_sample_min_height = self.tile_sample_min_height
        else:
            tile_sample_min_height = self.tile_sample_min_size

        if hasattr(self, "tile_sample_min_width"):
            tile_sample_min_width = self.tile_sample_min_width
        else:
            tile_sample_min_width = self.tile_sample_min_size

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        count = 0
        rows = []
        requests = []
        dtype = None
        for j in range(0, height, self.tile_sample_stride_height):
            row = []
            for k in range(0, width, self.tile_sample_stride_width):
                if count % world_size == rank:
                    tile = x[:, :, :, j: j + tile_sample_min_height, k: k + tile_sample_min_width]
                    self.clear_cache()
                    t = tile.shape[2]
                    iter_ = 1 + (t - 1) // 4
                    for i in range(iter_):
                        self._enc_conv_idx = [0]
                        if i == 0:
                            out = self.encoder(
                                tile[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                            )
                        else:
                            out_ = self.encoder(
                                tile[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx,
                            )
                            out = torch.cat([out, out_], 2)

                    enc = self.quant_conv(out)
                    dtype = out.dtype
                    # mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
                    # enc = torch.cat([mu, logvar], dim=1)
                    # self.clear_cache()
                    # tile = self.encoder(tile)
                    if rank != 0:
                        req = dist.isend(enc, 0)
                        requests.append(req)
                        pass
                else:
                    enc = None
                    if rank == 0:
                        cur_height = min(tile_latent_min_height, (height - j) // self.spatial_compression_ratio)
                        cur_width = min(tile_latent_min_width, (width - k) // self.spatial_compression_ratio)
                        cur_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1
                        shape = (batch_size, 32, cur_num_frames, cur_height, cur_width)
                        enc = torch.empty(shape, device=device, dtype=dtype)
                        req = dist.irecv(enc, count % world_size)
                        requests.append((req, enc))
                        pass
                row.append(enc)
                count += 1
            rows.append(row)
        self.clear_cache()

        count = 0
        for jj in range(len(rows)):
            for kk in range(len(rows[jj])):
                if count % world_size == rank:
                    if rank != 0:
                        req = requests.pop(0)
                        req.wait()
                else:
                    if rank == 0:
                        req, tensor = requests.pop(0)
                        req.wait()
                        rows[jj][kk] = tensor
                count += 1

        requests_send = []
        requests_rev = []
        if rank == 0:
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = blend_v(rows[i - 1][j], tile, blend_height)
                    if j > 0:
                        tile = blend_h(row[j - 1], tile, blend_width)
                    result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
                result_rows.append(torch.cat(result_row, dim=-1))

            enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
            req = dist.isend(enc, rank + 1)
            requests_send.append(req)
        else:
            cur_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1
            shape = (batch_size, 32, cur_num_frames, latent_height, latent_width)
            enc = torch.empty(shape, device=device, dtype=dtype)
            req = dist.irecv(enc, rank - 1)
            requests_rev.append((req, enc))

        if rank == 0:
            req = requests_send.pop(0)
            req.wait()
        elif rank < world_size - 1:
            req, tensor = requests_rev.pop(0)
            req.wait()
            enc = tensor
            req = dist.isend(enc, rank + 1)
            req.wait()
        elif rank == world_size - 1:
            req, tensor = requests_rev.pop(0)
            req.wait()
            enc = tensor

        return enc

    vae._encode = new__encode.__get__(vae)

    return vae


