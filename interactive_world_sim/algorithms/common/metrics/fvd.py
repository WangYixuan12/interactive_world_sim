"""FVD

Adopted from https://github.com/cvpr2022-stylegan-v/stylegan-v
Verified to be the same as tf version by https://github.com/universome/fvd-comparison
"""

import html
import io
import re
import urllib
import urllib.request
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import requests
import scipy
import torch
import torch.nn as nn


def get_cached_model_path(model_name: str) -> Path:
    """Get the path for a cached model file"""
    cache_dir = Path(__file__).parent
    return cache_dir / model_name


def open_url(
    url: str,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
) -> Any:
    """Download the given URL and return a binary-mode file object to access the data"""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- \
                                    please try again later"
                            )

                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:  # noqa
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)  # type: ignore


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert feats.ndim == 2, f"Expected 2D array, got {feats.ndim}D"
    if feats.shape[0] == 1:
        feats = np.concatenate([feats, feats], axis=0)
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


class FrechetVideoDistance(nn.Module):
    """Frechet Video Distance (FVD) metric"""

    def __init__(self) -> None:
        super().__init__()
        detector_url = (
            "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
        )
        # Return raw features before the softmax layer.
        self.detector_kwargs = dict(rescale=False, resize=True, return_features=True)

        # Check if model is cached
        cached_model_path = get_cached_model_path("i3d_torchscript.pt")

        if cached_model_path.exists():
            # Load from cache
            self.detector = torch.jit.load(str(cached_model_path)).eval()
        else:
            # Download and cache the model
            print("Downloading I3D detector model...")
            with open_url(detector_url, verbose=False) as f:
                model_data = f.read()

            # Save to cache
            with open(cached_model_path, "wb") as cache_file:
                cache_file.write(model_data)

            # Load from the saved file
            self.detector = torch.jit.load(str(cached_model_path)).eval()
            print(f"Model cached at: {cached_model_path}")

    @torch.no_grad()
    def compute(self, videos_fake: torch.Tensor, videos_real: torch.Tensor) -> float:
        """Compute FVD between fake and real videos

        :param videos_fake: predicted video tensor of shape
            (frame, batch, channel, height, width)
        :param videos_real: ground-truth observation tensor of shape
            (frame, batch, channel, height, width)
        :return:
        """
        n_frames, batch_size, c, h, w = videos_fake.shape
        if n_frames < 2:
            raise ValueError("Video must have more than 1 frame for FVD")

        videos_fake = videos_fake.permute(1, 2, 0, 3, 4).contiguous()
        videos_real = videos_real.permute(1, 2, 0, 3, 4).contiguous()

        # detector takes in tensors of shape [batch_size, c, video_len, h, w]
        # with range -1 to 1
        feats_fake = (
            self.detector(
                videos_fake.to(next(self.parameters()).device), **self.detector_kwargs
            )
            .cpu()
            .numpy()
        )
        feats_real = (
            self.detector(
                videos_real.to(next(self.parameters()).device), **self.detector_kwargs
            )
            .cpu()
            .numpy()
        )

        try:
            fvd = compute_fvd(feats_fake, feats_real)
        except ValueError:
            fvd = 1e5
        return fvd
