import argparse
import os
import sys


REF = """
If you use SynthStrip in your analysis, please cite:
----------------------------------------------------
SynthStrip: Skull-Stripping for Any Brain Image
A Hoopes, JS Mora, AV Dalca, B Fischl, M Hoffmann
NeuroImage 206 (2022), 119474
https://doi.org/10.1016/j.neuroimage.2022.119474

Website: https://synthstrip.io
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robust, universal skull-stripping for brain images of any type."
    )
    parser.add_argument("-i", "--image", metavar="FILE", required=True, help="input image to skullstrip")
    parser.add_argument("-o", "--out", metavar="FILE", help="save stripped image to file")
    parser.add_argument("-m", "--mask", metavar="FILE", help="save binary brain mask to file")
    parser.add_argument("-d", "--sdt", metavar="FILE", help="save distance transform to file")
    parser.add_argument("-g", "--gpu", action="store_true", help="use the GPU")
    parser.add_argument(
        "-b",
        "--border",
        default=1,
        type=float,
        help="mask border threshold in mm, defaults to 1",
    )
    parser.add_argument("-t", "--threads", type=int, help="PyTorch CPU threads, PyTorch default if unset")
    parser.add_argument("-f", "--fill", type=float, help="BG fill value, defaults to min(image.min, 0)")
    parser.add_argument("--no-csf", action="store_true", help="exclude CSF from brain border")
    parser.add_argument("--model", metavar="FILE", help="alternative model weights")
    parser.add_argument("-v", "--version", action="store_true", help="print SynthStrip version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_args = sys.argv[1:] if argv is None else list(argv)

    # Do not wait for third-party imports just to show usage/version.
    if len(raw_args) == 0 or "-h" in raw_args or "--help" in raw_args:
        parser.print_help()
        print(REF)
        return 1
    if "-v" in raw_args or "--version" in raw_args:
        print(os.environ.get("SYNTHSTRIP_VERSION", "unknown"))
        return 1

    args = parser.parse_args(raw_args)

    import numpy as np
    import surfa as sf
    import torch
    import torch.nn as nn

    # Sanity check on the outputs.
    if not args.out and not args.mask and not args.sdt:
        sf.system.fatal("Must provide at least one -o, -m, or -d output flag.")

    # Necessary for speed gains in the original script.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Configure device.
    if args.gpu:
        if not torch.cuda.is_available():
            sf.system.fatal("-g flag provided but CUDA is not available")
        device = torch.device("cuda")
        device_name = "GPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    def extend_sdt(sdt, border=1):
        """Extend SynthStrip's narrow-band signed distance transform (SDT)."""
        if border < int(sdt.max()):
            return sdt

        # Find bounding box.
        mask = sdt < 1
        keep = np.nonzero(mask)
        low = np.min(keep, axis=-1)
        upp = np.max(keep, axis=-1)

        # Add requested border.
        gap = int(border + 0.5)
        low = (max(i - gap, 0) for i in low)
        upp = (min(i + gap, d - 1) for i, d in zip(upp, mask.shape))

        # Compute EDT within bounding box. Keep interior values.
        ind = tuple(slice(a, b + 1) for a, b in zip(low, upp))
        out = np.full_like(sdt, fill_value=100)
        out[ind] = sf.Volume(mask[ind]).distance()
        out[keep] = sdt[keep]

        return sdt.new(out)

    print(f"Configuring model on the {device_name}")

    class StripModel(nn.Module):
        def __init__(
            self,
            nb_features=16,
            nb_levels=7,
            feat_mult=2,
            max_features=64,
            nb_conv_per_level=2,
            max_pool=2,
            return_mask=False,
        ):
            super().__init__()

            ndims = 3

            # Build feature list automatically.
            if isinstance(nb_features, int):
                if nb_levels is None:
                    raise ValueError("must provide unet nb_levels if nb_features is an integer")
                feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
                feats = np.clip(feats, 1, max_features)
                nb_features = [
                    np.repeat(feats[:-1], nb_conv_per_level),
                    np.repeat(np.flip(feats), nb_conv_per_level),
                ]
            elif nb_levels is not None:
                raise ValueError("cannot use nb_levels if nb_features is not an integer")

            # Extract any surplus (full resolution) decoder convolutions.
            enc_nf, dec_nf = nb_features
            nb_dec_convs = len(enc_nf)
            final_convs = dec_nf[nb_dec_convs:]
            dec_nf = dec_nf[:nb_dec_convs]
            self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

            if isinstance(max_pool, int):
                max_pool = [max_pool] * self.nb_levels

            # Cache downsampling / upsampling operations.
            max_pooling = getattr(nn, f"MaxPool{ndims}d")
            self.pooling = [max_pooling(s) for s in max_pool]
            self.upsampling = [nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool]

            # Configure encoder (down-sampling path).
            prev_nf = 1
            encoder_nfs = [prev_nf]
            self.encoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = enc_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.encoder.append(convs)
                encoder_nfs.append(prev_nf)

            # Configure decoder (up-sampling path).
            encoder_nfs = np.flip(encoder_nfs)
            self.decoder = nn.ModuleList()
            for level in range(self.nb_levels - 1):
                convs = nn.ModuleList()
                for conv in range(nb_conv_per_level):
                    nf = dec_nf[level * nb_conv_per_level + conv]
                    convs.append(ConvBlock(ndims, prev_nf, nf))
                    prev_nf = nf
                self.decoder.append(convs)
                if level < (self.nb_levels - 1):
                    prev_nf += encoder_nfs[level]

            # Remaining convolutions at full resolution.
            self.remaining = nn.ModuleList()
            for nf in final_convs:
                self.remaining.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf

            if return_mask:
                self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
                self.remaining.append(nn.Softmax(dim=1))
            else:
                self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

        def forward(self, x):
            x_history = [x]
            for level, convs in enumerate(self.encoder):
                for conv in convs:
                    x = conv(x)
                x_history.append(x)
                x = self.pooling[level](x)

            for level, convs in enumerate(self.decoder):
                for conv in convs:
                    x = conv(x)
                if level < (self.nb_levels - 1):
                    x = self.upsampling[level](x)
                    x = torch.cat([x, x_history.pop()], dim=1)

            for conv in self.remaining:
                x = conv(x)

            return x

    class ConvBlock(nn.Module):
        def __init__(self, ndims, in_channels, out_channels, stride=1, activation="leaky"):
            super().__init__()

            conv = getattr(nn, f"Conv{ndims}d")
            self.conv = conv(in_channels, out_channels, 3, stride, 1)
            if activation == "leaky":
                self.activation = nn.LeakyReLU(0.2)
            elif activation is None:
                self.activation = None
            else:
                raise ValueError(f"Unknown activation: {activation}")

        def forward(self, x):
            out = self.conv(x)
            if self.activation is not None:
                out = self.activation(out)
            return out

    with torch.no_grad():
        model = StripModel()
        model.to(device)
        model.eval()

    # Load model weights.
    if args.model is not None:
        modelfile = args.model
        print("Using custom model weights")
    else:
        version = "1"
        print(f"Running SynthStrip model version {version}")
        fshome = os.environ.get("FREESURFER_HOME")
        if fshome is None:
            sf.system.fatal("FREESURFER_HOME env variable must be set! Make sure FreeSurfer is properly sourced.")
        if args.no_csf:
            print("Excluding CSF from brain boundary")
            modelfile = os.path.join(fshome, "models", f"synthstrip.nocsf.{version}.pt")
        else:
            modelfile = os.path.join(fshome, "models", f"synthstrip.{version}.pt")

    checkpoint = torch.load(modelfile, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image = sf.load_volume(args.image)
    print(f"Input image read from: {args.image}")

    # Loop over frames to keep memory lower.
    print(f"Processing frame (of {image.nframes}):", end=" ", flush=True)
    dist = []
    mask = []
    for frame_num in range(image.nframes):
        print(frame_num + 1, end=" ", flush=True)
        frame = image.new(image.framed_data[..., frame_num])

        # Conform, fit to shape with factors of 64.
        conformed = frame.conform(voxsize=1.0, dtype="float32", method="nearest", orientation="LIA")
        conformed = conformed.crop_to_bbox()
        target_shape = np.clip(np.ceil(np.array(conformed.shape[:3]) / 64).astype(int) * 64, 192, 320)
        conformed = conformed.reshape(target_shape)

        # Normalize.
        conformed -= conformed.min()
        conformed = (conformed / conformed.percentile(99)).clip(0, 1)
        inp = torch.from_numpy(conformed.data[np.newaxis, np.newaxis]).to(device)

        # Predict SDT.
        with torch.no_grad():
            sdt = model(inp).squeeze().cpu()

        # Extend SDT if needed, unconform.
        sdt = extend_sdt(conformed.new(sdt), border=args.border)
        sdt = sdt.resample_like(image, fill=100)
        dist.append(sdt)

        # Extract mask, keep largest connected component.
        mask.append((sdt < args.border).connected_component_mask(k=1, fill=True))

    dist = sf.stack(dist)
    mask = sf.stack(mask)
    print("done")

    if args.out:
        fill = np.min([image.min(), 0]) if args.fill is None else args.fill
        image[mask == 0] = fill
        image.save(args.out)
        print(f"Set background to: {fill}")
        print(f"Masked image saved to: {args.out}")

    if args.mask:
        image.new(mask).save(args.mask)
        print(f"Binary brain mask saved to: {args.mask}")

    if args.sdt:
        image.new(dist).save(args.sdt)
        print(f"Distance transform saved to: {args.sdt}")

    print(REF)
    return 0

