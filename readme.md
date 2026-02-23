# SynthStrip

Source code for the SynthStrip skull-stripping tool. For documentation, visit [synthstrip.io](https://synthstrip.io).

## Local environment (uv)

This repository is extracted from `freesurfer/dev/mri_synthstrip` and now uses a standard `src/` Python package layout while keeping the original CLI behavior.

### Python compatibility

- `Python >=3.10,<3.13`
- `numpy>=1.26,<3`
- `torch==2.1.2` on Python `<3.12`
- `torch>=2.2,<3` on Python `>=3.12`
- `surfa` pinned to the same commit used by the upstream Dockerfiles

### Setup

```shell
uv lock
# or
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync
source .venv/bin/activate
export FREESURFER_HOME="$(pwd)"
```

### Install as a package

```shell
uv sync
uv run mri_synthstrip -h
```

The command is exposed through `pyproject.toml` as a console script. A compatibility wrapper `./mri_synthstrip` is also kept for direct execution.

### Model files

`mri_synthstrip` needs model weights (`synthstrip.1.pt` or `synthstrip.nocsf.1.pt`).

- Default path: `$FREESURFER_HOME/models/`
- Or provide `--model /path/to/weights.pt`

If this repo was produced by `git filter-repo`, the checked-in `synthstrip*.pt` files may still be `git-annex` symlinks whose targets are missing. In that case, fetch real model files before running inference.

### CLI usage

```shell
mri_synthstrip -i input.nii.gz -m mask.nii.gz
mri_synthstrip -i input.nii.gz -o stripped.nii.gz
mri_synthstrip -i input.nii.gz -d sdt.nii.gz
mri_synthstrip --no-csf -i input.nii.gz -m mask.nii.gz
mri_synthstrip --model /path/to/synthstrip.1.pt -i input.nii.gz -m mask.nii.gz
```

Run `mri_synthstrip -h` for all options.

## Building containers

The `mri_synthstrip` command is automatically built into FreeSurfer.
To create a standalone SynthStrip container, first fetch the latest model files and unlock them to replace the `git-annex` symlinks:

```shell
git fetch datasrc
git annex get .
git annex unlock synthstrip.*.pt
```

Then build and push the container to the [Docker Hub](https://hub.docker.com/u/freesurfer), tagging the new version appropriately:

```shell
VERSION=X.X
docker build -f Dockerfile.cpu -t freesurfer/synthstrip:${VERSION} .
docker push freesurfer/synthstrip:${VERSION}
```

Remember to point the default "latest" tag to the new container.

```shell
docker tag freesurfer/synthstrip:${VERSION} freesurfer/synthstrip:latest
docker push freesurfer/synthstrip:latest
```

Finally, update the version reference in the `synthstrip-docker` and `synthstrip-singularity` scripts:

```shell
sed -i "s/^\(version = \).*/\1'$VERSION'/" synthstrip-docker
sed -i "s/\(synthstrip.\)[0-9.]*\(\.\|$\)/\1$VERSION\2/g" synthstrip-singularity
git diff synthstrip-*
```

Lock the model files again.

```shell
git annex lock synthstrip.*.pt
```

## Exporting requirements

Export updated requirement files as build artifacts for users who wish to build environments:

```shell
docker build -f Dockerfile.cpu --target export --output env .
```
