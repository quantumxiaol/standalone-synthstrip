#!/usr/bin/env bash
source "$(dirname $0)/../test.sh"

t() { test_command mri_synthstrip "$@" ; }

# skull-stripped image
t -i in.mgz -o out.mgz
compare_vol out.mgz stripped.mgz

# GPU flags
t --gpu -g -i in.mgz --out out.mgz
compare_vol out.mgz stripped.mgz

# CPU threads
t --threads 4 -t 4 -i in.mgz --out out.mgz
compare_vol out.mgz stripped.mgz

# binary mask
t --image in.mgz -m out.mgz
compare_vol out.mgz mask.mgz

# distance transform without output validation
t -i in.mgz -d out.mgz
t -i in.mgz --sdt out.mgz

# default border value
t -b 1 -i in.mgz -m out.mgz
compare_vol out.mgz mask.mgz

# increased border
t --border 2 -i in.mgz -m out.mgz
compare_vol out.mgz border.mgz

# large border with SDT extension
t -b 8 -i in.mgz --mask out.mgz
compare_vol out.mgz large.mgz

# multiple frames, NIfTI format
t -i multi.in.nii.gz -m out.nii.gz
compare_vol out.nii.gz multi.mask.nii.gz

# large FOV without `torch.quantile` error
t -i fov320.in.nii.gz -m out.nii.gz

# explicit fill value
t -i in.mgz -o out.mgz -f 255 --fill 255
mris_calc out.mgz stats 2>&1 | grep 'Max.*255\.'
