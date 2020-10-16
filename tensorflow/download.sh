#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Download from nvidia packages to build GPU containers for balena
NVIDIA_URL="https://developer.nvidia.com/embedded/r32-2-3_Release_v1.0/t186ref_release_aarch64/Tegra186_Linux_R32.2.3_aarch64.tbz2"
CUDA_URL="https://storage.googleapis.com/autonomy-vision/packages/jetpack_4.3_xavier/cudnn/"
CUDA_ARM64="https://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/JETPACK_422_b21/cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb"
FILE_LIST=(
    "cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb"
    "libcudnn7_7.6.3.28-1%2Bcuda10.0_arm64.deb"
    "libcudnn7-dev_7.6.3.28-1%2Bcuda10.0_arm64.deb"
)
URL_LIST=()
TEGRA_DRIVERS="Tegra186_Linux_R32.2.3_aarch64.tbz2"
URL_LIST+=${NVIDIA_URL}
for i in ${FILE_LIST[@]}; do
    if [ ${i} = ${FILE_LIST[0]} ];  
    then
        URL_LIST+=("$CUDA_ARM64")
    else
        URL_LIST+=("$CUDA_URL$i")
    fi
done
printf "%s\n" "${URL_LIST[@]}"
echo "downloading files ..."

# Download in parallel all packages, if exits then continue
echo ${URL_LIST[@]} | sed 's/\*\*/ -P /g' | xargs -n 1 -P 8 wget -nc -q

# TODO(davidnet): Add to get untar to tmp files and get the all the packages from the Tegra186 folder
# Create temporal folder
TMPDIR=$(mktemp -d)
cp $TEGRA_DRIVERS $TMPDIR
pushd $TMPDIR
echo "temporal directory:" $TMPDIR

tar --strip-components=2 -xvf $TEGRA_DRIVERS --wildcards --no-anchored 'Linux_for_Tegra/nv_tegra/*.tbz2'
rm -rf $TEGRA_DRIVERS
mv nv_sample_apps/nvgstapps.tbz2 .
rm -rf nv_sample_apps
popd

mv $TMPDIR/* .

echo "L4T files downloaded"