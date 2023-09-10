#! /usr/bin/env sh
export version=v2.4.2

export ai_toolchain_package_path=/home/leo/Downloads/ai_toolchain

export dataset_path=/home/leo/Downloads/dataset

docker run -it \
  -v "$ai_toolchain_package_path":/open_explorer \
  -v "$dataset_path":/data \
  openexplorer/ai_toolchain_centos_7_xj3:"${version}"
