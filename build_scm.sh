#!/bin/bash
set -e

# 默认构建 xfuser 的 wheel 包
# 如果指定了 CUSTOM_TOS_AK 和 CUSTOM_TOS_SK，则上传 wheel 包到 tos
# 如果指定了 CUSTOM_DOCKER_USERNAME 和 CUSTOM_DOCKER_PASSWORD，则构建 docker 镜像并推送到 dockerhub
# - 镜像内默认使用本次构建出的 wheel 包，如果指定了 CUSTOM_XFUSER_VERSION，则使用指定的 wheel 包

ROOT_DIR=$(pwd)
PYTHON=python3
PIP="$PYTHON -m pip"
BUILD_TIME=$(date +%Y%m%d%H%M)
# 获取当前分支名，并将特殊字符转换为下划线
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
echo "BRANCH_NAME: $BRANCH_NAME"

# 如果分支是以 release_ 或 release/ 开头，则将 release_ 或 release/ 替换为空
if [[ $BRANCH_NAME =~ ^release[\/_] ]]; then
    BRANCH_NAME=${BRANCH_NAME#release}
    BRANCH_NAME=${BRANCH_NAME#/}
    BRANCH_NAME=${BRANCH_NAME#_}
    # 如果分支里还有 / ，则将 / 替换为 .
    BRANCH_NAME=${BRANCH_NAME//\//.}
    if [[ ! -z $BRANCH_NAME ]]; then
        BRANCH_NAME=.${BRANCH_NAME}
    fi
    VERSION_SUFFIX=+byted${BRANCH_NAME}.${BUILD_TIME}
else
    VERSION_SUFFIX=+byted.${BUILD_TIME}
fi

echo "VERSION_SUFFIX: $VERSION_SUFFIX"

TOS_UTIL_URL=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi

VERSION=$(sed -n 's/^__version__\s*=\s*"\([^"]*\)"/\1/p' xfuser/__version__.py)

XFUSER_WHEEL_VERSION=$VERSION$VERSION_SUFFIX
echo "Building xfuser version $XFUSER_WHEEL_VERSION"

xfuser_version_bk=xfuser/__version__.py.bk
cp xfuser/__version__.py $xfuser_version_bk

sed -i "s|^__version__\s*=\s*\"\([^\"]*\)\"|__version__ = \"$XFUSER_WHEEL_VERSION\"|" xfuser/__version__.py

if ! command -v build &> /dev/null; then
    apt update && apt install -y python3-pip
    $PIP install build --no-cache-dir --break-system-packages
fi

$PYTHON -m build

OUTPUT_PATH=$ROOT_DIR/output
mkdir -p $OUTPUT_PATH
mv dist/* $OUTPUT_PATH/
mv $xfuser_version_bk xfuser/__version__.py

if [ ! -z "$CUSTOM_XFUSER_VERSION" ] || [ -z "$CUSTOM_TOS_AK" ] || [ -z "$CUSTOM_TOS_SK" ]; then
    echo "specified CUSTOM_XFUSER_VERSION or (CUSTOM_TOS_AK or CUSTOM_TOS_SK) are not set, skip uploading to tos"
else
    # 上传制品到 tos
    wget $TOS_UTIL_URL -O tosutil && chmod +x tosutil
    for wheel_file in $(find $OUTPUT_PATH -name "*.whl"); do
        echo "uploading $wheel_file to tos..."
        ./tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/xfuser/$(basename $wheel_file) -re cn-beijing -e tos-cn-beijing.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi

if [ -z "${CUSTOM_DOCKER_USERNAME}" ] || [ -z "${CUSTOM_DOCKER_PASSWORD}" ]; then
    echo "CUSTOM_DOCKER_USERNAME or CUSTOM_DOCKER_PASSWORD is not set, skip building image"
else
    # 如果是 SCM 构建，则准备 docker 环境
    if [[ "${SCM_BUILD}" == "True" ]]; then
        source /root/start_dockerd.sh
    fi
    proxy_args=""
    if [ ! -z "$http_proxy" ]; then
        proxy_args="$proxy_args --build-arg http_proxy=$http_proxy"
    fi
    if [ ! -z "$https_proxy" ]; then
        proxy_args="$proxy_args --build-arg https_proxy=$https_proxy"
    fi
    if [ ! -z "${CUSTOM_XFUSER_VERSION}" ]; then
        xfuse_arg="--build-arg XFUSER_VERSION=${CUSTOM_XFUSER_VERSION}"
    fi
    if [ ! -z "${CUSTOM_WAN_BRANCH}" ]; then
        wan_branch_arg="--build-arg WAN_BRANCH=${CUSTOM_WAN_BRANCH}"
    fi
    if [ ! -z "${CUSTOM_FLA3_COMMIT}" ]; then
        fla3_commit_arg="--build-arg FLA3_COMMIT=${CUSTOM_FLA3_COMMIT}"
    fi

    IMAGE_TAG=$(echo v${XFUSER_WHEEL_VERSION#v} | sed 's/+/./g')
    IMAGE_TAG=$(echo "$IMAGE_TAG" | sed "s/\(.*\.\)[0-9]\+/\1$BUILD_TIME/")  # 替换为当前时戳
    echo "IMAGE_TAG: $IMAGE_TAG"
    TARGET_IMAGE=iaas-gpu-cn-beijing.cr.volces.com/serving/xdit:${IMAGE_TAG}
    docker login -u $CUSTOM_DOCKER_USERNAME -p $CUSTOM_DOCKER_PASSWORD iaas-gpu-cn-beijing.cr.volces.com
    docker buildx build --network=host --push -t $TARGET_IMAGE -f docker/Dockerfile.bd_iaas $xfuse_arg $proxy_args $wan_branch_arg $fla3_commit_arg .
    echo "Pushed image to $TARGET_IMAGE"
    echo ${TARGET_IMAGE} > $OUTPUT_PATH/image_name
fi
