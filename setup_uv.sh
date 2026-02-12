# Auto-install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[SETUP] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup_uv.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false


if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup_uv.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new virtual environment (.venv)"
    echo "  --basic                 Install basic dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    return
fi

# Get system information
WORKDIR=$(pwd)
if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    return 1
fi

if [ "$NEW_ENV" = true ] ; then
    uv venv --python 3.10 .venv
    source .venv/bin/activate
    if [ "$PLATFORM" = "cuda" ] ; then
        # Auto-detect system CUDA version
        if [ -x /usr/local/cuda/bin/nvcc ]; then
            SYSTEM_CUDA=$((/usr/local/cuda/bin/nvcc --version 2>/dev/null || true) | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/')
        elif command -v nvcc &> /dev/null; then
            SYSTEM_CUDA=$((nvcc --version 2>/dev/null || true) | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/')
        elif command -v nvidia-smi &> /dev/null; then
            SYSTEM_CUDA=$((nvidia-smi 2>/dev/null || true) | grep "CUDA Version" | sed 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/')
        else
            SYSTEM_CUDA=""
        fi

        SYSTEM_CUDA_MAJOR=$(echo $SYSTEM_CUDA | cut -d'.' -f1)
        SYSTEM_CUDA_MINOR=$(echo $SYSTEM_CUDA | cut -d'.' -f2)

        if [ "$SYSTEM_CUDA_MAJOR" = "12" ] && [ "$SYSTEM_CUDA_MINOR" -ge 4 ] 2>/dev/null; then
            TORCH_INDEX="cu124"
        elif [ "$SYSTEM_CUDA_MAJOR" = "12" ]; then
            TORCH_INDEX="cu121"
        elif [ "$SYSTEM_CUDA_MAJOR" = "11" ]; then
            TORCH_INDEX="cu118"
        else
            echo "[WARNING] Could not detect system CUDA version (got: '$SYSTEM_CUDA'). Defaulting to cu124."
            TORCH_INDEX="cu124"
        fi

        echo "[SETUP] System CUDA: $SYSTEM_CUDA -> using PyTorch index: $TORCH_INDEX"
        uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/$TORCH_INDEX
    elif [ "$PLATFORM" = "hip" ] ; then
        uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
    fi
fi

# Set CUDA_HOME for building CUDA extensions
if [ "$PLATFORM" = "cuda" ] && [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

if [ "$BASIC" = true ] ; then
    uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh "transformers<5" gradio==6.0.1 tensorboard pandas lpips zstandard
    uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    sudo apt install -y libjpeg-dev
    uv pip install pillow-simd
    uv pip install kornia timm
fi

if [ "$FLASHATTN" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        uv pip install psutil
        uv pip install flash-attn==2.7.3 --no-build-isolation
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        mkdir -p /tmp/extensions
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 python setup.py install #MI300 series
        cd $WORKDIR
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        uv pip install /tmp/extensions/nvdiffrast --no-build-isolation
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        uv pip install /tmp/extensions/nvdiffrec --no-build-isolation
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    uv pip install /tmp/extensions/CuMesh --no-build-isolation
fi

if [ "$FLEXGEMM" = true ] ; then
    mkdir -p /tmp/extensions
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    uv pip install /tmp/extensions/FlexGEMM --no-build-isolation
fi

if [ "$OVOXEL" = true ] ; then
    mkdir -p /tmp/extensions
    cp -r o-voxel /tmp/extensions/o-voxel
    uv pip install /tmp/extensions/o-voxel --no-build-isolation
fi
