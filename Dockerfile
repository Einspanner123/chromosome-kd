# 使用包含 CUDA 11.8 的 PyTorch 官方镜像作为基础
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# 设置工作目录
WORKDIR /app

# 设置环境变量，确保非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 1. 配置 APT 国内源 (使用阿里云)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装系统依赖（OpenMMLab 家族通常需要的库）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 配置 Conda 国内源 (使用清华源)
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --set show_channel_urls yes

# 3. 配置 Pip 国内源 (使用阿里云)
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 复制 Conda 环境配置文件
COPY environment.yml .

# 使用 Conda 根据配置文件更新 base 环境
# 这样镜像启动后默认就是配置好的环境，无需额外 activate
RUN conda env update -n base -f environment.yml && \
    conda clean -afy

# 复制当前项目代码到容器中
COPY . .

# 设置默认启动命令（可根据需要修改）
CMD ["python", "tools/train.py", "projects/LDMDet/configs/ldmdet_rf_heun_shifted_bs2_optimized.py"]
