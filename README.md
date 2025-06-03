# roop

这是一个基于叫做roop的国外项目进行改良的汉化项目，更加适合在Linux Ubuntu 20.04的操作系统上进行运行，项目的主要内容为roop作者编写，本项目基于源码的基础上进行修改部分功能以及添加中文日志，下面介绍如何进行部署

## 如何部署

由于roop项目需要gpu进行加速才可以达到工业需求的生产，但是大多数gpu服务器均是Linux Ubuntu 20.04，所以就需要详细的安装步骤使项目可以在系统上运行

### 环境检查

**python环境**

首先项目使用python进行运行所以python必不可少，该版本适应的使python3.10，所以需要安装这个版本，由于不同环境可以隔离不同版本，这里推荐使用conda进行管理python的虚拟空间

```shell
apt-get update 
#更新linux系统工具系统
conda create -n roop_env python=3.10
#使用conda创建项目所需要的虚拟空间
```

完成之后进行重启系统

```shell
conda init
#初始化
conda activate roop_env
#进去python虚拟空间
python
#验证版本
exit()
#退出编译器
```

**cuda环境**

如果使用cuda进行加速，需要提前安装好相关版本

```shell
nvidia-smi
#显示gpu系统硬件信息
nvcc -V
#显示cuda版本
dpkg -l | grep libcudnn
#显示cudnn版本
```

如果都显示出来说明已经安装好cuda环境，roop的这个版本适应于cuda11.8和cudnn8.6版本的环境，最好安装这个版本其他版本可能无法启动cuda加速，网上有安装教程下面仅仅是cudnn的导入指令

```shell
tar -xJvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz 
#解压安装包
cd cudnn-linux-x86_64-8.6.0.163_cuda11-archive
#进入文件夹内部
sudo cp ./lib/* /usr/local/cuda-11.8/lib64/
#修改文件
sudo cp ./include/* /usr/local/cuda-11.8/include/
#修改文件
```

### 项目导入和初始化

**项目导入**

使用git工具将项目进行导入，首先需要进行安装git然后进行导入

```shell
sudo apt install git-all
#安装git工具
git clone https://github.com/Dontlikemushroom/roop.git
#将项目导入在当前文件夹下
```

**项目依赖导入**

```shell
sudo apt install python3-pip
#安装python依赖安装工具
sudo apt install ffmpeg
#安装ffmpeg工具
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
#安装项目适配的torch相关版本
pip install -r requirements.txt
pip install -r requirements-headless.txt
#安装所有其他所需的依赖，前者为正常版本安装后者为无界面版本安装
```

**模型文件导入**

在使用项目的时候会需要下载相关模型，但是有的时候下载速度十分的慢，这个部分就是提取下载好相关的模型然后放入项目中，这样就不需要进行运行的时候进行下载安装

**inswapper_128.onnx模型**

在网上进行搜索相关文件提前下载，然后导入到linux系统中使用一下指令进行放入项目位置

```shell
mv inswapper_128.onnx 项目根目录下/models
#移动文件，如果显示没有文件夹，使用mkdir指令进行创建指令
```

**buffalo_l模型**

在网上进行搜索相关文件提前下载，然后导入到linux系统中使用一下指令进行放入项目位置

```shell
mv buffalo_l/ /root/.insightface/models
#移动文件，如果显示没有文件夹，使用mkdir指令进行创建指令
```

**open_nsfw_weights.h5模型**

在网上进行搜索相关文件提前下载，然后导入到linux系统中使用一下指令进行放入项目位置

```shell
mv open_nsfw_weights.h5 /root/.opennsfw2/weights
#移动文件，如果显示没有文件夹，使用mkdir指令进行创建指令
```

## 如何使用

项目启动主要是使用python运行run.py文件，但是提供了多种附加指令来实现用户需求

```
python run.py [options]

-h, --help                                                                 展示所用使用方式
-s SOURCE_PATH, --source SOURCE_PATH                                       换脸图片路径
-t TARGET_PATH, --target TARGET_PATH                                       作用视频/图片路径
-o OUTPUT_PATH, --output OUTPUT_PATH                                       输出路径
--frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]                    帧处理器选择，有脸部交换和脸部增强
--keep-fps                                                                 是否保持原视频帧率
--keep-frames                                                              是否保存临时帧
--skip-audio                                                               是否跳过目标音频处理
--many-faces                                                               是否处理多个人脸
--reference-face-position REFERENCE_FACE_POSITION                          选择扫描到的第几个人脸进行处理
--reference-frame-number REFERENCE_FRAME_NUMBER                            选择从第几帧开始扫描人脸进行处理
--similar-face-distance SIMILAR_FACE_DISTANCE                              相似度选择，越大识别个数越多，反之越少
--half-face int															   选择只替换从上到下整张脸的百分之几
--temp-frame-format {jpg,png}                                              设置临时帧格式
--temp-frame-quality [0-100]                                               设置临时帧质量
--output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}  设置输出视频编码器有默认、英伟达、amd等
--output-video-quality [0-100]                                             设置输出视频质量，数量越大质量越高
--max-memory MAX_MEMORY                                                    设置最大内存
--execution-provider {cpu} [{cpu} ...]                                     设置计算引擎，有cpu处理以及cuda加速处理
--execution-threads EXECUTION_THREADS                                      设置执行线程数，默认根据计算引擎自选
-v, --version                                                              版本信息
```

下面是一个使用指令的例子

```shell
python run.py -s /root/autodl-tmp/roop/resource/IMG_0765.JPG -t /root/autodl-tmp/roop/resource/2.mp4 -o /root/autodl-tmp/roop/outputs/20250601_1.mp4 --frame-processor face_swapper face_enhancer --keep-fps --output-video-encoder h264_nvenc --output-video-quality 80 --execution-provider cuda
#使用了脸部替换以及脸部增强、保持帧率、使用英伟达进行输出、视频质量为80以及使用cuda加速
```

以下是原作者的项目README内容

## This project has been discontinued

Yes, it still works, you can still use this software. It just won't recieve any updates now.

> I do not have the interest or time to oversee the development of this software. I thank all the amazing people who contributed to this project and made what it is in it's final form.

# Roop

> Take a video and replace the face in it with a face of your choice. You only need one image of the desired face. No dataset, no training.

[![Build Status](https://img.shields.io/github/actions/workflow/status/s0md3v/roop/ci.yml.svg?branch=main)](https://github.com/s0md3v/roop/actions?query=workflow:ci)

<img src="https://i.ibb.co/4RdPYwQ/Untitled.jpg"/>

## Installation

Be aware, the installation needs technical skills and is not for beginners. Please do not open platform and installation related issues on GitHub.

[Basic](https://github.com/s0md3v/roop/wiki/1.-Installation) - It is more likely to work on your computer, but will be quite slow

[Acceleration](https://github.com/s0md3v/roop/wiki/2.-Acceleration) - Unleash the full potential of your CPU and GPU


## Usage

Start the program with arguments:

```
python run.py [options]

-h, --help                                                                 show this help message and exit
-s SOURCE_PATH, --source SOURCE_PATH                                       select an source image
-t TARGET_PATH, --target TARGET_PATH                                       select an target image or video
-o OUTPUT_PATH, --output OUTPUT_PATH                                       select output file or directory
--frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]                    frame processors (choices: face_swapper, face_enhancer, ...)
--keep-fps                                                                 keep target fps
--keep-frames                                                              keep temporary frames
--skip-audio                                                               skip target audio
--many-faces                                                               process every face
--reference-face-position REFERENCE_FACE_POSITION                          position of the reference face
--reference-frame-number REFERENCE_FRAME_NUMBER                            number of the reference frame
--similar-face-distance SIMILAR_FACE_DISTANCE                              face distance used for recognition
--temp-frame-format {jpg,png}                                              image format used for frame extraction
--temp-frame-quality [0-100]                                               image quality used for frame extraction
--output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}  encoder used for the output video
--output-video-quality [0-100]                                             quality used for the output video
--max-memory MAX_MEMORY                                                    maximum amount of RAM in GB
--execution-provider {cpu} [{cpu} ...]                                     available execution provider (choices: cpu, ...)
--execution-threads EXECUTION_THREADS                                      number of execution threads
-v, --version                                                              show program's version number and exit
```


### Headless

Using the `-s/--source`, `-t/--target` and `-o/--output` argument will run the program in headless mode.


## Disclaimer

This software is designed to contribute positively to the AI-generated media industry, assisting artists with tasks like character animation and models for clothing.

We are aware of the potential ethical issues and have implemented measures to prevent the software from being used for inappropriate content, such as nudity.

Users are expected to follow local laws and use the software responsibly. If using real faces, get consent and clearly label deepfakes when sharing. The developers aren't liable for user actions.


## Licenses

Our software uses a lot of third party libraries as well pre-trained models. The users should keep in mind that these third party components have their own license and terms, therefore our license is not being applied.


## Credits

- [deepinsight](https://github.com/deepinsight) for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models.
- all developers behind the libraries used in this project


## Documentation

Read the [documentation](https://github.com/s0md3v/roop/wiki) for a deep dive.
