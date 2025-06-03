#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


# 在parse_args()参数解析后添加中文提示
def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    # 换脸图片路径
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    # 作用视频/图片路径
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    # 输出路径
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    # 帧处理器选择
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    # 是否保持原视频帧率
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    # 是否保存临时帧
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    # 是否跳过目标音频处理
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    # 是否处理多个人脸
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    # 选择扫描到的第几个人脸进行处理
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    # 选择从第几帧开始扫描人脸进行处理
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    # 相似度选择，越大识别个数越多，反之越少默认0.85
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    # 新增：是否只替换上半张脸
    program.add_argument('--half-face', help='swap percentage of upper face (0-10, 0=none, 10=full)', dest='half_face_swap', type=int, choices=range(11), metavar='[0-10]')
    # 设置临时帧格式
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    # 设置临时帧质量
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    # 设置输出视频编码器，可以选择libx264（默认）、h264_nvenc（英伟达显卡编码，也是最好的）、hevc_amf（AMD显卡加速）以及prores_ks（苹果M芯片加速）
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    # 设置输出视频质量，数量越大质量越高，反之越低，默认35
    program.add_argument('--output-video-quality', help='quality used for the output video', dest='output_video_quality', type=int, default=35, choices=range(101), metavar='[0-100]')
    # 设置最大内存
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    # 设置计算引擎，默认使用CPU
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    # 设置执行线程数，默认根据计算引擎自动选择，CPU一般为1，GPU一般为8
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    # 版本信息
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.headless = roop.globals.source_path is not None and roop.globals.target_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processor
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_quality = args.output_video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads
    roop.globals.half_face_swap = args.half_face_swap

    # 输出中文配置信息
    config_output = [
        '\n[运行配置]',
        f'源文件: {args.source_path or "未设置"}',
        f'目标文件: {args.target_path}',
        f'计算引擎: {args.execution_provider}',
        f'帧处理器: {",".join(args.frame_processor)}',
        f'保持帧率: {"是" if args.keep_fps else "否"}'
    ]
    update_status('\n'.join(config_output), 'CONFIG')

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


# 修改pre_check()的提示信息
def pre_check() -> bool:
    # 修改后pre_check函数
    if sys.version_info < (3, 9):
        update_status('Python版本需3.9以上，当前版本: {sys.version}', 'ERROR')
    if not shutil.which('ffmpeg'):
        update_status('未检测到FFmpeg，请访问<https://ffmpeg.org>安装', 'ERROR')
    update_status('环境检测通过! 输入--help查看完整参数', 'SUCCESS')
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)


def start() -> None:
    # 之前就检查过了模型的存在，这个位置是进行检查模型的输出是否存在，即原文件和目标脸文件
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('检测结束开始换脸')
    # 如果输入是图片进行执行，反之跳过函数进行执行下面视频处理过程
    if has_image_extension(roop.globals.target_path):
        if predict_image(roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        # 处理图像帧
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('处理中...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        # 验证图像处理结果
        if is_image(roop.globals.target_path):
            update_status('图片处理成功!')
        else:
            update_status('图片处理失败!')
        return
    # 处理视频文件
    # 进行违规内容检测
    # if predict_video(roop.globals.target_path):
    #     destroy()
    update_status('创建临时资源...')
    create_temp(roop.globals.target_path)
    # 提取视频帧
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'以 {fps} FPS 提取帧中...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('以默认30 FPS提取帧中...')
        extract_frames(roop.globals.target_path)
    # 处理视频帧
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('处理中...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('未找到有效视频帧')
        return
    # 合成输出视频
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'以 {fps} FPS 合成视频...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('以30 FPS合成视频...')
        create_video(roop.globals.target_path)
    # 处理音频
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('跳过音频处理')
    else:
        if roop.globals.keep_fps:
            update_status('恢复原始音频...')
        else:
            update_status('恢复音频（注意：帧率未保持可能影响音画同步）')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    # 清理临时文件
    update_status('清理临时资源...')
    clean_temp(roop.globals.target_path)
    # 验证视频处理结果
    if is_video(roop.globals.target_path):
        update_status('视频处理成功!')
    else:
        update_status('视频处理失败!')


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    #检查用户配置输入
    parse_args()
    #检查环境运行环境
    if not pre_check():
        return
    #检查模型文件是否存在
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            update_status(f'{frame_processor.NAME} 模型下载失败，请检查网络连接或手动下载模型文件')
            return
    # 检查是否有足够的内存
    limit_resources()
    # 开始处理，当没有gui界面的时候进行运行脚本无界面，反之启动用户互动界面程序
    if roop.globals.headless:
        update_status('开始处理...')
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
