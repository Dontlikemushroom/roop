from typing import Any, List, Callable
import cv2
import insightface
import threading
import numpy

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path,
                providers=roop.globals.execution_providers
            )
            # 新增设备状态提示
            update_status(f'正在使用 {roop.globals.execution_providers[0].split("ExecutionProvider")[0]} 加速')
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('请选择包含源图像的路径', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('源图像中未检测到人脸', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('请选择目标图像或视频路径', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    try:
        # 验证源人脸特征
        if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
            print("警告：源人脸特征无效")
            return temp_frame

        # 验证目标人脸
        if target_face is None:
            print("警告：目标人脸无效")
            return temp_frame

        # 确保 half_face_swap 有一个有效的值
        half_face_value = getattr(roop.globals, 'half_face_swap', 0)
        if half_face_value is None:
            half_face_value = 0

        if half_face_value > 0:
            # 获取人脸框的坐标
            if not hasattr(target_face, 'bbox') or target_face.bbox is None:
                print("警告：目标人脸框无效")
                return temp_frame

            x1, y1, x2, y2 = map(int, target_face.bbox)
            
            # 保存原始图像的副本
            original_frame = temp_frame.copy()
            
            # 进行完整的换脸
            swapped_frame = get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
            if swapped_frame is None:
                print("警告：换脸操作失败")
                return temp_frame
            
            # 计算替换比例（0-10转换为实际的像素位置）
            face_height = y2 - y1
            swap_height = int(face_height * (half_face_value / 10))
            swap_y = y1 + swap_height
            
            # 创建一个遮罩，上部分为1，下部分为0
            mask = numpy.zeros_like(temp_frame)
            mask[y1:swap_y, x1:x2] = 1
            
            # 在边界处创建渐变过渡区域
            blend_zone_height = int(face_height * 0.1)  # 10%的脸部高度作为过渡区
            if blend_zone_height > 0:
                for i in range(blend_zone_height):
                    alpha = 1.0 - (i / blend_zone_height)
                    y = min(swap_y + i, temp_frame.shape[0] - 1)
                    mask[y, x1:x2] = alpha
            
            # 使用遮罩组合上半部分的换脸结果和下半部分的原始图像
            temp_frame = swapped_frame * mask + original_frame * (1 - mask)
            
            return temp_frame.astype('uint8')
        else:
            return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    except Exception as e:
        print(f"警告：换脸过程中出错: {str(e)}")
        return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        # 对每一帧都进行人脸检测和处理
        if roop.globals.many_faces:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            target_face = get_one_face(temp_frame)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
        cv2.imwrite(temp_frame_path, temp_frame)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    # 直接处理每一帧，不需要参考帧
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
