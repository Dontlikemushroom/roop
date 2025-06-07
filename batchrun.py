import os
import subprocess
import argparse


def process_videos(source_image, resource_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历资源目录下的所有文件
    for filename in os.listdir(resource_dir):
        # 检查是否为视频文件（可根据需要添加更多扩展名）
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_path = os.path.join(resource_dir, filename)

            # 构建输出路径（保持原名，存放在输出目录）
            output_path = os.path.join(output_dir, f"swapped_{filename}")

            # 构建命令
            command = [
                "python", "run.py",
                "-s", source_image,
                "-t", video_path,
                "-o", output_path,
                "--frame-processor", "face_swapper", "face_enhancer",
                "--keep-fps",
                "--many-faces",
                "--similar-face-distance", "80",
                # "--half-face", "7",
                "--output-video-encoder", "h264_nvenc",
                "--output-video-quality", "80",
                "--execution-provider", "cuda",
                "--execution-threads", "8"
            ]

            print(f"\n{'=' * 50}")
            print(f"处理视频: {filename}")
            print(f"输出路径: {output_path}")
            print(f"执行命令: {' '.join(command)}")

            # 执行命令
            try:
                subprocess.run(command, check=True)
                print(f"✓ 完成处理: {filename}")
            except subprocess.CalledProcessError as e:
                print(f"✕ 处理失败: {filename}, 错误: {e}")
            except Exception as e:
                print(f"✕ 发生意外错误: {e}")


if __name__ == "__main__":
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='批量视频换脸处理')
    parser.add_argument('--source', default='/root/autodl-tmp/roop/resource/IMG_0765.JPG',
                        help='源图片路径')
    parser.add_argument('--resource_dir', default='/root/autodl-tmp/roop/resource',
                        help='视频资源目录')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/roop/outputs',
                        help='输出目录')

    args = parser.parse_args()

    print(f"源图片: {args.source}")
    print(f"视频目录: {args.resource_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'=' * 50}\n开始批量处理...")

    # 执行处理
    process_videos(args.source, args.resource_dir, args.output_dir)

    print("\n所有视频处理完成！")