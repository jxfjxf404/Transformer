"""
============================================================
数据集下载与预处理脚本
============================================================
支持 Multi30k 数据集的下载和预处理。

用法:
  python download_data.py              # 下载 Multi30k
  python download_data.py --skip-download  # 仅预处理（数据已存在）

参考:
  - Multi30k 数据集: https://github.com/multi30k/dataset
============================================================
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from tqdm import tqdm


def download_file_with_progress(url: str, filepath: str):
    """
    带进度条下载文件。

    参数:
        url: 下载链接
        filepath: 保存路径
    """
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f, tqdm(
                desc=os.path.basename(filepath),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

        print(f"下载完成: {filepath}")
        return True

    except urllib.error.URLError as e:
        print(f"下载失败: {url}")
        print(f"错误: {e}")
        return False
    except Exception as e:
        print(f"下载 {filepath} 时出错: {e}")
        return False


def download_multi30k(data_dir: str = "data/multi30k"):
    """
    下载 Multi30k 数据集。

    参数:
        data_dir: 数据保存目录
    """
    print("=" * 60)
    print("下载 Multi30k 英德翻译数据集")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)

    urls = [
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.en", "train.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/train.de", "train.de"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.en", "val.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/val.de", "val.de"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.en", "test.en"),
        ("https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/test_2016_flickr.de", "test.de"),
    ]

    success_count = 0

    for url, filename in urls:
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            print(f"文件已存在，跳过: {filename}")
            success_count += 1
            continue

        print(f"\n下载 {filename}...")
        if download_file_with_progress(url, filepath):
            success_count += 1

    print("\n" + "=" * 60)
    if success_count == len(urls):
        print("数据集下载完成!")
    else:
        print(f"警告: 仅成功下载 {success_count}/{len(urls)} 个文件")
        print("\n如果下载失败，请尝试以下方法:")
        print("1. 检查网络连接")
        print("2. 手动下载文件到 data/multi30k/ 目录")
        print("3. 使用备用数据源")
    print("=" * 60)

    return success_count == len(urls)


def verify_data(data_dir: str = "data/multi30k") -> bool:
    """
    验证数据集文件是否完整。

    参数:
        data_dir: 数据目录

    返回:
        数据是否完整
    """
    required_files = [
        "train.en", "train.de",
        "val.en", "val.de",
        "test.en", "test.de",
    ]

    print("\n验证数据集文件...")

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        elif os.path.getsize(filepath) == 0:
            print(f"警告: 文件为空: {filename}")
            missing_files.append(filename)

    if missing_files:
        print(f"缺失文件: {', '.join(missing_files)}")
        return False

    print("数据集文件验证通过!")
    return True


def print_data_stats(data_dir: str = "data/multi30k"):
    """
    打印数据集统计信息。

    参数:
        data_dir: 数据目录
    """
    files = ["train.en", "train.de", "val.en", "val.de", "test.en", "test.de"]

    print("\n数据集统计:")
    print("-" * 60)

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            print(f"{filename:15s}: {len(lines):6d} 行")
        else:
            print(f"{filename:15s}: 文件不存在")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="下载和预处理 Multi30k 数据集")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/multi30k",
        help="数据保存目录"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载步骤（假设数据已存在）"
    )
    args = parser.parse_args()

    if not args.skip_download:
        success = download_multi30k(args.data_dir)

        if not success:
            print("\n" + "=" * 60)
            print("备用方案: 手动下载数据集")
            print("=" * 60)
            print("请访问以下链接手动下载数据:")
            print("https://github.com/multi30k/dataset/tree/master/data/task1")
            print("\n将以下文件保存到 data/multi30k/ 目录:")
            print("  - train.en, train.de (训练集)")
            print("  - val.en, val.de (验证集)")
            print("  - test_2016_flickr.en, test_2016_flickr.de (测试集)")
            print("\n下载后重命名测试集文件:")
            print("  test_2016_flickr.en -> test.en")
            print("  test_2016_flickr.de -> test.de")
            print("=" * 60)
            return 1

    if verify_data(args.data_dir):
        print_data_stats(args.data_dir)
        print("\n数据集准备完成，可以开始训练!")
        print("运行: python train_translation.py")
        return 0
    else:
        print("\n数据集不完整，请检查文件!")
        return 1


if __name__ == "__main__":
    exit(main())