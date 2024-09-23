import tensorflow as tf
import os
import tarfile

# 下载 CIFAR-10 数据集
def download_and_extract_cifar10(data_dir):
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
    if not os.path.exists(filepath):
        filepath, _ = tf.keras.utils.get_file(filepath, cifar10_url, extract=False)
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=data_dir)

data_dir = 'data/cifar-10-batches-bin'
download_and_extract_cifar10(data_dir)
