import tensorflow as tf


def build_input(dataset, data_path, batch_size, mode):
    """Build CIFAR image and labels.

    Args:
      dataset: Either 'cifar10' or 'cifar100'.
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """

    # 数据集参数
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    else:
        raise ValueError('Not supported dataset %s', dataset)

    # 数据读取参数
    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    # 获取文件名列表
    data_files = tf.io.gfile.glob(data_path)
    if not data_files:
        raise ValueError(f'No files found for data_path: {data_path}')

    def _parse_function(value):
        record = tf.reshape(tf.io.decode_raw(value, tf.uint8), [record_bytes])
        label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
        depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]), [depth, image_size, image_size])
        image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
        if mode == 'train':
            image = tf.image.resize_with_crop_or_pad(image, image_size + 4, image_size + 4)
            image = tf.image.random_crop(image, [image_size, image_size, 3])
            image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        return image, label

    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(_parse_function)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=10000).repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    images, labels = iterator.get_next()

    # 将标签数据由稀疏格式转换成稠密格式
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.compat.v1.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    # 检测数据维度
    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # 添加图片总结
    tf.compat.v1.summary.image('images', images)
    return images, labels
