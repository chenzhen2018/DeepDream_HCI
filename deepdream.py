# coding:utf-8
# 导入要用到的基本模块。
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import PIL
import scipy


# 创建图，会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
# 读取模型文件
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()  #
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def get_convlayer():
    """
    获取模型中所有卷积层的名称
    :return:
    """
    list_convlayer = []
    if not os.path.exists('model_name_convlayer.txt'):

        file = open('model_name_convlayer.txt', 'w')  # 打开文件
        for op in graph.get_operations():
            if op.type == 'Conv2D' and 'import/' in op.name:
                list_convlayer.append('%s %s' % (op.name, str(graph.get_tensor_by_name(op.name + ':0').get_shape())))
                file.write('%s %s\n' % (op.name, str(graph.get_tensor_by_name(op.name + ':0').get_shape())))  # 写入
        sess.close()
    else:
        file = open('model_name_convlayer.txt', 'r')
        for line in file.readlines():
            line = line.strip('\n')
            list_convlayer.append(line)
    file.close()
    return list_convlayer


def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, hw))
    img = img / 255 * (max - min) + min
    return img

def calc_grad_tiled(img, t_grad, tile_size=512):
    """
    计算梯度
    Paramater
        img: 金字塔分解，金字塔恢复后的背景图片
        t_grad: 梯度
        tile_size: 处理块大小
    Returns:

    """
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 先在行上做整体移动，再在列上做整体移动
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img0,
                     iter_n=2, step=1.5, octave_n=2, octave_scale=1.4):
    """
    生成带背景的图

    Paramater
        t_obj: 通道
        img0: 背景图片
        inter_n: 迭代次数
        step: 学习率
        octave: 放大次数
        octave_scale: 放大倍数
    """
    t_score = tf.reduce_mean(t_obj)  # 计算均值，最大化t_score
    t_grad = tf.gradients(t_score, t_input)[0]  # 计算梯度

    img = img0
    # 同样将图像进行金字塔分解
    # 此时提取高频、低频的方法比较简单。直接缩放就可以
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))   # 低频成分
        hi = img - resize(lo, hw)   # 图像减去低频成分的缩放得到高频成分
        img = lo
        octaves.append(hi)

    # 先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]   # 获取金字塔下一层，最顶层是img，也就是最后一层的lo
            img = resize(img, hi.shape[:2]) + hi    # 金字塔中下一层的形状 hi.shape[:2]，循环恢复成原始大小；
        for i in range(iter_n): # 迭代

            print('iter_n: %d' % i)
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')

    img = img.clip(0, 255)
    return img

def generate_img(path_bgimg, select_conv_layer, iter_n, octave_n, octave_scale, channel, bool_all_channel):
    """
    根据背景图片生成图片

    Parameter:
        path_bgimg: 背景图片的路径
        select_conv_layer: 选择的卷积层名称
        iter_n: 迭代次数
        octave_n: 放大次数
        octave_scale: 放大倍数
        channel: 选择的通道数
        bool_all_channel: 是否使用全通道

    Return:
        new_img: 新生成的图片数组
    """

    img0 = PIL.Image.open(path_bgimg)  # 打开图片
    img0 = np.float32(img0)  # 转换数据类型


    layer_output = graph.get_tensor_by_name("%s:0" % select_conv_layer)
    if bool_all_channel == 1:
        print(select_conv_layer)
        new_img = render_deepdream(tf.square(layer_output), img0, iter_n=iter_n, octave_n=octave_n,
                                   octave_scale=octave_scale)
    else:
        print(select_conv_layer)
        new_img = render_deepdream(layer_output[:, :, :, channel], img0, iter_n=iter_n, octave_n=octave_n,
                                   octave_scale=octave_scale)
    return new_img

