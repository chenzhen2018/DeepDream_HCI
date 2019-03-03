#-*-coding:utf-8-*-
# author: by chenzhen


# =============================================================================================
from tkinter import *
import tkinter
import tkinter.filedialog
import os
import tkinter.messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import deepdream
import scipy
import numpy as np
import matplotlib.pyplot as plt

##########
# 窗口属性
##########
root = tkinter.Tk()
root.title('Deep Dream')
root.geometry('1200x600')

formatImg = ['jpg']


# =============================================================================================
##########
# 支撑函数
##########

def resize(w, h, w_box, h_box, pil_image):
  # 对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例

  f1 = 1.0*w_box/w # 1.0 forces float division in Python2
  f2 = 1.0*h_box/h
  factor = min([f1, f2])
  width = int(w*factor)
  height = int(h*factor)
  return pil_image.resize((width, height), Image.ANTIALIAS)


def show_img(img_path):
    """
    选择背景文件后，显示在窗口中
    """
    pil_image = Image.open(img_path)    # 打开图片
    # 期望显示大小
    w_box = 400
    h_box = 400
    # 获取原始图像的大小
    w, h = pil_image.size
    pil_image_resized = resize(w, h, w_box, h_box, pil_image)

    # 把PIL图像对象转变为Tkinter的PhotoImage对象
    tk_image = ImageTk.PhotoImage(pil_image_resized)

    img = tkinter.Label(image=tk_image, width=w_box, height=h_box)
    img.image = tk_image
    img.place(x=50, y=100)


def choose_imgfile():
    img_path = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    if img_path[-3:] not in formatImg:
        tkinter.messagebox.askokcancel(title='出错', message='未选择图片或图片格式不正确')   # 弹出错误窗口
        return
    else:
        variable_path_bgimg.set(img_path)  # 设置变量
        show_img(img_path)   # 显示图片


def show_new_img(img_array):
    """
    在窗口中显示生成的图片，(之后进行修改，将选择的背景图片与新生成的图片，使用的show函数写成一致)
    """
    plt.imshow(np.uint8(img_array))
    plt.xticks([])
    plt.yticks([])
    plt.show()


class ImageArray():
    """
    是为了能够保存新生成的图像数组
    """
    def __init__(self):
        self.img_array = None

    def set_imgarray(self, img_array):
        self.img_array = img_array

    def get_imgarray(self):
        return self.img_array


def get_parameter():
    """
    从窗口中获取用户设置的参数，并对其检查并返回；
    :return:
    """
    select_conv_layer = combox_name_convlayer.get()  # 选择的卷积层名称，全称
    if bool_all_channel.get() == 1:  # 判断是否使用全通都
        print('bool_var_channel:', 1)
        select_conv_layer = select_conv_layer.split('_')[0]
    else:
        select_conv_layer = select_conv_layer.split('/conv')[0]

    path_bgimg = entry_path_bgimg.get()  # 选择的背景图片路径名
    iter_n = combobox_iter_num.get()  # 迭代的次数
    octave_n = combobox_octave_n.get()  # 放大次数
    octave_scale = combobox_octave_scale.get()  # 放大倍数
    channel = combobox_channel.get()  # 所选择的通道数

    return select_conv_layer, path_bgimg, iter_n, octave_n, octave_scale, channel

# 实例化组对象
imgarray = ImageArray()


def generate_img():
    # 获取参数
    select_conv_layer, path_bgimg, iter_n, octave_n, octave_scale, channel = get_parameter()

    # 判断是否选择卷积层和背景图片
    if select_conv_layer is '' or path_bgimg is '':
        tkinter.messagebox.askokcancel(title='出错', message='shit\n未选择背景图片或卷积层')   # 弹出错误窗口
    else:
        # 生成图片
        img_array = deepdream.generate_img(path_bgimg, select_conv_layer, int(iter_n), int(octave_n), float(octave_scale), int(channel), int(bool_all_channel.get()))

        # 将新生成图像赋值给数组对象，用于保存
        imgarray.set_imgarray(img_array)

        # 提示生成成功，并进行显示
        tkinter.messagebox.showinfo(title='提示信息', message='生成图片成功，即将显示')  # 弹出提示信息
        show_new_img(img_array)

def save_img():
    """
    保存图片，现在不使用了；
    因为使用plt显示图片，其中自带的有将图片保存到本地；
    :return:
    """
    img_array = imgarray.get_imgarray()
    if np.sum(img_array == None) == 1:
        tkinter.messagebox.askokcancel(title='出错', message='未正常生成图片')  # 弹出错误窗口
    else:
        img_name = tkinter.filedialog.asksaveasfilename(title='保存图片')  # 弹出窗口，进行保存，返回文件名
        if img_name[-3:] not in formatImg:
            tkinter.messagebox.askokcancel(title='出错', message='图片格式不正确\n请使用jpg格式')  # 弹出错误窗口
            return
        scipy.misc.toimage(img_array).save(img_name)
        tkinter.messagebox.showinfo(title='提示信息', message='保存图片成功')  # 弹出提示信息


# =============================================================================================
##################
# 窗口部件
##################

variable_path_bgimg = tkinter.StringVar() # 字符串变量

############
# 选择背景图片并显示
############
# label : 选择文件
label_select_bgimg = tkinter.Label(root, text='选择背景图片：')
label_select_bgimg.grid(row=0, column=0)

# Entry: 显示图片文件路径地址
entry_path_bgimg = tkinter.Entry(root, width=80, textvariable=variable_path_bgimg)
entry_path_bgimg.grid(row=0, column=1)

# Button: 选择图片文件
button_select_bgimg = tkinter.Button(root, text="选择", command=choose_imgfile)
button_select_bgimg.grid(row=0, column=2)

############
# 选择使用的卷积层
############
# label : 选择使用的卷积层
label_select_bgimg = tkinter.Label(root, text='选择卷积层：')
label_select_bgimg.grid(row=1, column=0)

# combobox: 下拉菜单框
combox_name_convlayer = ttk.Combobox(root, width=80, height=20, textvariable='jjjj', state='readonly')
combox_name_convlayer.grid(row=1, column=1)
combox_name_convlayer["values"] = deepdream.get_convlayer()


############
# 开始生成图片并保存
############
# Button: 执行识别程序按钮
button_recogImg = tkinter.Button(root, text="开始生成", command=generate_img)
button_recogImg.grid(row=1, column=2)


# Button: 保存新生成的图片
# button_recogImg = tkinter.Button(root, text="保存", command=save_img)
# button_recogImg.grid(row=1, column=3)


############
# 参数设置
############
# 迭代次数
var_iter_num = tkinter.StringVar(); var_iter_num.set(10)
label_iter_num = tkinter.Label(root, text='iter_num:', font=18)
label_iter_num.grid(row=8, column=4)
combobox_iter_num = ttk.Combobox(root, width=10, height=8, textvariable=var_iter_num)
combobox_iter_num.grid(row=8, column=5)
combobox_iter_num["values"] = [i+1 for i in range(20)]

# 放大倍数
var_octave_scale = tkinter.StringVar(); var_octave_scale.set(1.4)
label_octave_scale = tkinter.Label(root, text='octave_scale:', font=18)
label_octave_scale.grid(row=12, column=4)
combobox_octave_scale = ttk.Combobox(root, width=10, height=5,  textvariable=var_octave_scale)
combobox_octave_scale.grid(row=12, column=5)
combobox_octave_scale["values"] = list(np.linspace(1, 2, 11))

# 放大次数
var_octave_n = tkinter.StringVar(); var_octave_n.set(5)
label_octave_n = tkinter.Label(root, text='octave_n:', font=18)
label_octave_n.grid(row=16, column=4)
combobox_octave_n = ttk.Combobox(root, width=10, height=5,  textvariable=var_octave_n)
combobox_octave_n.grid(row=16, column=5)
combobox_octave_n["values"] = [i+1 for i in range(20)]

# 选择通道
var_channel = tkinter.StringVar(); var_channel.set(139)
label_channel = tkinter.Label(root, text='channel:', font=18)
label_channel.grid(row=20, column=4)
combobox_channel = ttk.Combobox(root, width=10, height=20,  textvariable=var_channel)
combobox_channel.grid(row=20, column=5)
combobox_channel["values"] = [i+1 for i in range(150)]
label_channel_point = tkinter.Label(root, text='注意:应小于该卷积层通道总数目', font=18)
label_channel_point.grid(row=21, column=5)

# 是否使用全通道
bool_all_channel = tkinter.IntVar()
checkbutton_all_channel = tkinter.Checkbutton(root, text='All Channels？', font=18, variable=bool_all_channel)
checkbutton_all_channel.grid(row=22, column=5)

root.mainloop()
