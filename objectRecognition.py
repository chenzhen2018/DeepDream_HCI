# encoding: utf-8

from tkinter import *
import tkinter
import tkinter.filedialog
import os
import tkinter.messagebox
from PIL import Image, ImageTk
import cifar10_eval_perImg

# 窗口属性
root = tkinter.Tk()
root.title('CIRFAR-10图片识别')
root.geometry('800x600')

formatImg = ['jpg']


def resize(w, h, w_box, h_box, pil_image):
  # 对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例

  f1 = 1.0*w_box/w # 1.0 forces float division in Python2
  f2 = 1.0*h_box/h
  factor = min([f1, f2])
  width = int(w*factor)
  height = int(h*factor)
  return pil_image.resize((width, height), Image.ANTIALIAS)

def showImg():
    img1 = entry_imgPath.get()  # 获取图片路径地址
    pil_image = Image.open(img1)    # 打开图片
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


def choose_file():
    text_showClass.delete(0.0, END) # 清空输出结果文本框，在再次选择图片文件之前清空上次结果；
    selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')  # 选择文件
    if selectFileName[-3:] not in formatImg:
        tkinter.messagebox.askokcancel(title='出错', message='未选择图片或图片格式不正确')   # 弹出错误窗口
        return
    else:
        e.set(selectFileName)  # 设置变量
        showImg()   # 显示图片


def ouputOfModel():
    # 完成识别，显示类别
    # 图片文件路径
    text_showClass.delete(0.0, END) # 清空上次结果文本框
    img_path = entry_imgPath.get()  # 获取所选择的图片路径地址

    # 判断是否存在改图片
    if not os.path.exists(img_path):
        tkinter.messagebox.askokcancel(title='出错', message='未选择图片文件或图片格式不正确')
    else:

        # 得到输出结果，以及相应概率
        label_pred, prob = cifar10_eval_perImg.evaluate_perImg(img_path)
        # 通过训练的模型，计算得到相对应输出类别

        # 清空文本框中的内容，写入识别出来的类别
        text_showClass.config(state=NORMAL)
        text_showClass.insert('insert', '%s: %s' % (label_pred, prob))



##################
# 窗口部件
##################

e = tkinter.StringVar() # 字符串变量

# label : 选择文件
label_selectImg = tkinter.Label(root, text='选择图片：')
label_selectImg.grid(row=0, column=0)

# Entry: 显示图片文件路径地址
entry_imgPath = tkinter.Entry(root, width=80, textvariable=e)
entry_imgPath.grid(row=0, column=1)

# Button: 选择图片文件
button_selectImg = tkinter.Button(root, text="选择", command=choose_file)
button_selectImg.grid(row=0, column=2)

# Button: 执行识别程序按钮
button_recogImg = tkinter.Button(root, text="开始识别", command=ouputOfModel)
button_recogImg.grid(row=0, column=3)

# Text: 显示结果类别文本框
text_showClass = tkinter.Text(root, width=25, height=1, font='18',)
text_showClass.grid(row=1, column=1)
text_showClass.config(state=DISABLED)

root.mainloop()
