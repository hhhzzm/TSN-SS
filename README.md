# Readme
基本上是 https://github.com/yjxiong/tsn-pytorch 中的代码，根据大作业数据集要求预处理了数据
## main.py
主程序，训练代码，训练命令：
`python main hmdb51 RGB train_list.txt valid_list.txt --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 8000 -b 8 -j 8 --dropout 0.8 --resume _rgb_checkpoint.pth.tar`
参数：
+ resume 继续之前的模型训练
+ epochs 随便写的，写大一点多跑跑

## train\_list.txt & valid\_list.txt
训练集和测试集的路径，绝对路径，想在本地跑需要重新搞一下，可以用videoextract.py自己生成，每一行是有三个元素，路径、帧数、类别
+ 路径是视频的路径，但是训练的时候使用的视频中的帧进行训练的，所以在视频的文件夹下每一个视频都有一个同名（不含后缀）的文件夹，里面是所有的帧的图片，可以使用videoextract.py文件自己提取（注意修改路径）
+ 帧数 字面意思
+ 类别 一共51类对应0-50，字母序对应

## videoextract.py
这个脚本主要有三个功能，建立类别和数字映射，提取每一帧并且统计，生成train\_list.txt和valid\_list.txt
