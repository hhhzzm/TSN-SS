import os
import sys
import time

out_path = ''


def dump_frames(vid_path):
    import cv2
    video = cv2.VideoCapture(vid_path)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    fcount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    for i in range(fcount):
        ret, frame = video.read()
        if ret == False:
            # print('{} done'.format(vid_name))
            # print("%d/%d" % (i - 1, fcount - 1))
            sys.stdout.flush()
            return i - 1
        # cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame)
        # access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        # file_list.append(access_path)
    return i

def classes_init():
    dict = {}
    num = 0
    class_root = r"M:\Dataset\hmdb51_org"
    for root, dirs, items in os.walk(class_root):
        for item in items:
            if item.split('.')[1] != 'rar':
                continue
            label = item.split('.')[0]
            dict[label] = num
            num += 1
    return dict


class_dict = classes_init()
print(class_dict)

root_dir = r"M:\Dataset\hmdb51_org\videos"

frames = {}

for root, dirs, items in os.walk(root_dir):
    for item in items:
        if item.split('.')[1] != 'avi':
            continue
        item_dir = os.path.join(root_dir, item)
        x = dump_frames(item_dir)
        print(item, x)
        frames[item] = x

split_dir = r"M:\Download\test_train_splits\testTrainMulti_7030_splits"
train_list = []
valid_list = []
for root, dirs, items in os.walk(split_dir):
    for item in items:
        if item.split('.')[0][-1] != '1':
            continue
        ty = '_'.join(item.split('_')[:-2])
        split_file = open(os.path.join(root, item))
        lines = split_file.readlines()
        for line in lines:
            print(line)
            print(line.split(' '))
            a, b, _ = line.split(' ')
            frame = frames[a]
            label = class_dict[ty]
            print(ty, label, a, frame)
            path = os.path.join(root_dir, a.split('.')[0])
            if b == '1':
                train_list.append(' '.join([path, str(frame), str(label)]))
            elif b == '2':
                valid_list.append(' '.join([path, str(frame), str(label)]))
with open("train_list.txt", "w") as f:
    f.write('\n'.join(train_list))
with open("valid_list.txt", "w") as f:
    f.write('\n'.join(valid_list))
