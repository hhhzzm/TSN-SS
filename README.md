train：
python main hmdb51 RGB train_list.txt valid_list.txt --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 8000 -b 8 -j 8 --dropout 0.8 --resume _rgb_checkpoint.pth.tar

