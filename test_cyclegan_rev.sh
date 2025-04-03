set -ex
nohup python test.py --dataroot /data1/home/baehyunsung/hynix/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange \
               --name apple2orange_cyclegan --model cycle_gan --phase test --gpu_id 6 --no_dropout > test.log 2>&1 &
