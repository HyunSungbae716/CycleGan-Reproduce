set -ex
nohup python train.py --dataroot /data1/home/baehyunsung/hynix/pytorch-CycleGAN-and-pix2pix/datasets/apple2orange \
                      --name apple2orange_cyclegan \
                      --model cycle_gan \
                      --pool_size 50 \
                      --no_dropout \
                      --gpu_id 6 \
                      --display_id -1 > output2.log 2>&1 &
