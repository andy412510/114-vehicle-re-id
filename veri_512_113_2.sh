# VIT-S
# msmt17, market1501, VeRi
python train/main.py \
	-b 512 \
	-d VeRi \
	-K 4 \
	--patch-rate 0.075 \
	--gpu 4,5,6,7 \
	--eps 0.7 \
	--epochs 60 \
	--logs-dir ./log/VeRi/vit_small_imagenet \
	--data-dir /home/andy/ICASSP_data/data/ \
	-pp /home/andy/ICASSP_data/pretrain/PASS/vit_small_imagenet.pth \
	--multi-neck
