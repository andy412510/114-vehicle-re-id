# VIT-S
# market1501, VeRi
python train/main.py \
	-b 512 \
	-d VeRi \
	-K 64 \
	--patch-rate 0.4 \
	--gpu 4,5,6,7 \
	--eps 0.8 \
	--epochs 60 \
	--logs-dir ./log/VeRi/pass_imagenet_lup \
	--data-dir /home/andy/ICASSP_data/data/ \
	-pp /home/andy/ICASSP_data/pretrain/PASS/pass_veriwild_imagenet_lup.pth \
