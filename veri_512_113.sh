# VIT-S
# msmt17, market1501, VeRi
python train/main.py \
	-b 512 \
	-d VeRi \
	-K 4 \
	--patch-rate 0.075 \
	--gpu 0,1,2,3 \
	--eps 0.7 \
	--epochs 60 \
	--logs-dir ./log/VeRi/pass_vit_small_full \
	--data-dir /home/andy/ICASSP_data/data/ \
	-pp /home/andy/ICASSP_data/pretrain/PASS/pass_veriwild_imagenet_lup.pth \
