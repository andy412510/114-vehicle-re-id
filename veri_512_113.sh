# VIT-S
# market1501, VeRi
python train/main.py \
	-b 512 \
	-d VeRi \
	-K 64 \
	--neg-dedup \
	--patch-rate 0.4 \
	--gpu 0,1,2,3 \
	--eps 0.8 \
	--epochs 60 \
	--logs-dir ./log/ \
	--data-dir /home/andy/ICASSP_data/data/ \
	-pp /home/andy/ICASSP_data/pretrain/PASS/pass_imagenet_lup.pth \
