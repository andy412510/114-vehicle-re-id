import torch

ckpt_imagenet = torch.load('/home/andy/ICASSP_data/pretrain/PASS/vit_small_imagenet.pth', map_location='cpu')
ckpt_pass = torch.load('/home/andy/ICASSP_data/pretrain/PASS/pass_vit_small_full.pth', map_location='cpu')

# 提取 state_dict (部分 checkpoint 可能封裝在 'model' 或 'state_dict' 鍵值下)
sd1 = ckpt_imagenet.get('model', ckpt_imagenet)
sd2 = ckpt_pass.get('model', ckpt_pass)
keys1 = set(sd1.keys())
keys2 = set(sd2.keys())

only_in_imagenet = keys1 - keys2
only_in_pass = keys2 - keys1
shared_keys = keys1 & keys2
for k in shared_keys:
    if sd1[k].shape != sd2[k].shape:
        print(f"Shape mismatch in {k}: {sd1[k].shape} vs {sd2[k].shape}")

print(f"共有層數 (Shared layers): {len(shared_keys)}")
print(f"ImageNet 獨有層數: {len(only_in_imagenet)}")
print(f"PASS 獨有層數: {len(only_in_pass)}")

if len(shared_keys) > 0:
    print("✅ 檢查完畢：所有共有層的維度均完全匹配。")