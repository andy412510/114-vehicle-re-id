# TCMM: 車輛重識別專案 (Vehicle Re-identification)

本專案基於 TCMM (Trans-Contextual Memory Model) 與 Vision Transformer (ViT) 架構，針對 VeRi-776 資料集進行車輛重識別 (Vehicle ReID) 任務。專案包含完整的訓練、評估以及多種視覺化工具（t-SNE、Heatmap、DINO Attention、Ranking List）。

## 1. 環境配置 (Environment Setup)

建議使用 Conda 建立 Python 3.8.19 環境：

```bash
# 建立並安裝核心依賴
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

**核心依賴項目：**
- PyTorch 1.8.0
- scikit-learn 0.23.1
- numpy 1.23.5
- faiss-gpu 1.7.2
- openTSNE
- timm 0.3.4

## 2. 資料集準備 (Dataset Preparation)

1. 下載 **VeRi-776** 資料集。
2. 下載預訓練模型（例如 `pass_imagenet_lup.pth`）。
3. 確保資料夾結構符合預期，或在指令中指定 `--data-dir`。

## 3. 程式執行與路徑設定 (Usage & Configuration)

### 常用路徑與參數設定
在執行腳本時，可以透過參數修改設定：
- `--data-dir`: 資料集路徑 (預設為 `data`)
- `--logs-dir`: 日誌與模型儲存路徑 (預設為 `../logs`)
- `--gpu`: 指定 GPU ID (預設為 `0`)
- `--batch-size`: 批次大小 (預設為 `128`)
- `--pretrained_path`: 預訓練模型路徑 (預設為 `../pass_imagenet_lup.pth`)

### A. 模型訓練 (Training)
直接執行訓練腳本：
```bash
python train/main.py --data-dir [YOUR_DATA_PATH] --gpu 0
```
最佳模型將儲存於 `logs/model_best.pth.tar`。

### B. 評估 (Evaluation)
執行評估腳本以獲取 mAP 與 Rank-1 指標：
```bash
python train/evaluate.py
```

### C. 視覺化工具 (Visualization)
本專案整合了多種視覺化功能，主要透過 `train/main_vis.py` 執行（需先有訓練好的模型）。

1. **t-SNE 分佈圖**：
   分析測試集特徵的空間分佈。
   ```bash
   python train/main_vis.py --visualize tsne
   ```
   輸出：`tsne_visualization.jpg` 儲存於 `logs` 目錄。

2. **Heatmap 熱力圖**：
   生成指定圖片的注意力熱力圖。
   ```bash
   python train/main_vis.py --visualize heatmap --vis_image "path/to/image.jpg"
   ```
   輸出：`<image_name>_heatmap.png` 儲存於 `logs` 目錄。

3. **DINO 自注意力圖 (Self-Attention)**：
   分析 ViT 模型各個 head 的注意力分佈。
   ```bash
   python train/main_vis.py --visualize dino_attention --vis_image "path/to/image.jpg"
   ```
   輸出：於 `logs` 下建立新資料夾儲存注意力圖。

4. **Ranking List 排序視覺化**：
   此功能已整合在評估流程中，會輸出查詢圖片與對應的前幾名匹配結果。
   範例結果可參考 `ranking_list_vis/` 資料夾。

## 4. 檔案結構說明 (Project Structure)

```
.
├── train/
│   ├── main.py              # 核心訓練腳本
│   ├── main_vis.py          # 訓練與視覺化整合腳本
│   ├── evaluate.py          # 評估指標計算
│   ├── evaluate_heatmap.py  # 熱力圖評估
│   ├── vis_t-sne.py         # 獨立 t-SNE 繪製工具
│   └── TCMM/                # 核心模型、資料集處理與工具函式庫
├── logs/                    # 訓練日誌與模型權重 (checkpoint)
├── ranking_list_vis/        # Ranking List 視覺化結果示例
├── requirements.txt         # 專案依賴清單
└── pass_imagenet_lup.pth    # 預訓練權重文件
```

## 5. 參考資料 (References)
- [openTSNE](https://github.com/pavlin-policar/openTSNE)
- [DINO Attention Visualization](https://github.com/facebookresearch/dino/blob/main/visualize_attention.py)
- [Deep Person Re-ID (Ranking List Vis)](https://github.com/KaiyangZhou/deep-person-reid)
