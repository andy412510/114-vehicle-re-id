# Development Environment
windows  
Python 3.8.19  
Pytorch
and other mod in environment.yaml
# Download Data
Download one datasets:  
1. VeRi-776
    
Download VeRi pretrained:  
https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset

# Path Setting 
### Modify the following code to match your data path.
TCMM pretrained path:  
`checkpoint = load_checkpoint(osp.join(args.logs_dir, '512_K4_r0.075_outlers.pth.tar'))`

GPU setting:  
`parser.add_argument('--gpu', type=str, default='0,1,2,3')`

batch size setting:  
`parser.add_argument('-b', '--batch-size', type=int, default=2048)`

data folder path:  
`arser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/andy/ICASSP_data/data/')` 

# Evaluate
python ./train/evaluate.py

# Visualization
### T-SNE visualization
python ./train/vis_t-sne.py  

Reference: https://github.com/pavlin-policar/openTSNE/tree/master  

### Attention map visualization
python ./train/evaluate_heatmap.py  

Reference: https://github.com/facebookresearch/dino/blob/main/visualize_attention.py  
./train/TCMM/dino-main/visualize_attention.py  

Related:  
./train/evaluate_heatmap.py  

./train/TCMM/evaluators_heatmap.py:  
input batch data, file name list and model to vis_attention  
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluators_heatmap.py#L63  
follow reference work to obtain attention map, set path and patch here:  
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluators_heatmap.py#L21  

./train/TCMM/models/vision_transformer_heatmap.py  
### Ranking list visualization
This method is already integrated into the `Evaluate` program.

Reference: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/reidtools.py
./train/TCMM/evaluation_metrics/ranking_list_vis.py

Related:
./TCMM/evaluation_metrics/ranking.py

./TCMM/evaluation_metrics/ranking.py:
input batch id, top5 indices/matches list, query/gallery image path to top5_plot
https://github.com/andy412510/TCMM/blob/b206973dc8e8511ebde93323e188dd59fcd94176/train/TCMM/evaluation_metrics/ranking.py#L116

./train/TCMM/evaluation_metrics/ranking_list_vis.py:
Obtain ranking list visualization from reference work, set the visualization details and format here:
https://github.com/andy412510/TCMM/blob/c7c617e1224fac14418a2130594eb6318a12fa33/train/TCMM/evaluation_metrics/ranking_list_vis.py#L23


以下是main_vis.py使用方式
本版本僅改動main_vis.py / datasets/_init_.py 以及新增 datasets/veri.py
 =================================================================================
# 腳本使用說明 (How to Use This Script)
 =================================================================================

# 1. 訓練模型 (To Train the model):
    直接執行此腳本，不要加上 --visualize 參數。
    `python train/main.py`
    程式將會進行訓練，並將最佳模型儲存於日誌 (logs) 目錄下。

# 2. 生成視覺化圖表 (To Generate Visualizations):
    在模型訓練完成後 (日誌目錄中已有 'model_best.pth.tar' 檔案)，使用 --visualize 參數。
    程式會跳過訓練，直接載入最佳模型來生成指定的圖表。

#    A. t-SNE 分佈圖:
       此功能會分析整個測試集的特徵分佈。
       `python train/main.py --visualize tsne`
       輸出: `tsne_visualization.jpg` 將會儲存在日誌目錄下。

#    B. Heatmap 熱力圖:
       為單張圖片生成注意力熱力圖，需要搭配 --vis_image 參數指定圖片路徑。
       `python train/main.py --visualize heatmap --vis_image "path/to/your/image.jpg"`
       輸出: `<image_name>_heatmap.png` 將會儲存在日誌目錄下。

#    C. DINO 自注意力圖:
       為單張圖片生成 ViT 模型每個 head 的自注意力圖，需要搭配 --vis_image 參數。
       `python train/main.py --visualize dino_attention --vis_image "path/to/your/image.jpg"`
       輸出: 日誌目錄下會建立一個新的資料夾，內含所有注意力圖。

 =================================================================================
