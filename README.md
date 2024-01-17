# RDLNET

-------------------------------------------------------------------------------------------------------------
1. windows預先安裝
    Nvidia CUDA ToolKit 11.1
    CUDNN 8.0.5


2. anaconda創建一個虛擬環境
    conda create -n open-mmlab python=3.9 -y
    conda activate open-mmlab

3. 安裝插件
    終端機 cd..動到專案目錄( RDLNET_wang )底下, 依序安裝
    conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    pip install -r requirements.txt
    pip install -v -e .



-------------------------------------------------------------------------------------------------------------
DataSets 準備
Download REDS, Vimeo90K, vid4 三個資料集
神經網路需要下採樣的低解析度圖，如果官方沒有給的話，就要執行以下自己做bicubic下採樣
其中source_folder, target_folder需要改當前位置

1.
python tools/image_resize_REDS4.py

2.
python tools/image_resize_vid.py

3.
python tools/image_resize_vimeo90k.py

-------------------------------------------------------------------------------------------------------------
主網路backbones
mmedit\models\backbones\sr_backbones\basicvsr_net_RDN_v01.py
mmedit\models\backbones\sr_backbones\basicvsr_net_RDN_v01_x2.py
mmedit\models\backbones\sr_backbones\basicvsr_net_RDN_v11.py



-------------------------------------------------------------------------------------------------------------
# 相關訓練參數設計
# 更改configs底下這幾個.py，其中lq_folder、gt_folder需要更改成當前位置
# RDLNET(RDBs=5), VSRx4
configs\restorers\basicvsr\basicvsr_RDN_v01_TUF_reds4.py
configs\restorers\basicvsr\basicvsr_RDN_v01_TUF_vid4.py
configs\restorers\basicvsr\basicvsr_RDN_v01_TUF_vimeo90k_bi.py

# RDLNET(RDBs=5), VSRx2
configs\restorers\basicvsr\basicvsr_RDN_v01_x2_TUF_reds4.py
configs\restorers\basicvsr\basicvsr_RDN_v01_x2_TUF_vid4.py
configs\restorers\basicvsr\basicvsr_RDN_v01_x2_TUF_vimeo90k_bi.py

# RDLNET(RDBs=3), VSRx4
configs\restorers\basicvsr\basicvsr_RDN_v11_S4_TUF_reds4.py
configs\restorers\basicvsr\basicvsr_RDN_v11_S4_TUF_vid4.py
configs\restorers\basicvsr\basicvsr_RDN_v11_S4_TUF_vimeo90k_bi.py

-------------------------------------------------------------------------------------------------------------
# 訓練部分
# RDLNET(RDBs=5), VSRx4, train
# Use REDS for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_reds4.py
# Use Vimeo90k for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_vimeo90k_bi.py


# RDLNET(RDBs=5), VSRx2, train
# Use REDS for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_reds4.py
# Use Vimeo90k for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_vimeo90k_bi.py

# RDLNET(RDBs=3), VSRx4, train
# Use REDS for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_reds4.py
# Use Vimeo90k for training
python tools/train.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_vimeo90k_bi.py



-------------------------------------------------------------------------------------------------------------
# 測試部分
# RDLNET(RDBs=5), VSRx4, test
# Trained by REDS, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_reds4.py work_dirs/basicvsr_RDN_v01_reds4_220220/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_reds4_results.pkl --save-path work_dirs/output/reds4

# Trained by REDS, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v01_reds4_220220/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_vimeo90k_results.pkl --save-path work_dirs/output/vimeo_90k_T

# Trained by REDS, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_vid4.py work_dirs/basicvsr_RDN_v01_reds4_220220/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_vid4_results.pkl --save-path work_dirs/output/vid4

# Trained by Vimeo90k, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_reds4.py work_dirs/basicvsr_RDN_v01_vimeo90k_bi_220303/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_reds4_vimeo90k_pretrained_results.pkl --save-path work_dirs/output/reds4

# Trained by Vimeo90k, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v01_vimeo90k_bi_220303/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_vimeo90k_vimeo90k_pretrained_results.pkl --save-path work_dirs/output/vimeo_90k_T

# Trained by Vimeo90k, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_vid4.py work_dirs/basicvsr_RDN_v01_vimeo90k_bi_220303/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v01_vid4_results.pkl --save-path work_dirs/output/vid4



-------------------------------------------------------------------------------------------------------------
# 測試部分
# RDLNET(RDBs=5), VSRx2, test
# Trained by REDS, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_reds4.py work_dirs/basicvsr_RDN_v01_x2_reds4_220904/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/REDS_trained/REDS4/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/REDS_trained/REDS4

# Trained by REDS, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v01_x2_reds4_220904/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/REDS_trained/vimeo90k/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/REDS_trained/vimeo90k

# Trained by REDS, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_vid4.py work_dirs/basicvsr_RDN_v01_x2_reds4_220904/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/REDS_trained/vid4/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/REDS_trained/vid4


# Trained by Vimeo90k, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_reds4.py work_dirs/basicvsr_RDN_v01_x2_vimeo90k_bi_220905/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/REDS4/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/REDS4

# Trained by Vimeo90k, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v01_x2_vimeo90k_bi_220905/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/vimeo90k/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/vimeo90k

# Trained by Vimeo90k, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v01_x2_TUF_vid4.py work_dirs/basicvsr_RDN_v01_x2_vimeo90k_bi_220905/iter_150000.pth --out work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/vid4/results.pkl --save-path work_dirs/output/BasicVSR_RDN_x2/vimeo90k_trained/vid4





-------------------------------------------------------------------------------------------------------------
# 測試部分
RDLNET(RDBs=3), VSRx4, test
# Trained by REDS, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_reds4.py work_dirs/basicvsr_RDN_v11_S4_reds4_220514/iter_300000.pth --out work_dirs/output/BasicVSR_RDN_v11_reds_pretrained_reds4_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_reds_pretrained/reds4

# Trained by REDS, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v11_S4_reds4_220514/iter_300000.pth --out work_dirs/output/BasicVSR_RDN_v11_reds_pretrained_vimeo90k_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_reds_pretrained/vimeo90k_T

# Trained by REDS, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_vid4.py work_dirs/basicvsr_RDN_v11_S4_reds4_220514/iter_300000.pth --out work_dirs/output/BasicVSR_RDN_v11_reds_pretrained_vid4_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_reds_pretrained/vid4

# Trained by Vimeo90k, Test on REDS4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_reds4.py work_dirs/basicvsr_RDN_v11_S4_vimeo90k_bi_220518/iter_300000.pth --out work_dirs/output/BasicVSR_RDN_v11_vimeo90k_pretrained_reds4_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_vimeo90k_pretrained/reds4

# Trained by Vimeo90k, Test on Vimeo90k-T
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_vimeo90k_bi.py work_dirs/basicvsr_RDN_v11_S4_vimeo90k_bi_220518/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v11_vimeo90k_pretrained_vimeo90k_T_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_vimeo90k_pretrained/vimeo90k_T

# Trained by Vimeo90k, Test on Vid4
python tools/test.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_vid4.py work_dirs/basicvsr_RDN_v11_S4_vimeo90k_bi_220518/iter_300000.pth --out work_dirs/output/basicVSR_RDN_v11_vimeo90k_pretrained_vid4_results.pkl --save-path work_dirs/output/BasicVSR_RDN_v11_vimeo90k_pretrained/vid4






-------------------------------------------------------------------------------------------------------------
執行時間測試
1. BasicVSR
python tools/get_runtime.py configs/restorers/basicvsr/basicvsr_reds4.py

2. RDLNET(RDBs=5)
python tools/get_runtime.py configs/restorers/basicvsr/basicvsr_RDN_v01_TUF_reds4.py

3. RDLNET(RDBs=3)
python tools/get_runtime.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_reds4.py


-------------------------------------------------------------------------------------------------------------
計算量FLOPs測試
1. BasicVSR
python tools/get_flops.py configs/restorers/basicvsr/basicvsr_reds4.py --shape 10 3 128 128

2. RDLNET(RDBs=5)
python tools/get_flops.py configs/restorers/basicvsr/basicvsr_RDN_v01_linux_reds4.py --shape 10 3 128 128

3. RDLNET(RDBs=3)
python tools/get_flops.py configs/restorers/basicvsr/basicvsr_RDN_v11_S4_TUF_reds4.py --shape 10 3 128 128


