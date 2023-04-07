<!--
 * @Author: windzu windzu1@gmail.com
 * @Date: 2023-04-07 11:31:51
 * @LastEditors: windzu windzu1@gmail.com
 * @LastEditTime: 2023-04-07 14:36:56
 * @Description: 
 * Copyright (c) 2023 by windzu, All Rights Reserved. 
-->
# Simple Track 二次开发

## 环境准备

**基本环境**
```bash
conda create -n simple_track python=3.8
conda activate simple_track
pip3 install -r requirements.txt
```

**安装 mot_3d**
```bash
conda activate simple_track
pip3 install -e ./
```

## 数据准备

使用作者提供的数据进行测试,下在数据集并解压到根目录下即可，下载地址：[demo_data](https://www.dropbox.com/s/m8vt7t7tqofaoq2/demo_data.zip?dl=0)

## 运行
> 使用`q`退出当前帧的显示，并显示下一帧

```bash
python tools/demo.py \
    --name demo \
    --det_name cp \
    --obj_type vehicle \
    --config_path configs/waymo_configs/vc_kf_giou.yaml \
    --data_folder ./demo_data/ \
    --visualize
```