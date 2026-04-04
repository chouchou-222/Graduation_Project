# SFFNet 逐层张量尺寸流图（Vaihingen 配置）

本文按源码静态推导张量尺寸，覆盖训练常用输入 `512x512` 与验证常用输入 `1024x1024`。

## 1. 关键前提（来自配置与数据增强）

- 类别数 `num_classes=6`：`GeoSeg/config/vaihingen/sffnet.py`
- 训练集增强包含 `SmartCropV1(crop_size=512)`：`GeoSeg/geoseg/datasets/vaihingen_dataset.py`
- 验证增强 `val_aug` 不裁剪，只做归一化：`GeoSeg/geoseg/datasets/vaihingen_dataset.py`

因此：

- 训练时常见输入：`img [B, 3, 512, 512]`, `mask [B, 512, 512]`
- 验证时常见输入：`img [B, 3, 1024, 1024]`, `mask [B, 1024, 1024]`

## 2. 通用尺寸公式（输入为 `[B, 3, H, W]`）

### 2.1 Backbone 输出（ConvNeXt Tiny, `out_indices=(0,1,2,3)`)

- `res1: [B, 96,  H/4,  W/4]`
- `res2: [B, 192, H/8,  W/8]`
- `res3: [B, 384, H/16, W/16]`
- `res4: [B, 768, H/32, W/32]`

### 2.2 通道对齐 + 上采样到同尺度

- `conv2(res2) -> [B, 96, H/8,  W/8]`
- `conv3(res3) -> [B, 96, H/16, W/16]`
- `conv4(res4) -> [B, 96, H/32, W/32]`
- 三者上采样到 `res1` 尺寸后：
  - `res2_u/res3_u/res4_u -> [B, 96, H/4, W/4]`
- 拼接：
  - `middleres = cat([res2_u,res3_u,res4_u], dim=1)`
  - `middleres -> [B, 288, H/4, W/4]`

### 2.3 FMS（频域 + 空域映射）

输入 `middleres [B, 288, H/4, W/4]`：

- 小波分解 `DWTForward(J=1)`：
  - 低频 `yL_raw: [B, 288, H/8, W/8]`
  - 高频三子带 `yH_raw: [B, 288, 3, H/8, W/8]`
- 高频拼接后 `cat(HL,LH,HH)`：
  - `yH_cat: [B, 864, H/8, W/8]`
- `1x1` 降回 288 通道后再输出头：
  - `yL: [B, 96, H/8, W/8]`
  - `yH: [B, 96, H/8, W/8]`
- 全局分支 `glb`、局部分支 `local`（都含 stride=2 下采样）：
  - `glb:   [B, 96, H/8, W/8]`
  - `local: [B, 96, H/8, W/8]`

FMS 最终返回：

- `(fusefeature_L, fusefeature_H, glb, local)`
- 全部都是 `[B, 96, H/8, W/8]`

### 2.4 MDAF 双域对齐 + WF 融合

- `MDAF_L(fusefeature_L, glb)   -> [B, 96, H/8, W/8]`
- `MDAF_H(fusefeature_H, local) -> [B, 96, H/8, W/8]`
- `WF1(glb, local)              -> [B, 96, H/8, W/8]`

### 2.5 解码与输出

- `down(middleres)`（`1x1`, 288->96）：
  - `[B, 96, H/4, W/4]`
- `WF1` 输出上采样到 `H/4` 再与上式相加：
  - `[B, 96, H/4, W/4]`
- `WF2(., res1)`：
  - `[B, 96, H/4, W/4]`
- `segmentation_head`：
  - `[B, 6, H/4, W/4]`
- 最后双线性插值回原图：
  - `logits: [B, 6, H, W]`

## 3. 两个常用场景的完整流图

### 3.1 训练常见输入：`[B, 3, 512, 512]`

1. Backbone  
`res1 [B,96,128,128]`  
`res2 [B,192,64,64]`  
`res3 [B,384,32,32]`  
`res4 [B,768,16,16]`

2. 通道对齐 + 上采样拼接  
`conv2/3/4 -> [B,96,64,64]/[B,96,32,32]/[B,96,16,16]`  
上采样后都变 `[B,96,128,128]`  
`middleres = cat -> [B,288,128,128]`

3. FMS  
`yL_raw [B,288,64,64]`  
`yH_raw [B,288,3,64,64]`  
`yH_cat [B,864,64,64]`  
输出 `fusefeature_L/fusefeature_H/glb/local` 全部 `[B,96,64,64]`

4. MDAF + WF1  
`glb [B,96,64,64]`  
`local [B,96,64,64]`  
`WF1 -> [B,96,64,64]`

5. 解码  
`down(middleres) -> [B,96,128,128]`  
`WF1上采样 -> [B,96,128,128]`  
相加后 `[B,96,128,128]`  
`WF2 -> [B,96,128,128]`  
`seg head -> [B,6,128,128]`  
上采样回原图：`[B,6,512,512]`

6. Loss 对接  
`mask [B,512,512]`  
`Loss(logits, mask) -> 标量`

### 3.2 验证常见输入：`[B, 3, 1024, 1024]`

1. Backbone  
`res1 [B,96,256,256]`  
`res2 [B,192,128,128]`  
`res3 [B,384,64,64]`  
`res4 [B,768,32,32]`

2. 拼接  
`middleres [B,288,256,256]`

3. FMS 输出  
`fusefeature_L/fusefeature_H/glb/local` 全部 `[B,96,128,128]`

4. 解码输出  
`seg head [B,6,256,256]`  
最终 `logits [B,6,1024,1024]`

5. Loss 对接  
`mask [B,1024,1024]`  
`Loss(logits, mask) -> 标量`

## 4. 训练脚本中的尺寸衔接

- `training_step`: `img, mask = batch['img'], batch['gt_semantic_seg']`
- `prediction = self.net(img)`，再 `loss = self.trainloss(prediction, mask)`

所以尺寸约束是：

- `prediction.shape == [B, num_classes, H, W]`
- `mask.shape == [B, H, W]`

在 Vaihingen 配置里 `num_classes=6`，可直接与上面两种输入场景对应。

## 5. 对齐到源码（便于你对照阅读）

- SFFNet 主流程：`GeoSeg/geoseg/models/SFFNet/SFFNet.py`
- FMS 小波与双分支：`GeoSeg/geoseg/models/SFFNet/FMS.py`
- MDAF 双域对齐：`GeoSeg/geoseg/models/SFFNet/MDAF.py`
- Vaihingen 数据增强/尺寸来源：`GeoSeg/geoseg/datasets/vaihingen_dataset.py`
- 训练步与 loss 对接：`GeoSeg/train_supervision.py`
