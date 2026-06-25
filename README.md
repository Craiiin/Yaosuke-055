姚苏珂-202411081026-计算机师范
# 光线追踪实验报告

## 一、实验概述

本实验基于 Whitted-Style 光线追踪模型，使用 Taichi 语言在 GPU 上实现了一个完整的光线追踪渲染器。实验实现了硬阴影、镜面反射、玻璃材质折射以及多重采样抗锯齿等核心图形学效果。

## 二、实验目标完成情况

### 2.1 理论理解：光线投射 vs 光线追踪

| 特性 | 光线投射 (Ray Casting) | 光线追踪 (Ray Tracing) |
|------|----------------------|----------------------|
| 光线传播 | 仅主射线，击中即停止 | 可发射次级射线继续传播 |
| 反射效果 | ❌ 无法实现 | ✅ 可实现镜面反射 |
| 折射效果 | ❌ 无法实现 | ✅ 可实现玻璃材质 |
| 阴影表现 | 可通过阴影射线实现 | 可通过阴影射线实现 |

在本实验中，漫反射球击中后停止传播（光线投射特性），而镜面球和玻璃球会继续发射次级射线（光线追踪特性）。

### 2.2 全局光照效果

#### 硬阴影实现
从交点向光源发射暗影射线，检测路径上是否有遮挡：

```python
shadow_ray_orig = p + N * 1e-4
shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
dist_to_light = (light_pos - p).norm()
in_shadow = shadow_t < dist_to_light
```

#### 理想镜面反射
根据反射定律计算反射方向：

$$\mathbf{R} = \mathbf{L}_{in} - 2(\mathbf{L}_{in} \cdot \mathbf{N})\mathbf{N}$$

#### 玻璃材质与折射
根据斯涅尔定律计算折射方向：

$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

### 2.3 GPU 编程思维：迭代代替递归

将传统递归算法改写为适合 GPU 的迭代循环：

```python
for bounce in range(max_bounces):
    if 漫反射材质:
        计算光照后 break
    elif 镜面/玻璃材质:
        更新光线起点和方向，继续循环
```

## 三、场景搭建

### 3.1 几何体定义

| 物体 | 位置 | 半径 | 材质 | 颜色 |
|------|------|------|------|------|
| 玻璃球 | (-1.2, 0.0, 0.0) | 1.0 | 玻璃 | 透明带折射 |
| 镜面球 | (1.2, 0.0, 0.0) | 1.0 | 镜面反射 | 银色 |
| 无限大地平面 | y = -1.0 | - | 漫反射 | 黑白棋盘格 |

### 3.2 棋盘格纹理实现

```python
ix = ti.floor(p.x * grid_scale)
iz = ti.floor(p.z * grid_scale)
if (ix + iz) % 2 == 0:
    hit_c = ti.Vector([0.3, 0.3, 0.3])  # 深色格子
else:
    hit_c = ti.Vector([0.8, 0.8, 0.8])  # 浅色格子
```

## 四、核心技术实现

### 4.1 反射与折射

```python
@ti.func
def refract(I, N, ior_ratio):
    cos_theta_i = max(-1.0, min(1.0, I.dot(N)))
    sin2_theta_i = max(0.0, 1.0 - cos_theta_i * cos_theta_i)
    sin2_theta_t = ior_ratio * ior_ratio * sin2_theta_i
    
    is_total_internal = sin2_theta_t >= 1.0  # 全反射检测
    result_dir = ti.Vector([0.0, 0.0, 0.0])
    
    if not is_total_internal:
        cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
        result_dir = ior_ratio * I + (ior_ratio * cos_theta_i - cos_theta_t) * N
        result_dir = normalize(result_dir)
    
    return result_dir, is_total_internal
```

### 4.2 菲涅尔效应

使用 Schlick 近似计算反射系数：

$$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$$

其中 $R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2$

### 4.3 多重采样抗锯齿 (MSAA)

每个像素均匀采样 4 个子像素区域：

```python
# 采样点布局
offsets = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]

for offset_x, offset_y in offsets:
    u = (i + offset_x - width/2) / height * 2
    v = (j + offset_y - height/2) / height * 2
    color_sum += trace_path(ro, rd, ...)

final_color = color_sum / 4
```

### 4.4 阴影痤疮修复

射线起点沿法线方向偏移微小量，防止与自身表面相交：

```python
ro = p + N * 1e-4   # 反射射线起点
shadow_ray_orig = p + N * 1e-4  # 阴影射线起点
```

## 五、用户交互界面

### 5.1 控件说明

| 控件 | 范围 | 默认值 | 功能 |
|------|------|--------|------|
| Light X | -5.0 ~ 5.0 | 2.0 | 光源 X 坐标 |
| Light Y | 1.0 ~ 8.0 | 4.0 | 光源 Y 坐标 |
| Light Z | -5.0 ~ 5.0 | 3.0 | 光源 Z 坐标 |
| Max Bounces | 1 ~ 8 | 5 | 最大光线弹射次数 |
| Anti-Aliasing | 0 ~ 1 | 1 | 抗锯齿开关 |
| Glass IOR | 1.2 ~ 2.5 | 1.5 | 玻璃折射率 |

### 5.2 参数效果说明

**Max Bounces 对比**：
- **Bounces = 1**：光线击中漫反射物体后停止，镜面球显示黑色
- **Bounces = 3**：镜面球出现一次反射，可看到地板倒影
- **Bounces = 5**：玻璃球出现完整折射和反射效果

**Anti-Aliasing 效果**：
- **关闭**：边缘有明显锯齿，渲染速度快
- **开启**：边缘平滑连续，渲染速度约为 4 倍
<img width="480" height="395" alt="NRy00ToN_converted" src="https://github.com/user-attachments/assets/3aa77832-3d0d-4c27-94f4-954bdefd445f" />

## 六、运行指南

### 6.1 环境要求

- Python 3.8+
- Taichi 1.7.4+

### 6.2 安装依赖

```bash
pip install taichi
```

### 6.3 运行程序

```bash
python ray_tracing.py
```

### 6.4 操作说明

1. 启动后自动开始渲染
2. 使用右侧控制面板调整参数
3. 观察场景效果的实时变化

## 七、技术参数

| 参数 | 值 |
|------|-----|
| 分辨率 | 800 × 600 |
| 摄像机位置 | (0, 1, 5) |
| 摄像机方向 | (0, -0.2, -1) |
| 镜面反射率 | 0.8 |
| 玻璃反射率 | 0.9 |
| 玻璃透射率 | 0.95 |
| 环境光强度 | 0.2 |
| 漫反射强度 | 0.8 |
| 空气折射率 | 1.0 |
| 玻璃默认折射率 | 1.5 |

## 八、实验总结

通过本实验，我成功实现了：

1. ✅ **理论理解**：掌握了光线投射与光线追踪的本质区别
2. ✅ **全局光照**：实现了硬阴影、镜面反射和玻璃折射
3. ✅ **GPU 编程思维**：将递归算法改写为迭代循环
4. ✅ **性能优化**：实现 MSAA 抗锯齿，平衡画质与性能
5. ✅ **交互体验**：构建了实时可调参数的控制面板

### 关键技术难点与解决方案

| 难点 | 解决方案 |
|------|----------|
| Shadow Acne | 射线起点沿法线偏移 1e-4 |
| 全反射模拟 | 检测临界角，条件分支处理 |
| Taichi 分支内返回限制 | 函数返回元组替代条件返回 |
| GPU 数组索引限制 | 使用 if-elif 分支展开索引 |


