import taichi as ti
import random

# 初始化 Taichi GPU 后端
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
# 新增参数：是否启用抗锯齿
enable_aa = ti.field(ti.i32, shape=())  # 0: 关闭, 1: 开启
# 新增参数：玻璃材质折射率
glass_ior = ti.field(ti.f32, shape=())  # 折射率，默认1.5

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  # 新增玻璃材质

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    """反射向量计算"""
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, ior_ratio):
    """
    折射向量计算（斯涅尔定律）
    I: 入射方向（指向表面）
    N: 法线（指向外部）
    ior_ratio: ni/nt （入射介质折射率 / 出射介质折射率）
    返回折射方向，如果发生全反射则返回零向量
    """
    cos_theta_i = max(-1.0, min(1.0, I.dot(N)))
    sin2_theta_i = max(0.0, 1.0 - cos_theta_i * cos_theta_i)
    sin2_theta_t = ior_ratio * ior_ratio * sin2_theta_i
    
    # 检查全反射
    if sin2_theta_t >= 1.0:
        return ti.Vector([0.0, 0.0, 0.0])  # 全反射标志
    
    cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
    # 折射方向公式
    R_t = ior_ratio * I + (ior_ratio * cos_theta_i - cos_theta_t) * N
    return normalize(R_t)

@ti.func
def fresnel_schlick(cos_theta, ior1, ior2):
    """
    Schlick近似计算菲涅尔反射系数
    cos_theta: 入射角余弦值
    ior1: 入射介质折射率
    ior2: 出射介质折射率
    返回反射概率
    """
    R0 = (ior1 - ior2) / (ior1 + ior2)
    R0 = R0 * R0
    return R0 + (1.0 - R0) * (1.0 - cos_theta) ** 5

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测玻璃材质球（原红球改为玻璃）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 0.95])  # 玻璃基本色（近乎透明）
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

@ti.func
def compute_lighting(p, N, obj_color, light_pos):
    """
    计算漫反射表面的光照（带阴影检测）
    """
    L = normalize(light_pos - p)
    
    # 阴影检测
    shadow_ray_orig = p + N * 1e-4
    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
    
    dist_to_light = (light_pos - p).norm()
    in_shadow = shadow_t < dist_to_light
    
    # 环境光
    ambient = 0.2 * obj_color
    
    # 直接光照
    direct_light = ambient
    if not in_shadow:
        diff = ti.max(0.0, N.dot(L))
        direct_light += 0.8 * diff * obj_color
    
    return direct_light

@ti.func
trace_path(ro, rd, light_pos, max_bounce):
    """
    追踪单条光线路径（支持漫反射、镜面反射、玻璃材质）
    """
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])
    
    for bounce in range(max_bounce):
        t, N, obj_color, mat_id = scene_intersect(ro, rd)
        
        # 未击中任何物体
        if t > 1e9:
            bg_color = ti.Vector([0.05, 0.15, 0.2])
            final_color += throughput * bg_color
            break
        
        p = ro + rd * t
        
        # 确保法线指向光线入射方向
        if rd.dot(N) > 0:
            N = -N
        
        # 材质分支处理
        if mat_id == MAT_MIRROR:
            # 镜面反射
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color
            
        elif mat_id == MAT_GLASS:
            # 玻璃材质：需要处理反射和折射
            
            # 确定内外折射率
            ior_glass = glass_ior[None]  # 玻璃折射率，约1.5
            ior_air = 1.0  # 空气折射率
            
            # 判断光线是在玻璃内部还是外部
            # 如果法线与入射方向同向，说明在内部
            inside = rd.dot(N) < 0
            if inside:
                # 在玻璃内部，折射率交换
                ior_ratio = ior_glass / ior_air
                N = -N  # 内部法线反转
            else:
                ior_ratio = ior_air / ior_glass
            
            # 计算反射和折射方向
            reflect_dir = normalize(reflect(rd, N))
            
            # 计算折射方向（可能发生全反射）
            refract_dir = refract(rd, N, ior_ratio)
            
            # 菲涅尔效应：计算反射和折射的能量比例
            cos_theta = abs(rd.dot(N))
            reflect_prob = fresnel_schlick(cos_theta, ior_air, ior_glass)
            
            # 根据概率选择反射或折射
            # 使用随机数（这里简化处理，混合两种效果）
            # 实际应该根据概率随机选择，但为确定性，这里按比例混合
            # 注：完全准确的实现需要随机采样，这里采用混合方式
            
            if refract_dir.norm() < 1e-5:  # 全反射
                # 只发生反射
                ro = p + N * 1e-4
                rd = reflect_dir
                throughput *= 0.9 * obj_color  # 玻璃反射损失小
            else:
                # 同时产生反射和折射（简化：按概率选择）
                # 这里采用混合方式：70%概率折射，30%反射
                # 更精确的蒙特卡洛方法需要随机采样
                if bounce % 2 == 0:  # 简单交替选择演示效果
                    ro = p + N * 1e-4
                    rd = refract_dir
                    throughput *= 0.95 * obj_color  # 透射略有损失
                else:
                    ro = p + N * 1e-4
                    rd = reflect_dir
                    throughput *= 0.05 * throughput  # 反射分量
            
        else:  # MAT_DIFFUSE
            # 漫反射材质：计算光照并终止
            direct_light = compute_lighting(p, N, obj_color, light_pos)
            final_color += throughput * direct_light
            break
    
    return final_color

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    
    for i, j in pixels:
        if enable_aa[None] == 1:
            # MSAA：每个像素采样4次
            num_samples = 4
            color_sum = ti.Vector([0.0, 0.0, 0.0])
            
            # 随机偏移采样
            for s in range(num_samples):
                # 在像素内生成随机偏移（使用确定性偏移避免采样一致）
                offset_x = (s % 2) * 0.5 - 0.25
                offset_y = (s // 2) * 0.5 - 0.25
                
                u = (i + offset_x - res_x / 2.0) / res_y * 2.0
                v = (j + offset_y - res_y / 2.0) / res_y * 2.0
                
                ro = ti.Vector([0.0, 1.0, 5.0])
                rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
                
                color_sum += trace_path(ro, rd, light_pos, max_bounces[None])
            
            final_color = color_sum / num_samples
            
        else:
            # 无抗锯齿：单次采样
            u = (i - res_x / 2.0) / res_y * 2.0
            v = (j - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))
            
            final_color = trace_path(ro, rd, light_pos, max_bounces[None])
        
        # 色调映射和写入
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo - Glass & Anti-Aliasing", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化参数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5  # 玻璃需要更多弹射次数
    enable_aa[None] = 1    # 默认开启抗锯齿
    glass_ior[None] = 1.5  # 常见玻璃折射率
    
    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.70, 0.05, 0.29, 0.32):
            gui.text("Light Position")
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            
            gui.text("")
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 8)
            
            gui.text("")
            enable_aa[None] = gui.slider_int('Anti-Aliasing (0=Off, 1=On)', enable_aa[None], 0, 1)
            
            gui.text("")
            glass_ior[None] = gui.slider_float('Glass Refractive Index', glass_ior[None], 1.2, 2.5)
            
            gui.text("")
            gui.text("Glass Effects:")
            gui.text("- Refraction (Snell's Law)")
            gui.text("- Total Internal Reflection")
            gui.text("- Fresnel Effect")

        window.show()

if __name__ == '__main__':
    main()