/*
 * Vulkan示例 - 绘制多个动画齿轮（模拟glxgears外观）
 *
 * 所有齿轮使用单一索引、顶点和统一缓冲区，展示了Vulkan最佳实践：将缓冲区/内存分配数量保持在最低水平
 * 我们利用索引偏移和实例索引在绘制时访问每个齿轮的缓冲区数据
 */

#include "vulkanexamplebase.h"

const uint32_t numGears = 3;

// 用于在构造时传递齿轮定义
struct GearDefinition {
    float innerRadius;    // 内半径
    float outerRadius;    // 外半径
    float width;         // 宽度
    int numTeeth;        // 齿数
    float toothDepth;    // 齿深
    glm::vec3 color;     // 颜色
    glm::vec3 pos;       // 位置
    float rotSpeed;      // 旋转速度
    float rotOffset;     // 旋转偏移
};

/*
 * 齿轮类
 * 包含单个齿轮的属性及生成顶点和索引的函数
 */
class Gear {
public:
    // 用于渲染齿轮的顶点数据结构
    struct Vertex {
        glm::vec3 position;  // 位置
        glm::vec3 normal;    // 法线
        glm::vec3 color;     // 颜色
    };

    glm::vec3 color;         // 颜色
    glm::vec3 pos;           // 位置
    float rotSpeed{ 0.0f };  // 旋转速度
    float rotOffset{ 0.0f }; // 旋转偏移
    // 绘制时用于访问单一缓冲区的偏移量
    uint32_t indexCount{ 0 }; // 索引数量
    uint32_t indexStart{ 0 }; // 索引起始位置

// 为当前齿轮生成顶点和索引数据
    void generate(GearDefinition& gearDefinition, std::vector<Vertex>& vertexBuffer, std::vector<uint32_t>& indexBuffer) {
        // 从齿轮定义中获取属性
        this->color = gearDefinition.color;
        this->pos = gearDefinition.pos;
        this->rotOffset = gearDefinition.rotOffset;
        this->rotSpeed = gearDefinition.rotSpeed;

        // 局部变量声明
        int i; // 循环计数器
        float r0, r1, r2; // 三个关键半径：内径、齿根半径、齿顶半径
        float ta, da; // 当前角度和角度增量
        float u1, v1, u2, v2, len; // 用于计算法向量的临时变量
        float cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da; // 不同角度的余弦值
        float sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da; // 不同角度的正弦值
        int32_t ix0, ix1, ix2, ix3, ix4, ix5; // 顶点索引

        // 记录当前索引缓冲区的起始位置
        indexStart = static_cast<uint32_t>(indexBuffer.size());

        // 计算三个关键半径
        r0 = gearDefinition.innerRadius; // 内径
        r1 = gearDefinition.outerRadius - gearDefinition.toothDepth / 2.0f; // 齿根半径
        r2 = gearDefinition.outerRadius + gearDefinition.toothDepth / 2.0f; // 齿顶半径
        // 计算角度增量（每个齿轮齿分为4段）
        da = static_cast<float>(2.0 * M_PI / gearDefinition.numTeeth / 4.0);

        glm::vec3 normal; // 法向量

        // 定义添加三角形的lambda函数
        auto addFace = [&indexBuffer](int a, int b, int c) {
            indexBuffer.push_back(a);
            indexBuffer.push_back(b);
            indexBuffer.push_back(c);
        };

        // 定义添加顶点的lambda函数
        auto addVertex = [this, &vertexBuffer](float x, float y, float z, glm::vec3 normal) {
            Vertex v{};
            v.position = { x, y, z };
            v.normal = normal;
            v.color = this->color;
            vertexBuffer.push_back(v);
            return static_cast<int32_t>(vertexBuffer.size()) - 1;
        };

        // 遍历每个齿轮齿
        for (i = 0; i < gearDefinition.numTeeth; i++) {
            // 计算当前齿的起始角度
            ta = i * static_cast<float>(2.0 * M_PI / gearDefinition.numTeeth);

            // 预计算当前齿轮齿不同角度的三角函数值（用于生成齿轮几何形状）
            // ta (theta angle): 当前齿轮齿的起始角度（弧度制）
            // da (delta angle): 角度增量，将每个齿轮齿分为4段进行计算

            // 计算当前角度ta的余弦值（齿根起点）
            cos_ta = cos(ta);

            // 计算ta + da角度的余弦值（齿根到齿顶的过渡点1）
            cos_ta_1da = cos(ta + da);

            // 计算ta + 2*da角度的余弦值（齿顶中点）
            cos_ta_2da = cos(ta + 2.0f * da);

            // 计算ta + 3*da角度的余弦值（齿顶到齿根的过渡点2）
            cos_ta_3da = cos(ta + 3.0f * da);

            // 计算ta + 4*da角度的余弦值（下一个齿的齿根起点）
            cos_ta_4da = cos(ta + 4.0f * da);

            // 计算当前角度ta的正弦值（齿根起点）
            sin_ta = sin(ta);

            // 计算ta + da角度的正弦值（齿根到齿顶的过渡点1）
            sin_ta_1da = sin(ta + da);

            // 计算ta + 2*da角度的正弦值（齿顶中点）
            sin_ta_2da = sin(ta + 2.0f * da);

            // 计算ta + 3*da角度的正弦值（齿顶到齿根的过渡点2）
            sin_ta_3da = sin(ta + 3.0f * da);

            // 计算ta + 4*da角度的正弦值（下一个齿的齿根起点）
            sin_ta_4da = sin(ta + 4.0f * da);

            /* 技术细节说明：
            每个齿轮齿被划分为4个区段：
            - 齿根圆弧段（ta到ta+da）
            - 齿侧斜边1（ta+da到ta+2da）
            - 齿顶圆弧段（ta+2da到ta+3da）
            - 齿侧斜边2（ta+3da到ta+4da）

            // 计算齿轮侧面法向量（用于光照计算和表面着色）

            // 1. 计算第一段齿侧面（齿根到齿顶过渡段）的向量：
            //    - 起点：齿根点(r1*cos_ta, r1*sin_ta)
            //    - 终点：齿侧过渡点1(r2*cos_ta_1da, r2*sin_ta_1da)
            u1 = r2 * cos_ta_1da - r1 * cos_ta;  // 向量x分量 = 终点x坐标 - 起点x坐标
            v1 = r2 * sin_ta_1da - r1 * sin_ta;  // 向量y分量 = 终点y坐标 - 起点y坐标

            // 2. 归一化处理（转换为单位向量）：
            len = sqrt(u1 * u1 + v1 * v1);  // 计算向量长度（勾股定理）
            u1 /= len;  // x分量归一化
            v1 /= len;  // y分量归一化
            // 现在(u1, v1)是单位方向向量

            // 3. 计算第二段齿侧面（齿顶到齿根过渡段）的向量：
            //    - 起点：齿顶中点(r2*cos_ta_2da, r2*sin_ta_2da)
            //    - 终点：齿侧过渡点2(r1*cos_ta_3da, r1*sin_ta_3da)
            u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;  // 向量x分量
            v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;  // 向量y分量
            // 注：此处未归一化，因为后续可能根据使用场景决定是否归一化

            /* 技术说明：
            1. 这些向量将用于：
               - 生成齿侧面的法线向量（通过交换xy分量并取负）
               - 计算光照反射效果（法线影响明暗）
               - 确保齿轮侧面平滑着色

            2. 向量方向说明：
               - 第一段向量方向：从齿根指向齿侧凸起
               - 第二段向量方向：从齿顶指向齿侧凹陷

            3. 几何关系：
               ┌───────────────┐
               │      /│       │
               │     / │       │  ← 齿侧面
               │    /  │       │
               │   •───•       │  ← u1/v1向量
               │  /    │       │
               └───────────────┘
            */

            /* 前表面（齿轮的正面） */
            // 定义法向量 - 指向Z轴正方向（垂直于齿轮前表面）
            normal = glm::vec3(0.0f, 0.0f, 1.0f);
            ix0 = addVertex(
                    r0 * cos_ta,            // X坐标：内圆半径*cos(角度)
                    r0 * sin_ta,            // Y坐标：内圆半径*sin(角度)
                    gearDefinition.width * 0.5f, // Z坐标：齿轮宽度的一半（前表面）
                    normal                  // 法向量
            );
            ix1 = addVertex(
                    r1 * cos_ta,            // 外圆半径*cos(角度)
                    r1 * sin_ta,            // 外圆半径*sin(角度)
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix2 = addVertex(r0 * cos_ta, r0 * sin_ta, gearDefinition.width * 0.5f, normal);
            ix3 = addVertex(
                    r1 * cos_ta_3da,        // 外圆半径*cos(角度+3*增量)
                    r1 * sin_ta_3da,        // 外圆半径*sin(角度+3*增量)
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix4 = addVertex(
                    r0 * cos_ta_4da,        // 内圆半径*cos(角度+4*增量)
                    r0 * sin_ta_4da,        // 内圆半径*sin(角度+4*增量)
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix5 = addVertex(
                    r1 * cos_ta_4da,        // 外圆半径*cos(角度+4*增量)
                    r1 * sin_ta_4da,        // 外圆半径*sin(角度+4*增量)
                    gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);
            addFace(ix2, ix3, ix4);
            addFace(ix3, ix5, ix4);

            /* 齿轮前表面的齿形部分（齿顶区域）*/
            // 法向量依然朝前（Z轴正方向）
            normal = glm::vec3(0.0f, 0.0f, 1.0f);
            ix0 = addVertex(
                    r1 * cos_ta,                // X坐标：外圆半径r1*cos(当前角度)
                    r1 * sin_ta,                // Y坐标
                    gearDefinition.width * 0.5f, // Z坐标：前表面
                    normal
            );
            ix1 = addVertex(
                    r2 * cos_ta_1da,            // 使用更大的半径r2（齿顶半径）
                    r2 * sin_ta_1da,            // 角度增加1个增量da
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix2 = addVertex(
                    r1 * cos_ta_3da,            // 外圆半径
                    r1 * sin_ta_3da,            // 角度增加3个增量
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix3 = addVertex(
                    r2 * cos_ta_2da,            // 齿顶半径
                    r2 * sin_ta_2da,            // 角度增加2个增量
                    gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);

            /* 后表面（齿轮的背面）*/
            // 定义法向量 - 指向Z轴负方向（垂直于齿轮背面）
            normal = glm::vec3(0.0f, 0.0f, -1.0f);
            ix0 = addVertex(
                    r1 * cos_ta,                // X坐标：外圆半径*cos(角度)
                    r1 * sin_ta,                // Y坐标
                    -gearDefinition.width * 0.5f, // Z坐标：负的半宽（背面）
                    normal
            );
            ix1 = addVertex(
                    r0 * cos_ta,                // 内圆半径*cos(角度)
                    r0 * sin_ta,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix2 = addVertex(
                    r1 * cos_ta_3da,
                    r1 * sin_ta_3da,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix3 = addVertex(
                    r0 * cos_ta,
                    r0 * sin_ta,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix4 = addVertex(
                    r1 * cos_ta_4da,
                    r1 * sin_ta_4da,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix5 = addVertex(
                    r0 * cos_ta_4da,
                    r0 * sin_ta_4da,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);
            addFace(ix2, ix3, ix4);
            addFace(ix3, ix5, ix4);

            /* 齿轮后表面的齿形部分（背面齿顶区域）*/
            // 法向量依然朝后（Z轴负方向）
            normal = glm::vec3(0.0f, 0.0f, -1.0f);
            ix0 = addVertex(
                    r1 * cos_ta_3da,            // X坐标：外圆半径*cos(角度+3*增量)
                    r1 * sin_ta_3da,            // Y坐标
                    -gearDefinition.width * 0.5f, // Z坐标：负的半宽（背面）
                    normal
            );
            ix1 = addVertex(
                    r2 * cos_ta_2da,            // 使用齿顶半径r2
                    r2 * sin_ta_2da,            // 角度+2*增量
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix2 = addVertex(
                    r1 * cos_ta,                // 外圆半径r1
                    r1 * sin_ta,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            ix3 = addVertex(
                    r2 * cos_ta_1da,            // 齿顶半径r2
                    r2 * sin_ta_1da,            // 角度+1*增量
                    -gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);
            normal = glm::vec3(v1, -u1, 0.0f);
            ix0 = addVertex(
                    r1 * cos_ta,                // X坐标：外圆半径r1
                    r1 * sin_ta,                // Y坐标
                    gearDefinition.width * 0.5f, // Z坐标：正半宽（前表面）
                    normal                      // 侧面法向量
            );
            ix1 = addVertex(
                    r1 * cos_ta,
                    r1 * sin_ta,
                    -gearDefinition.width * 0.5f, // Z坐标：负半宽（后表面）
                    normal
            );
            ix2 = addVertex(
                    r2 * cos_ta_1da,            // 使用更大的齿顶半径r2
                    r2 * sin_ta_1da,            // 角度增加1个增量
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix3 = addVertex(
                    r2 * cos_ta_1da,
                    r2 * sin_ta_1da,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);

            /* 齿轮齿顶侧面（第二个侧面）*/
            // 定义法向量 - 沿齿轮径向向外（从圆心指向齿顶）
            normal = glm::vec3(cos_ta, sin_ta, 0.0f);  // 使用当前角度的单位向量作为法线
            ix0 = addVertex(
                    r2 * cos_ta_1da,            // X坐标：齿顶半径r2*cos(角度+da)
                    r2 * sin_ta_1da,            // Y坐标
                    gearDefinition.width * 0.5f, // Z坐标：正半宽（前表面）
                    normal                      // 径向法向量
            );
            ix1 = addVertex(
                    r2 * cos_ta_1da,
                    r2 * sin_ta_1da,
                    -gearDefinition.width * 0.5f, // Z坐标：负半宽（后表面）
                    normal
            );
            ix2 = addVertex(
                    r2 * cos_ta_2da,            // 齿顶半径r2*cos(角度+2da)
                    r2 * sin_ta_2da,
                    gearDefinition.width * 0.5f,
                    normal
            );
            ix3 = addVertex(
                    r2 * cos_ta_2da,
                    r2 * sin_ta_2da,
                    -gearDefinition.width * 0.5f,
                    normal
            );
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);

            // 第三个侧面
            normal = glm::vec3(v2, -u2, 0.0f);
            ix0 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, gearDefinition.width * 0.5f, normal);
            ix1 = addVertex(r2 * cos_ta_2da, r2 * sin_ta_2da, -gearDefinition.width * 0.5f, normal);
            ix2 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
            ix3 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);

            // 第四个侧面（齿根）
            normal = glm::vec3(cos_ta, sin_ta, 0.0f);
            ix0 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, gearDefinition.width * 0.5f, normal);
            ix1 = addVertex(r1 * cos_ta_3da, r1 * sin_ta_3da, -gearDefinition.width * 0.5f, normal);
            ix2 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, gearDefinition.width * 0.5f, normal);
            ix3 = addVertex(r1 * cos_ta_4da, r1 * sin_ta_4da, -gearDefinition.width * 0.5f, normal);
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);

            /* 内侧圆柱面 */
            ix0 = addVertex(r0 * cos_ta, r0 * sin_ta, -gearDefinition.width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0f));
            ix1 = addVertex(r0 * cos_ta, r0 * sin_ta, gearDefinition.width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0f));
            ix2 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, -gearDefinition.width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0f));
            ix3 = addVertex(r0 * cos_ta_4da, r0 * sin_ta_4da, gearDefinition.width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0f));
            addFace(ix0, ix1, ix2);
            addFace(ix1, ix3, ix2);
        }

        // 计算生成的索引数量
        indexCount = static_cast<uint32_t>(indexBuffer.size()) - indexStart;
    }
};

/*
 * Vulkan示例主类
 */
class VulkanExample : public VulkanExampleBase {
public:
    std::vector<Gear> gears{}; // 齿轮集合

    // Vulkan资源句柄
    VkPipeline pipeline{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };
    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };

    // 使用单一缓冲区存储所有齿轮数据（最佳实践）
    vks::Buffer vertexBuffer;  // 顶点缓冲区
    vks::Buffer indexBuffer;   // 索引缓冲区

    // 统一缓冲区数据结构
    struct UniformData {
        glm::mat4 projection;  // 投影矩阵
        glm::mat4 view;        // 视图矩阵
        glm::vec4 lightPos;    // 光源位置
        glm::mat4 model[numGears]; // 每个齿轮的模型矩阵
    } uniformData;

    vks::Buffer uniformBuffer; // 统一缓冲区

// 构造函数：初始化Vulkan齿轮示例
    VulkanExample() : VulkanExampleBase() {  // 调用基类构造函数初始化Vulkan基础环境

        // 设置窗口标题
        title = "Vulkan齿轮示例";  // 显示在窗口标题栏的文字

        // 配置相机参数
        camera.type = Camera::CameraType::lookat;  // 使用lookat相机模式（固定目标点）
        camera.setPosition(glm::vec3(0.0f, 2.5f, -16.0f));  // 设置相机初始位置（X,Y,Z坐标）
        camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));    // 设置相机初始旋转角度（绕X,Y,Z轴）
        camera.setPerspective(
                60.0f,                     // 垂直视野角度（FOV）
                (float)width / (float)height, // 宽高比（根据窗口尺寸计算）
                0.001f,                    // 近裁剪面距离
                256.0f                     // 远裁剪面距离
        );

        // 调整动画速度
        timerSpeed *= 0.25f;  // 将默认动画速度降低到25%（使齿轮旋转更慢，便于观察）

        /* 参数详解：
        1. 相机位置 (0.0f, 2.5f, -16.0f):
           - Y轴2.5单位：从上方俯视齿轮组
           - Z轴-16单位：与齿轮保持适当观察距离

        2. 透视投影参数：
           - 60度视野：接近人眼自然视角
           - 近裁剪面0.001：可看清近距离细节
           - 远裁剪面256：足够容纳场景深度

        3. 动画速度调整：
           - 原始timerSpeed通常为1.0
           - 乘以0.25后变为0.25倍速
           - 避免齿轮旋转过快影响观察
        */
    }

    ~VulkanExample() {
        if (device) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            indexBuffer.destroy();
            vertexBuffer.destroy();
            uniformBuffer.destroy();
        }
    }

    // 准备齿轮数据 - 初始化三个齿轮的几何数据并上传到GPU
    void prepareGears() {
        // 1. 定义三个齿轮的基本参数
        std::vector<GearDefinition> gearDefinitions(3); // 创建包含3个齿轮定义的数组

        /* 第一个齿轮：大型红色齿轮 */
        gearDefinitions[0] = {
                1.0f,   // 内半径(中心孔半径)
                4.0f,   // 外半径(齿轮整体半径)
                1.0f,   // 齿轮厚度
                20,     // 齿数
                0.7f,   // 齿深(齿的高度)
                {1.0f, 0.0f, 0.0f}, // 颜色(RGB红色)
                {-3.0f, 0.0f, 0.0f}, // 位置(XYZ坐标)
                1.0f,   // 旋转速度(正值为顺时针)
                0.0f    // 旋转初始偏移(角度)
        };

        /* 第二个齿轮：中型绿色齿轮 */
        gearDefinitions[1] = {
                0.5f,   // 内半径
                2.0f,   // 外半径
                2.0f,   // 厚度(比第一个齿轮厚)
                10,     // 齿数(比第一个齿轮少)
                0.7f,   // 齿深
                {0.0f, 1.0f, 0.2f}, // 绿色带点蓝
                {3.1f, 0.0f, 0.0f}, // 位置(与第一个齿轮啮合)
                -2.0f,  // 逆时针旋转(速度是第一个齿轮的两倍)
                -9.0f   // 初始偏移(确保齿轮齿正确啮合)
        };

        /* 第三个齿轮：小型蓝色齿轮 */
        gearDefinitions[2] = {
                1.3f,   // 较大的内半径
                2.0f,   // 外半径
                0.5f,   // 较薄的厚度
                10,     // 齿数
                0.7f,   // 齿深
                {0.0f, 0.0f, 1.0f}, // 纯蓝色
                {-3.1f, -6.2f, 0.0f}, // 位置(与第一个齿轮垂直啮合)
                -2.0f,  // 逆时针旋转
                -30.0f  // 较大的初始偏移
        };

        // 2. 生成所有齿轮的几何数据
        std::vector<Gear::Vertex> vertices{}; // 顶点数据容器
        std::vector<uint32_t> indices{};      // 索引数据容器

        gears.resize(gearDefinitions.size()); // 调整齿轮数组大小
        for (int32_t i = 0; i < gears.size(); i++) {
            // 为每个齿轮生成顶点和索引数据，并添加到总缓冲区
            gears[i].generate(gearDefinitions[i], vertices, indices);
        }

        // 3. 计算缓冲区大小
        size_t vertexBufferSize = vertices.size() * sizeof(Gear::Vertex); // 顶点缓冲区大小
        size_t indexBufferSize = indices.size() * sizeof(uint32_t);       // 索引缓冲区大小

        // 4. 创建临时暂存缓冲区(用于CPU到GPU的数据传输)
        vks::Buffer vertexStaging, indexStaging; // 暂存缓冲区对象

        // 创建顶点暂存缓冲区(主机可见)
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // 作为传输源
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // 主机可见内存
                &vertexStaging,
                vertexBufferSize,
                vertices.data()); // 传入顶点数据

        // 创建索引暂存缓冲区(主机可见)
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                &indexStaging,
                indexBufferSize,
                indices.data()); // 传入索引数据

        // 5. 创建设备本地缓冲区(高性能GPU内存)
        // 顶点缓冲区(设备本地)
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // 作为顶点缓冲区和传输目标
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // 设备本地内存(不可被CPU直接访问)
                &vertexBuffer,
                vertexBufferSize);

        // 索引缓冲区(设备本地)
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // 作为索引缓冲区和传输目标
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &indexBuffer,
                indexBufferSize);

        // 6. 使用命令缓冲区执行数据传输
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        // 复制顶点数据
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertexBuffer.buffer, 1, &copyRegion);

        // 复制索引数据
        copyRegion.size = indexBufferSize;
        vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indexBuffer.buffer, 1, &copyRegion);

        // 提交并等待命令缓冲区完成
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

        // 7. 清理临时资源
        vertexStaging.destroy(); // 销毁顶点暂存缓冲区
        indexStaging.destroy();  // 销毁索引暂存缓冲区
    }

    // 设置描述符 - 配置描述符池、布局和集合，用于着色器与缓冲区的绑定
    void setupDescriptors() {
        // 1. 配置描述符池大小
        // 指定我们需要的描述符类型和数量
        std::vector<VkDescriptorPoolSize> poolSizes = {
                // 统一缓冲区描述符：1个（用于存储全局矩阵和齿轮模型矩阵）
                vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
        };

        // 2. 创建描述符池
        // 参数说明：
        // poolSizes: 描述符类型和数量配置
        // gears.size(): 最大描述符集数量（这里设置为齿轮数量）
        VkDescriptorPoolCreateInfo descriptorPoolInfo =
                vks::initializers::descriptorPoolCreateInfo(poolSizes, static_cast<uint32_t>(gears.size()));
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // 3. 配置描述符布局绑定
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // 绑定0：顶点着色器统一缓冲区
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // 描述符类型：统一缓冲区
                        VK_SHADER_STAGE_VERTEX_BIT,        // 着色器阶段：顶点着色器
                        0)                                 // 绑定点：0
        };

        // 4. 创建描述符集布局
        // 描述符集布局定义了描述符的结构
        VkDescriptorSetLayoutCreateInfo descriptorLayout =
                vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

        // 5. 分配描述符集
        // 从描述符池中分配一个描述符集
        VkDescriptorSetAllocateInfo allocInfo =
                vks::initializers::descriptorSetAllocateInfo(
                        descriptorPool,    // 使用的描述符池
                        &descriptorSetLayout, // 描述符集布局
                        1);                // 分配数量
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        // 6. 更新描述符集
        // 将统一缓冲区与描述符绑定
        VkWriteDescriptorSet writeDescriptorSet =
                vks::initializers::writeDescriptorSet(
                        descriptorSet,                  // 目标描述符集
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // 描述符类型
                        0,                             // 绑定点
                        &uniformBuffer.descriptor);    // 缓冲区描述信息
        // 执行描述符更新
        vkUpdateDescriptorSets(
                vulkanDevice->logicalDevice, // 逻辑设备
                1,                          // 描述符写入数量
                &writeDescriptorSet,        // 描述符写入信息
                0,                          // 描述符复制数量
                nullptr);                   // 描述符复制信息
    }

    // 准备图形管线
    void preparePipelines() {
        // 1. 创建管线布局
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
                vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1); // 使用描述符集布局
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

        // 2. 配置管线状态
        // 输入装配状态：指定使用三角形列表作为图元类型
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
                vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

        // 光栅化状态：填充模式，背面剔除，顺时针为正面
        VkPipelineRasterizationStateCreateInfo rasterizationState =
                vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);

        // 颜色混合附件状态：禁用混合
        VkPipelineColorBlendAttachmentState blendAttachmentState =
                vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

        // 颜色混合状态：使用上述混合附件状态
        VkPipelineColorBlendStateCreateInfo colorBlendState =
                vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        // 深度模板状态：启用深度测试和写入，使用小于等于比较
        VkPipelineDepthStencilStateCreateInfo depthStencilState =
                vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

        // 视口状态：1个视口和1个裁剪矩形（实际值动态设置）
        VkPipelineViewportStateCreateInfo viewportState =
                vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

        // 多重采样状态：禁用多重采样
        VkPipelineMultisampleStateCreateInfo multisampleState =
                vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);

        // 动态状态：视口和裁剪矩形将在命令缓冲区中动态设置
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState =
                vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        // 3. 加载着色器
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
        shaderStages[0] = loadShader(getShadersPath() + "gears/gears.vert.spv", VK_SHADER_STAGE_VERTEX_BIT); // 顶点着色器
        shaderStages[1] = loadShader(getShadersPath() + "gears/gears.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT); // 片段着色器

        // 4. 配置顶点输入
        // 顶点输入绑定：指定顶点数据的步幅和输入率
        VkVertexInputBindingDescription vertexInputBinding = {
                vks::initializers::vertexInputBindingDescription(0, sizeof(Gear::Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
        };

        // 顶点输入属性：描述顶点数据的各个属性
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                // 位置0：顶点位置（vec3）
                vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Gear::Vertex, position)),
                // 位置1：顶点法线（vec3）
                vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Gear::Vertex, normal)),
                // 位置2：顶点颜色（vec3）
                vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Gear::Vertex, color)),
        };

        // 创建顶点输入状态
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();

        // 5. 创建图形管线
        VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass, 0);
        // 设置所有状态
        pipelineCreateInfo.pVertexInputState = &vertexInputStateCI;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        // 设置着色器阶段
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();

        // 创建管线
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
    }

    // 构建命令缓冲区
    void buildCommandBuffers() {
        // 命令缓冲区开始信息
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        // 设置清除值：颜色缓冲区和深度缓冲区
        VkClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;       // 颜色缓冲区清除为默认颜色
        clearValues[1].depthStencil = { 1.0f, 0 };      // 深度缓冲区清除为1.0(最远)，模板缓冲区清除为0

        // 渲染通道开始信息配置
        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;     // 使用的渲染通道
        renderPassBeginInfo.renderArea.offset.x = 0;    // 渲染区域起点x坐标
        renderPassBeginInfo.renderArea.offset.y = 0;    // 渲染区域起点y坐标
        renderPassBeginInfo.renderArea.extent.width = width;   // 渲染区域宽度
        renderPassBeginInfo.renderArea.extent.height = height; // 渲染区域高度
        renderPassBeginInfo.clearValueCount = 2;        // 清除值数量
        renderPassBeginInfo.pClearValues = clearValues; // 清除值数组

        // 为每个交换链图像创建命令缓冲区
        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
            // 设置当前帧缓冲区
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            // 开始记录命令缓冲区
            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            // 开始渲染通道
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            // 设置动态视口
            VkViewport viewport = vks::initializers::viewport(
                    (float)width,    // 视口宽度
                    (float)height,   // 视口高度
                    0.0f,           // 最小深度
                    1.0f);          // 最大深度
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            // 设置动态裁剪矩形
            VkRect2D scissor = vks::initializers::rect2D(
                    width,          // 裁剪矩形宽度
                    height,         // 裁剪矩形高度
                    0,              // x偏移
                    0);             // y偏移
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            // 绑定图形管线
            vkCmdBindPipeline(
                    drawCmdBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline);

            // 绑定顶点缓冲区
            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(
                    drawCmdBuffers[i],
                    0,                  // 第一个绑定
                    1,                  // 绑定数量
                    &vertexBuffer.buffer, // 顶点缓冲区
                    offsets);           // 偏移量数组

            // 绑定索引缓冲区
            vkCmdBindIndexBuffer(
                    drawCmdBuffers[i],
                    indexBuffer.buffer,  // 索引缓冲区
                    0,                  // 偏移量
                    VK_INDEX_TYPE_UINT32); // 索引类型

            // 绑定描述符集(包含统一缓冲区等资源)
            vkCmdBindDescriptorSets(
                    drawCmdBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS, // 绑定到图形管线
                    pipelineLayout,     // 管线布局
                    0,                  // 第一个描述符集
                    1,                  // 描述符集数量
                    &descriptorSet,     // 描述符集
                    0,                  // 动态偏移量数量
                    nullptr);           // 动态偏移量数组

            // 绘制所有齿轮(使用实例化绘制)
            for (auto j = 0; j < numGears; j++) {
                vkCmdDrawIndexed(
                        drawCmdBuffers[i],
                        gears[j].indexCount, // 该齿轮的索引数量
                        1,                  // 实例数量(每个齿轮绘制一次)
                        gears[j].indexStart, // 索引起始位置
                        0,                  // 顶点偏移
                        j);                 // 实例索引(用于访问统一缓冲区中的模型矩阵)
            }

            // 绘制UI(ImGui等)
            drawUI(drawCmdBuffers[i]);

            // 结束渲染通道
            vkCmdEndRenderPass(drawCmdBuffers[i]);

            // 结束命令缓冲区记录
            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    // 准备统一缓冲区
    void prepareUniformBuffers() {
        // 创建顶点着色器统一缓冲区
        VK_CHECK_RESULT(vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &uniformBuffer,
                sizeof(UniformData)));

        // 映射缓冲区
        VK_CHECK_RESULT(uniformBuffer.map());
    }

    // 更新统一缓冲区
    void updateUniformBuffers() {
        // 计算基于时间的旋转角度（每秒360度）
        // timer通常是从程序开始累计的时间，单位秒
        float degree = timer * 360.0f;

        // 更新全局矩阵数据
        uniformData.projection = camera.matrices.perspective;  // 设置投影矩阵（从相机获取）
        uniformData.view = camera.matrices.view;               // 设置视图矩阵（从相机获取）
        uniformData.lightPos = glm::vec4(0.0f, 0.0f, 2.5f, 1.0f); // 设置光源位置（x,y,z,w）

        // 更新每个齿轮的模型矩阵
        for (auto i = 0; i < numGears; i++) {
            Gear gear = gears[i];  // 获取当前齿轮

            // 初始化模型矩阵为单位矩阵
            uniformData.model[i] = glm::mat4(1.0f);

            // 平移变换：将齿轮移动到指定位置
            uniformData.model[i] = glm::translate(uniformData.model[i], gear.pos);

            // 旋转变换：
            // 1. gear.rotSpeed * degree - 基于时间的旋转（不同齿轮可能有不同转速）
            // 2. gear.rotOffset - 初始旋转偏移量（用于齿轮啮合）
            // 3. 绕Z轴旋转（使用右手坐标系，正旋转方向为逆时针）
            uniformData.model[i] = glm::rotate(
                    uniformData.model[i],
                    glm::radians((gear.rotSpeed * degree) + gear.rotOffset),
                    glm::vec3(0.0f, 0.0f, 1.0f));  // 旋转轴为Z轴
        }

        // 将更新后的统一数据复制到映射的GPU缓冲区
        // uniformBuffer.mapped 是通过vkMapMemory映射的CPU可访问的GPU内存指针
        // sizeof(UniformData) 确保复制整个统一缓冲区数据结构
        memcpy(uniformBuffer.mapped, &uniformData, sizeof(UniformData));

        // 注意：这里不需要手动刷新内存，因为创建缓冲区时使用了VK_MEMORY_PROPERTY_HOST_COHERENT_BIT标志
        // 该标志确保CPU和GPU内存自动保持同步
    }

    // 准备函数
    void prepare() {
        VulkanExampleBase::prepare();
        prepareGears();          // 准备齿轮数据
        prepareUniformBuffers(); // 准备统一缓冲区
        setupDescriptors();      // 设置描述符
        preparePipelines();      // 准备图形管线
        buildCommandBuffers();   // 构建命令缓冲区
        prepared = true;        // 标记准备完成
    }

// 绘制函数 - 每帧调用的主渲染函数
    void draw() {
        // 1. 准备当前帧
        // 等待上一帧完成，获取下一个交换链图像索引
        VulkanExampleBase::prepareFrame();

        // 2. 配置提交信息
        submitInfo.commandBufferCount = 1;  // 提交1个命令缓冲区
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];  // 使用当前缓冲区的命令缓冲区

        // 3. 提交命令缓冲区到图形队列
        // queue: 图形队列
        // 1: 提交信息数量
        // &submitInfo: 提交信息指针
        // VK_NULL_HANDLE: 不使用栅栏
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        // 4. 提交当前帧到交换链
        // 将渲染完成的图像提交给交换链进行呈现
        VulkanExampleBase::submitFrame();
    }

// 渲染函数 - 每帧调用的主渲染循环
    virtual void render() {
        // 检查是否已完成初始化准备
        if (!prepared)  // prepared标志表示所有Vulkan资源是否已初始化完成
            return;     // 如果未准备好，则跳过渲染

        // 1. 更新统一缓冲区数据
        // 将CPU端的矩阵数据(如模型、视图、投影矩阵)更新到GPU端的统一缓冲区
        updateUniformBuffers();

        // 2. 执行绘制命令
        // 提交命令缓冲区到GPU队列，执行实际渲染操作
        draw();
    }
};

// 主入口
VULKAN_EXAMPLE_MAIN()