#include "vulkanexamplebase.h"

#if defined(__ANDROID__)
// 出于性能原因，在安卓设备上减少粒子数量
#define PARTICLES_PER_ATTRACTOR (3 * 1024) // 定义每个吸引子的粒子数量为3072
#else
#define PARTICLES_PER_ATTRACTOR 4 * 1024
#endif

class VulkanExample : public VulkanExampleBase {
public:
    struct Textures {
        vks::Texture2D particle;
        vks::Texture2D gradient;
    } textures{};

    // Particle Definition
    struct Particle {
        glm::vec4 pos;                                // xyz = position, w = mass
        glm::vec4 vel;                                // xyz = velocity, w = gradient texture position
    };
    uint32_t numParticles{0};

    // 使用着色器存储缓冲区对象（SSBO）存储粒子数据
    // 该缓冲区由计算管线更新，并在图形管线中作为顶点缓冲区用于粒子渲染
    vks::Buffer storageBuffer;

    // 图形渲染部分的资源
    struct Graphics {
        uint32_t queueFamilyIndex{};                    // 图形队列家族索引（用于检查是否需要跨队列同步）
        VkDescriptorSetLayout descriptorSetLayout{};    // 粒子系统渲染描述符集布局（定义着色器资源绑定规则）
        VkDescriptorSet descriptorSet{};                // 粒子系统渲染描述符集（绑定纹理/缓冲资源）
        VkPipelineLayout pipelineLayout{};            // 图形管线布局（关联描述符集和着色器阶段）
        VkPipeline pipeline{};                        // 粒子渲染图形管线（包含完整渲染状态配置）
        VkSemaphore semaphore{};                      // 图形-计算同步信号量（协调跨队列执行顺序）

        // 统一缓冲区数据结构
        struct UniformData {
            glm::mat4 projection;    // 投影矩阵（3D->2D变换）
            glm::mat4 view;            // 视图矩阵（相机坐标系变换）
            glm::vec2 screenDim;    // 屏幕尺寸（用于视口适配）
        } uniformData;

        vks::Buffer uniformBuffer;                    // 统一缓冲区对象（存储投影/视图矩阵等场景数据）
    } graphics;

    // 计算管线相关资源
    struct Compute {
        uint32_t queueFamilyIndex{};        // 计算队列家族索引（用于判断是否与图形队列家族不同，需额外同步）
        VkQueue queue{};                    // 计算命令队列（可能与图形队列属于不同家族）
        VkCommandPool commandPool{};        // 计算命令池（独立于图形队列的命令池）
        VkCommandBuffer commandBuffer{};    // 计算命令缓冲（存储计算派发命令和内存屏障）
        VkSemaphore semaphore{};          // 计算-图形同步信号量（协调跨队列执行顺序）

        // 描述符系统
        VkDescriptorSetLayout descriptorSetLayout{};    // 计算着色器描述符集布局（定义资源绑定规则）
        VkDescriptorSet descriptorSet{};                // 计算着色器描述符集（绑定SSBO/UBO资源）

        // 管线配置
        VkPipelineLayout pipelineLayout{};    // 计算管线布局（关联描述符集布局）
        VkPipeline pipelineCalculate{};        // N体速度计算管线（第1阶段：计算粒子间作用力）
        VkPipeline pipelineIntegrate{};        // 运动积分管线（第2阶段：更新粒子位置）

        // 计算着色器统一数据
        struct UniformData {
            float deltaT{0.0f};            // 帧时间差（控制模拟速度，暂停时为0）
            __attribute__((unused)) int32_t particleCount{0};    // 粒子总数（由CPU传递至GPU）
            float gravity{0.002f};        // 引力系数（控制粒子间吸引力强度）
            float power{0.75f};            // 作用力系数（影响粒子运动强度）
            __attribute__((unused)) float soften{0.05f};            // 软化参数（防止距离过近时力计算溢出）
        } uniformData;

        vks::Buffer uniformBuffer;        // 统一缓冲区（存储粒子系统控制参数）
    } compute;

    // 构造函数：初始化示例参数
    VulkanExample() : VulkanExampleBase() {
        // 设置窗口标题（显示计算着色器实现的N体系统）
        title = "Compute shader N-body system";

        // 配置相机参数
        // -------------------------------------------------------------------------
        // 设置相机类型为lookat（目标观察模式）
        camera.type = Camera::CameraType::lookat;

        // 配置透视投影参数：
        // - 垂直视野角：60度
        // - 宽高比：窗口宽高比
        // - 近裁剪面：0.1单位
        // - 远裁剪面：512.0单位
        camera.setPerspective(60.0f, (float) width / (float) height, 0.1f, 512.0f);

        // 设置初始相机旋转（欧拉角）
        // - X轴旋转-26度（俯角）
        // - Y轴旋转75度（水平旋转）
        // - Z轴旋转0度
        camera.setRotation(glm::vec3(-26.0f, 75.0f, 0.0f));

        // 设置初始相机位置（世界坐标系）
        // - 初始位于原点后方14单位位置（z = -14）
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -14.0f));

        // 设置相机移动速度（单位：单位/秒）
        camera.movementSpeed = 2.5f; // 用于键盘控制的移动速度
    }

    // 析构函数：释放所有Vulkan资源
    ~VulkanExample() override {
        if (device) {
            // [1] 销毁图形管线相关资源
            // ---------------------------------------------------------------
            // 销毁图形Uniform缓冲
            graphics.uniformBuffer.destroy();
            // 销毁图形渲染管线
            vkDestroyPipeline(device, graphics.pipeline, nullptr);
            // 销毁管线布局
            vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
            // 销毁描述符集布局
            vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
            // 销毁图形信号量
            vkDestroySemaphore(device, graphics.semaphore, nullptr);

            // [2] 销毁计算管线相关资源
            // ---------------------------------------------------------------
            // 销毁计算Uniform缓冲
            compute.uniformBuffer.destroy();
            // 销毁计算管线布局
            vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
            // 销毁计算描述符集布局
            vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
            // 销毁计算管线（第一阶段：粒子作用力计算）
            vkDestroyPipeline(device, compute.pipelineCalculate, nullptr);
            // 销毁计算管线（第二阶段：运动积分）
            vkDestroyPipeline(device, compute.pipelineIntegrate, nullptr);
            // 销毁计算信号量
            vkDestroySemaphore(device, compute.semaphore, nullptr);
            // 销毁计算命令池（自动释放关联的命令缓冲）
            vkDestroyCommandPool(device, compute.commandPool, nullptr);

            // [3] 销毁公共资源
            // ---------------------------------------------------------------
            // 销毁粒子存储缓冲区（SSBO）
            storageBuffer.destroy();
            // 销毁粒子纹理资源
            textures.particle.destroy();
            // 销毁颜色渐变纹理资源
            textures.gradient.destroy();
        }
    }

    // 加载纹理资源
    void loadAssets() {
        // 加载粒子纹理（RGBA格式，用于渲染单个粒子的外观）
        // 文件路径：基础资源路径 + "textures/particle01_rgba.ktx"
        // 格式：VK_FORMAT_R8G8B8A8_UNORM（8位无符号归一化RGBA，适合颜色纹理）
        textures.particle.loadFromFile(
                getAssetPath() + "textures/particle01_rgba.ktx", // 粒子纹理路径
                VK_FORMAT_R8G8B8A8_UNORM,                       // 标准化RGBA格式（0.0-1.0范围）
                vulkanDevice,                                   // 使用的Vulkan设备
                queue                                           // 用于纹理传输的队列
        );

        // 加载颜色渐变纹理（控制粒子颜色随速度/位置变化）
        // 文件路径：基础资源路径 + "textures/particle_gradient_rgba.ktx"
        // 格式同粒子纹理，用于片段着色器的颜色插值
        textures.gradient.loadFromFile(
                getAssetPath() + "textures/particle_gradient_rgba.ktx",
                VK_FORMAT_R8G8B8A8_UNORM,
                vulkanDevice,
                queue
        );
    }

    // 构建图形渲染命令缓冲区
    void buildCommandBuffers() override {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        // 定义清除值（颜色缓冲和深度/模板缓冲）
        VkClearValue clearValues[2];
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // 黑色背景
        clearValues[1].depthStencil = {1.0f, 0};           // 深度初始值1.0（远平面）

        // 初始化渲染通道信息
        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;        // 关联的渲染流程
        renderPassBeginInfo.renderArea.offset = {0, 0};     // 渲染区域起点
        renderPassBeginInfo.renderArea.extent = {width, height}; // 渲染区域尺寸
        renderPassBeginInfo.clearValueCount = 2;            // 清除值数量
        renderPassBeginInfo.pClearValues = clearValues;      // 清除值指针

        // 为每个交换链图像创建命令缓冲区
        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
            // 设置当前帧缓冲区
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            // 开始录制命令缓冲区
            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo))

            // [1] 获取屏障（当图形与计算队列家族不同时）
            // ---------------------------------------------------------
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
                VkBufferMemoryBarrier buffer_barrier = {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        0,                                      // srcAccessMask（计算队列已释放所有权）
                        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,    // dstAccessMask（顶点输入阶段需要读取）
                        compute.queueFamilyIndex,              // 原队列家族（计算队列）
                        graphics.queueFamilyIndex,             // 目标队列家族（图形队列）
                        storageBuffer.buffer,                 // 目标缓冲区（粒子数据）
                        0,
                        storageBuffer.size
                };

                // 插入管线屏障：
                // - 生产阶段：管线开始（确保屏障在所有操作前执行）
                // - 消费阶段：顶点输入阶段
                vkCmdPipelineBarrier(
                        drawCmdBuffers[i],
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    // 生产阶段标记
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,   // 消费阶段
                        0,
                        0, nullptr,
                        1, &buffer_barrier,
                        0, nullptr);
            }

            // [2] 开始渲染通道
            // ---------------------------------------------------------
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo,
                                 VK_SUBPASS_CONTENTS_INLINE);

            // 设置动态视口和裁剪区域
            VkViewport viewport = vks::initializers::viewport((float) width, (float) height, 0.0f,
                                                              1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(
                    static_cast<int32_t>(width),
                    static_cast<int32_t>(height),
                    0, 0
            );
            vkCmdSetScissor(
                    drawCmdBuffers[i],
                    0,
                    1,
                    &scissor
            );

            // [3] 绑定图形渲染管线
            vkCmdBindPipeline(
                    drawCmdBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphics.pipeline
            );

            // 绑定描述符集（包含纹理和Uniform缓冲）
            vkCmdBindDescriptorSets(
                    drawCmdBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    graphics.pipelineLayout,
                    0,
                    1,
                    &graphics.descriptorSet,
                    0,
                    nullptr
            );

            // [4] 绑定顶点缓冲区并绘制粒子
            // ---------------------------------------------------------
            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(
                    drawCmdBuffers[i],
                    0,
                    1,
                    &storageBuffer.buffer,
                    offsets
            ); // 使用SSBO作为顶点缓冲
            vkCmdDraw(
                    drawCmdBuffers[i],
                    numParticles,
                    1,
                    0,
                    0
            ); // 绘制所有粒子（实例数=1）

            // [5] 绘制UI叠加层（调试信息/参数控制）
            drawUI(drawCmdBuffers[i]);

            // 结束渲染通道
            vkCmdEndRenderPass(drawCmdBuffers[i]);

            // [6] 释放屏障（将缓冲区所有权返还给计算队列）
            // ---------------------------------------------------------
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
                VkBufferMemoryBarrier buffer_barrier = {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,    // srcAccessMask（顶点阶段已读取）
                        0,                                      // dstAccessMask（无后续访问）
                        graphics.queueFamilyIndex,             // 原队列家族（图形队列）
                        compute.queueFamilyIndex,              // 目标队列家族（计算队列）
                        storageBuffer.buffer,
                        0,
                        storageBuffer.size
                };

                // 插入管线屏障：
                // - 生产阶段：顶点输入阶段
                // - 消费阶段：管线结束（计算队列后续操作）
                vkCmdPipelineBarrier(
                        drawCmdBuffers[i],
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,    // 生产阶段
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,  // 消费阶段标记
                        0,
                        0,
                        nullptr,
                        1,
                        &buffer_barrier,
                        0,
                        nullptr
                );
            }

            // 结束命令缓冲区录制
            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]))
        }
    }

    // 构建计算命令缓冲区（录制计算着色器调度命令）
    void buildComputeCommandBuffer() {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        // 开始录制计算命令
        VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo))

        // [1] 获取屏障（当图形与计算队列家族不同时需所有权转移）
        // -------------------------------------------------------------------------
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
            VkBufferMemoryBarrier buffer_barrier = {
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    0,                                      // 源访问掩码（无）
                    VK_ACCESS_SHADER_WRITE_BIT,             // 目标访问掩码（计算着色器写入）
                    graphics.queueFamilyIndex,              // 原队列家族
                    compute.queueFamilyIndex,               // 目标队列家族
                    storageBuffer.buffer,                   // 目标缓冲区
                    0,
                    storageBuffer.size
            };

            // 管线屏障作用范围：
            // - 生产阶段：管线开始（确保屏障在所有操作前执行）
            // - 消费阶段：计算着色器阶段
            vkCmdPipelineBarrier(
                    compute.commandBuffer,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,     // 生产者阶段标记
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 消费者阶段
                    0,
                    0,
                    nullptr,
                    1,
                    &buffer_barrier,
                    0,
                    nullptr
            );
        }

        // [2] 第一阶段：计算粒子间作用力
        // -------------------------------------------------------------------------
        // 绑定计算管线（粒子间引力计算）
        vkCmdBindPipeline(
                compute.commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                compute.pipelineCalculate
        );

        // 绑定描述符集（SSBO和UBO）
        vkCmdBindDescriptorSets(
                compute.commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                compute.pipelineLayout,
                0,
                1,
                &compute.descriptorSet,
                0,
                nullptr
        );
        // 调度计算工作组（每个工作组256个粒子）
        vkCmdDispatch(compute.commandBuffer,
                      numParticles / 256,  // 工作组数量 = 总粒子数 / 每工作组大小(256)
                      1,
                      1
        );

        // [3] 计算阶段间内存屏障
        // -------------------------------------------------------------------------
        VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
        bufferBarrier.buffer = storageBuffer.buffer;
        bufferBarrier.size = storageBuffer.descriptor.range;
        bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // 前一阶段写入
        bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;   // 下一阶段读取
        // 统一队列家族时不需要所有权转移
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        // 屏障作用范围：
        // - 生产阶段：计算着色器阶段
        // - 消费阶段：计算着色器阶段（同一管线的不同阶段）
        vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 生产者阶段
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 消费者阶段
                VK_FLAGS_NONE,
                0,
                nullptr,
                1,
                &bufferBarrier,
                0,
                nullptr
        );

        // [4] 第二阶段：积分运动方程
        // -------------------------------------------------------------------------
        // 绑定计算管线（欧拉积分更新位置）
        vkCmdBindPipeline(
                compute.commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                compute.pipelineIntegrate
        );
        // 调度相同数量的工作组
        vkCmdDispatch(
                compute.commandBuffer,
                numParticles / 256,
                1,
                1
        );

        // [5] 释放屏障（所有权转移回图形队列家族）
        // -------------------------------------------------------------------------
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
            VkBufferMemoryBarrier buffer_barrier = {
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    VK_ACCESS_SHADER_WRITE_BIT,            // 源访问掩码（计算写入）
                    0,                                      // 目标访问掩码（无）
                    compute.queueFamilyIndex,               // 原队列家族
                    graphics.queueFamilyIndex,              // 目标队列家族
                    storageBuffer.buffer,
                    0,
                    storageBuffer.size
            };

            // 屏障作用范围：
            // - 生产阶段：计算着色器阶段
            // - 消费阶段：管线结束（图形队列后续操作）
            vkCmdPipelineBarrier(
                    compute.commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 生产者阶段
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,  // 消费者阶段标记
                    0,
                    0,
                    nullptr,
                    1,
                    &buffer_barrier,
                    0,
                    nullptr
            );
        }

        // 结束命令缓冲录制
        vkEndCommandBuffer(compute.commandBuffer);
    }

    // 准备粒子存储缓冲区（SSBO），包含粒子的初始位置和速度数据
    void prepareStorageBuffers() {
        // [1] 定义吸引子位置（这些点会吸引周围粒子形成动态效果）
        // -------------------------------------------------------------------------
        // 每个吸引子周围生成 PARTICLES_PER_ATTRACTOR 个粒子
        std::vector<glm::vec3> attractors = {
                glm::vec3(5.0f, 0.0f, 0.0f),    // X轴正方向吸引子
                glm::vec3(-5.0f, 0.0f, 0.0f),   // X轴负方向吸引子
                glm::vec3(0.0f, 0.0f, 5.0f),    // Z轴正方向吸引子
                glm::vec3(0.0f, 0.0f, -5.0f),   // Z轴负方向吸引子
                glm::vec3(0.0f, 4.0f, 0.0f),    // Y轴正方向（上方）吸引子
                glm::vec3(0.0f, -8.0f, 0.0f),   // Y轴负方向（下方）吸引子
        };

        // 计算总粒子数 = 吸引子数量 × 每个吸引子的粒子数
        numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;

        // [2] 初始化粒子数据
        // -------------------------------------------------------------------------
        std::vector<Particle> particleBuffer(numParticles);

        // 初始化随机数生成器：基准测试模式使用固定种子，否则使用时间种子
        std::default_random_engine rndEngine(benchmark.active ? 0 : (unsigned) time(nullptr));
        std::normal_distribution<float> rndDist(0.0f, 1.0f); // 正态分布生成器

        // 为每个吸引子生成粒子群
        for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++) {
            for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++) {
                Particle &particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];

                // 每组第一个粒子作为高质量引力中心
                if (j == 0) {
                    particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f); // 位置偏移1.5倍，质量设为90000
                    particle.vel = glm::vec4(0.0f);                          // 初始速度为0
                } else { // 其他粒子随机分布
                    // 位置生成：在吸引子周围正态分布
                    glm::vec3 position = attractors[i] +
                                         glm::vec3(rndDist(rndEngine), rndDist(rndEngine),
                                                   rndDist(rndEngine)) * 0.75f;
                    // 调整Y轴位置避免过度扩散
                    float len = glm::length(glm::normalize(position - attractors[i]));
                    position.y *= 2.0f - (len * len);

                    // 速度生成：角速度交叉产生旋转效果 + 随机扰动
                    glm::vec3 angular =
                            glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f); // 交替旋转方向
                    glm::vec3 velocity = glm::cross((position - attractors[i]), angular) +
                                         glm::vec3(rndDist(rndEngine), rndDist(rndEngine),
                                                   rndDist(rndEngine) * 0.025f);

                    // 质量生成：正态分布生成75 ± 37.5
                    float mass = (rndDist(rndEngine) * 0.5f + 0.5f) * 75.0f;
                    particle.pos = glm::vec4(position, mass);
                    particle.vel = glm::vec4(velocity, 0.0f); // velocity.w 用于颜色渐变
                }

                // 颜色渐变偏移：根据吸引子索引设置（用于片段着色器颜色插值）
                particle.vel.w = (float) i * 1.0f / static_cast<float>(attractors.size());
            }
        }

        // [3] 创建GPU存储缓冲区
        // -------------------------------------------------------------------------
        compute.uniformData.particleCount = static_cast<int32_t>(numParticles); // 传递粒子数量到计算着色器
        VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

        // 创建暂存缓冲区（CPU可见，用于初始数据上传）
        vks::Buffer stagingBuffer;
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,                       // 用途：传输源
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &stagingBuffer,
                storageBufferSize,
                particleBuffer.data()                                   // 直接填充初始数据
        );

        // 创建设备本地存储缓冲区（最终存储位置）
        // 用途：计算管线存储缓冲 + 图形管线顶点缓冲 + 传输目标
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // 设备本地内存（快速访问）
                &storageBuffer,
                storageBufferSize
        );

        // [4] 传输数据到设备内存
        // -------------------------------------------------------------------------
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                true
        );
        VkBufferCopy copyRegion = {};
        copyRegion.size = storageBufferSize;
        vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffer.buffer, 1, &copyRegion);

        // 如果图形和计算队列家族不同，添加传输屏障
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
            VkBufferMemoryBarrier buffer_barrier = {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, // 图形队列的访问标志
                    .dstAccessMask = 0,                                   // 不需要后续访问
                    .srcQueueFamilyIndex = graphics.queueFamilyIndex,
                    .dstQueueFamilyIndex = compute.queueFamilyIndex,
                    .buffer = storageBuffer.buffer,
                    .offset = 0,
                    .size = storageBuffer.size
            };

            vkCmdPipelineBarrier(
                    copyCmd,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,   // 生产者阶段：顶点输入
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, // 消费者阶段：无实际阶段
                    0,
                    0, nullptr,
                    1, &buffer_barrier,
                    0, nullptr
            );
        }

        // 提交传输命令并等待完成
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

        // [5] 清理暂存缓冲区
        // -------------------------------------------------------------------------
        stagingBuffer.destroy(); // 数据已传输，临时缓冲可释放
    }

    // 准备图形渲染相关资源（Uniform缓冲、描述符、管线等）
    void prepareGraphics() {
        // [1] 创建顶点着色器Uniform缓冲
        // ------------------------------------------------------------------------------------
        // 用途：存储顶点着色器需要的全局参数（投影/视图矩阵、屏幕尺寸等）
        // 内存特性：主机可见（可映射） + 主机一致性（自动同步）
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,          // 缓冲类型：Uniform缓冲
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |        // 允许CPU端访问
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,        // CPU/GPU内存自动同步
                &graphics.uniformBuffer,                     // 目标缓冲对象
                sizeof(Graphics::UniformData)                // 数据大小匹配结构体
        );
        VK_CHECK_RESULT(graphics.uniformBuffer.map())    // 映射内存供CPU写入

        // [2] 创建描述符池（描述符资源池）
        // ------------------------------------------------------------------------------------
        // 描述符类型及数量配置：
        std::vector<VkDescriptorPoolSize> poolSizes = {
                vks::initializers::descriptorPoolSize(
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),   // 2个Uniform缓冲（图形+计算各1）
                vks::initializers::descriptorPoolSize(
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),   // 1个存储缓冲（粒子数据）
                vks::initializers::descriptorPoolSize(
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2) // 2个纹理采样器（粒子+渐变）
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo =
                vks::initializers::descriptorPoolCreateInfo(
                        poolSizes,  // 池容量配置
                        2           // 最大分配的描述符集数量（图形+计算各1）
                );
        VK_CHECK_RESULT(
                vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool))

        // [3] 创建描述符集布局（资源绑定规则）
        // ------------------------------------------------------------------------------------
        // 描述符绑定配置（对应着色器中的binding点）：
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // 绑定0：粒子纹理（片段着色器使用）
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        VK_SHADER_STAGE_FRAGMENT_BIT,  // 仅片段着色器访问
                        0                              // binding=0
                ),
                // 绑定1：颜色渐变纹理（片段着色器使用）
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        VK_SHADER_STAGE_FRAGMENT_BIT,
                        1                              // binding=1
                ),
                // 绑定2：Uniform缓冲（顶点着色器使用）
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_VERTEX_BIT,    // 仅顶点着色器访问
                        2                              // binding=2
                )
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout =
                vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(
                vkCreateDescriptorSetLayout(
                        device,
                        &descriptorLayout,
                        nullptr,
                        &graphics.descriptorSetLayout
                )
        )

        // [4] 分配并更新描述符集（绑定具体资源）
        // ------------------------------------------------------------------------------------
        VkDescriptorSetAllocateInfo allocInfo =
                vks::initializers::descriptorSetAllocateInfo(
                        descriptorPool,                                 // 从哪个池分配
                        &graphics.descriptorSetLayout,       // 使用的布局
                        1                              // 分配1个描述符集
                );
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSet))

        // 将实际资源绑定到描述符集
        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                // 绑定0：粒子纹理描述符
                vks::initializers::writeDescriptorSet(
                        graphics.descriptorSet,
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        0,                              // binding=0
                        &textures.particle.descriptor   // 纹理+采样器描述符
                ),
                // 绑定1：渐变纹理描述符
                vks::initializers::writeDescriptorSet(
                        graphics.descriptorSet,
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        1,                              // binding=1
                        &textures.gradient.descriptor
                ),
                // 绑定2：Uniform缓冲描述符
                vks::initializers::writeDescriptorSet(
                        graphics.descriptorSet,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        2,                              // binding=2
                        &graphics.uniformBuffer.descriptor // 缓冲描述符
                )
        };
        vkUpdateDescriptorSets(
                device,
                static_cast<uint32_t>(writeDescriptorSets.size()),
                writeDescriptorSets.data(),
                0,
                nullptr
        );

        // [5] 创建管线布局（关联描述符布局）
        // ------------------------------------------------------------------------------------
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
                vks::initializers::pipelineLayoutCreateInfo(
                        &graphics.descriptorSetLayout, // 使用的描述符集布局
                        1                              // 描述符集布局数量
                );
        VK_CHECK_RESULT(
                vkCreatePipelineLayout(
                        device,
                        &pipelineLayoutCreateInfo,
                        nullptr,
                        &graphics.pipelineLayout
                )
        )

        // [6] 配置图形管线状态
        // ------------------------------------------------------------------------------------
        // 输入组装状态：点列表（每个顶点为一个粒子）
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
                vks::initializers::pipelineInputAssemblyStateCreateInfo(
                        VK_PRIMITIVE_TOPOLOGY_POINT_LIST, // 绘制点精灵
                        0,                // 无原始重启
                        VK_FALSE
                );

        // 光栅化状态：填充模式，禁用背面剔除
        VkPipelineRasterizationStateCreateInfo rasterizationState =
                vks::initializers::pipelineRasterizationStateCreateInfo(
                        VK_POLYGON_MODE_FILL,     // 填充模式
                        VK_CULL_MODE_NONE,        // 不剔除任何面
                        VK_FRONT_FACE_COUNTER_CLOCKWISE, // 逆时针为正面
                        0                         // 无深度偏移
                );

        // 颜色混合附件状态：初始配置（后续会修改为加色混合）
        VkPipelineColorBlendAttachmentState blendAttachmentState =
                vks::initializers::pipelineColorBlendAttachmentState(
                        0xf,            // 启用所有颜色通道（RGBA）
                        VK_FALSE        // 初始禁用混合
                );

        VkPipelineColorBlendStateCreateInfo colorBlendState =
                vks::initializers::pipelineColorBlendStateCreateInfo(
                        1,                // 附件数量
                        &blendAttachmentState
                );

        // 深度模板状态：禁用深度测试和写入
        VkPipelineDepthStencilStateCreateInfo depthStencilState =
                vks::initializers::pipelineDepthStencilStateCreateInfo(
                        VK_FALSE,       // 禁用深度测试
                        VK_FALSE,       // 禁用深度写入
                        VK_COMPARE_OP_ALWAYS
                );

        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(
                1,
                1,
                0
        );
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(
                VK_SAMPLE_COUNT_1_BIT,
                0
        );

        // 视口状态：使用动态视口和裁剪
        std::vector<VkDynamicState> dynamicStateEnables = {
                VK_DYNAMIC_STATE_VIEWPORT,  // 动态设置视口
                VK_DYNAMIC_STATE_SCISSOR    // 动态设置裁剪区域
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
                vks::initializers::pipelineDynamicStateCreateInfo(
                        dynamicStateEnables
                );

        // [7] 配置顶点输入（从SSBO读取粒子数据）
        // ------------------------------------------------------------------------------------
        // 顶点绑定描述：单个绑定，步长为粒子结构体大小
        std::vector<VkVertexInputBindingDescription> inputBindings = {
                vks::initializers::vertexInputBindingDescription(
                        0,                          // 绑定索引
                        sizeof(Particle),           // 每个顶点的数据步长
                        VK_VERTEX_INPUT_RATE_VERTEX // 逐顶点数据
                )
        };

        // 顶点属性描述：
        std::vector<VkVertexInputAttributeDescription> inputAttributes = {
                // 属性0：位置+质量（vec4）
                vks::initializers::vertexInputAttributeDescription(
                        0,                          // 绑定索引
                        0,                          // Location 0
                        VK_FORMAT_R32G32B32A32_SFLOAT, // vec4格式
                        offsetof(Particle, pos)     // 在结构体中的偏移量
                ),
                // 属性1：速度+渐变位置（vec4）
                vks::initializers::vertexInputAttributeDescription(
                        0,                          // 绑定索引
                        1,                          // Location 1
                        VK_FORMAT_R32G32B32A32_SFLOAT,
                        offsetof(Particle, vel)
                )
        };

        VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
        vertexInputState.pVertexBindingDescriptions = inputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

        // [8] 加载着色器模块
        // ------------------------------------------------------------------------------------
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

        // 顶点着色器：处理粒子位置变换
        shaderStages[0] = loadShader(
                getShadersPath() + "computenbody/particle.vert.spv",
                VK_SHADER_STAGE_VERTEX_BIT
        );
        // 片段着色器：处理颜色混合与纹理采样
        shaderStages[1] = loadShader(
                getShadersPath() + "computenbody/particle.frag.spv",
                VK_SHADER_STAGE_FRAGMENT_BIT
        );

        // [9] 创建图形管线
        // ------------------------------------------------------------------------------------
        VkGraphicsPipelineCreateInfo pipelineCreateInfo =
                vks::initializers::pipelineCreateInfo(
                        graphics.pipelineLayout, // 管线布局
                        renderPass,              // 关联的渲染流程
                        0                        // 子流程索引
                );

        // 装配管线状态结构体
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.renderPass = renderPass;

        // 配置加色混合参数（实现粒子发光效果）
        blendAttachmentState.colorWriteMask = 0xF;
        blendAttachmentState.blendEnable = VK_TRUE;                     // 启用混合
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;            // RGB相加
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // 源颜色权重1.0
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; // 目标颜色权重1.0
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;           // Alpha相加
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

        VK_CHECK_RESULT(
                vkCreateGraphicsPipelines(
                        device,
                        pipelineCache,     // 管线缓存（加速后续管线创建）
                        1,
                        &pipelineCreateInfo,
                        nullptr,
                        &graphics.pipeline
                )
        )

        // [10] 创建同步信号量（用于计算与图形管线同步）
        // ------------------------------------------------------------------------------------
        VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(
                vkCreateSemaphore(
                        device,
                        &semaphoreCreateInfo,
                        nullptr,
                        &graphics.semaphore
                )
        )

        // 初始触发信号量（避免首次渲染等待未触发的信号量）
        VkSubmitInfo submitInfo = vks::initializers::submitInfo();
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &graphics.semaphore;
        VK_CHECK_RESULT(
                vkQueueSubmit(
                        queue,
                        1,
                        &submitInfo,
                        VK_NULL_HANDLE
                )
        )
        VK_CHECK_RESULT(vkQueueWaitIdle(queue)) // 等待初始化完成

        // [11] 构建命令缓冲区（录制绘制命令）
        // ------------------------------------------------------------------------------------
        buildCommandBuffers();
    }

    // 准备计算管线及相关资源
    void prepareCompute() {
        // [1] 获取计算队列
        // -------------------------------------------------------------------------
        // 注意：计算队列家族可能与图形队列不同，需在设备创建时查询确定
        // 这会影响后续的同步策略（见buildComputeCommandBuffer中的内存屏障）
        vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);

        // [2] 创建计算着色器Uniform缓冲
        // -------------------------------------------------------------------------
        // 用途：存储计算着色器参数（时间步长、粒子数、引力系数等）
        vulkanDevice->createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                &compute.uniformBuffer,
                sizeof(Compute::UniformData)
        );
        VK_CHECK_RESULT(compute.uniformBuffer.map()) // 映射供CPU更新

        // [3] 创建计算描述符集布局
        // -------------------------------------------------------------------------
        // 绑定规则：
        // 绑定0：粒子存储缓冲区（SSBO，读+写）
        // 绑定1：Uniform缓冲（只读参数）
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // 存储缓冲区类型
                        VK_SHADER_STAGE_COMPUTE_BIT,         // 计算着色器阶段访问
                        0                                    // binding=0
                ),
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        VK_SHADER_STAGE_COMPUTE_BIT,
                        1                                   // binding=1
                ),
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout =
                vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(
                vkCreateDescriptorSetLayout(
                        device,
                        &descriptorLayout,
                        nullptr,
                        &compute.descriptorSetLayout
                )
        )

        // [4] 分配并更新计算描述符集
        // -------------------------------------------------------------------------
        VkDescriptorSetAllocateInfo allocInfo =
                vks::initializers::descriptorSetAllocateInfo(
                        descriptorPool,
                        &compute.descriptorSetLayout,
                        1
                );
        VK_CHECK_RESULT(
                vkAllocateDescriptorSets(
                        device,
                        &allocInfo,
                        &compute.descriptorSet
                )
        )

        // 绑定实际资源到描述符
        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
                // 绑定0：粒子存储缓冲区（SSBO）
                vks::initializers::writeDescriptorSet(
                        compute.descriptorSet,
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        0,
                        &storageBuffer.descriptor
                ),
                // 绑定1：Uniform缓冲
                vks::initializers::writeDescriptorSet(
                        compute.descriptorSet,
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        1,
                        &compute.uniformBuffer.descriptor
                )
        };
        vkUpdateDescriptorSets(
                device,
                static_cast<uint32_t>(computeWriteDescriptorSets.size()),
                computeWriteDescriptorSets.data(),
                0,
                nullptr
        );

        // [5] 创建计算管线布局
        // -------------------------------------------------------------------------
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
                vks::initializers::pipelineLayoutCreateInfo(
                        &compute.descriptorSetLayout,
                        1
                );
        VK_CHECK_RESULT(
                vkCreatePipelineLayout(
                        device,
                        &pipelineLayoutCreateInfo,
                        nullptr,
                        &compute.pipelineLayout
                )
        )

        // [6] 创建计算管线（分两阶段）
        // -------------------------------------------------------------------------
        VkComputePipelineCreateInfo computePipelineCreateInfo =
                vks::initializers::computePipelineCreateInfo(
                        compute.pipelineLayout,
                        0
                );

        // 阶段1：粒子间作用力计算管线
        // -------------------------------------------------------------------------
        computePipelineCreateInfo.stage =
                loadShader(
                        getShadersPath() + "computenbody/particle_calculate.comp.spv",
                        VK_SHADER_STAGE_COMPUTE_BIT
                );

        // 配置共享内存特化常量
        // 根据设备限制计算每个工作组可用的共享内存大小
        uint32_t sharedDataSize = std::min(
                (uint32_t) 1024,
                (uint32_t) (vulkanDevice->properties.limits.maxComputeSharedMemorySize /
                            sizeof(glm::vec4))
        );
        VkSpecializationMapEntry specializationMapEntry =
                vks::initializers::specializationMapEntry(
                        0,
                        0,
                        sizeof(uint32_t)
                ); // 常量ID=0
        VkSpecializationInfo specializationInfo =
                vks::initializers::specializationInfo(
                        1,
                        &specializationMapEntry,
                        sizeof(uint32_t),
                        &sharedDataSize
                );
        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

        // 创建第一阶段计算管线
        VK_CHECK_RESULT(
                vkCreateComputePipelines(
                        device,
                        pipelineCache,
                        1,
                        &computePipelineCreateInfo,
                        nullptr,
                        &compute.pipelineCalculate
                )
        )

        // 阶段2：粒子运动积分管线
        // -------------------------------------------------------------------------
        computePipelineCreateInfo.stage =
                loadShader(
                        getShadersPath() + "computenbody/particle_integrate.comp.spv",
                        VK_SHADER_STAGE_COMPUTE_BIT
                );
        VK_CHECK_RESULT(
                vkCreateComputePipelines(
                        device,
                        pipelineCache,
                        1,
                        &computePipelineCreateInfo,
                        nullptr,
                        &compute.pipelineIntegrate
                )
        )

        // [7] 创建计算命令池
        // -------------------------------------------------------------------------
        // 使用计算队列家族创建独立命令池（可能与图形队列家族不同）
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex; // 计算队列家族索引
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // 允许重置命令缓冲
        VK_CHECK_RESULT(
                vkCreateCommandPool(
                        device,
                        &cmdPoolInfo,
                        nullptr,
                        &compute.commandPool
                )
        )

        // [8] 创建计算命令缓冲
        // -------------------------------------------------------------------------
        compute.commandBuffer = vulkanDevice->createCommandBuffer(  // 创建主命令缓冲
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                compute.commandPool
        );

        // [9] 创建计算同步信号量
        // -------------------------------------------------------------------------
        VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(
                vkCreateSemaphore(
                        device,
                        &semaphoreCreateInfo,
                        nullptr,
                        &compute.semaphore
                )
        )

        // [10] 构建计算命令缓冲（录制计算命令）
        // -------------------------------------------------------------------------
        buildComputeCommandBuffer();
    }

    // 更新计算着色器所需的Uniform缓冲数据
    void updateComputeUniformBuffers() {
        // [1] 计算时间步长deltaT
        // -----------------------------------------------------------------
        // - 若暂停(paused)，deltaT=0（停止物理模拟）
        // - 否则根据帧时间(frameTimer)计算步长，0.05为缩放系数，用于控制模拟速度
        compute.uniformData.deltaT = paused ? 0.0f : frameTimer * 0.05f;

        // [2] 更新GPU端Uniform缓冲
        // -----------------------------------------------------------------
        // 直接将CPU端数据拷贝到已映射的GPU内存区域
        // 注意：该缓冲需使用VK_MEMORY_PROPERTY_HOST_VISIBLE | VK_MEMORY_PROPERTY_HOST_COHERENT 内存属性
        // 确保CPU写入后自动同步到GPU，无需手动调用vkFlushMappedMemoryRanges
        memcpy(
                compute.uniformBuffer.mapped,   // 目标：GPU映射内存指针
                &compute.uniformData,           // 源：CPU端数据结构
                sizeof(Compute::UniformData)
        );  // 数据大小需严格匹配着色器端定义
    }

    // 更新图形渲染所需的Uniform缓冲区数据
    void updateGraphicsUniformBuffers() {
        // 更新投影矩阵：将相机的透视投影矩阵存入Uniform数据结构
        graphics.uniformData.projection = camera.matrices.perspective;

        // 更新视图矩阵：将相机的观察矩阵存入Uniform数据结构
        graphics.uniformData.view = camera.matrices.view;

        // 更新屏幕尺寸：将当前窗口的宽高转换为vec2传入着色器
        // 可用于屏幕空间坐标计算（如粒子大小适配屏幕分辨率）
        graphics.uniformData.screenDim = glm::vec2((float) width, (float) height);

        // 将更新后的Uniform数据从CPU内存拷贝至GPU映射内存
        // 注意：此处假设Uniform缓冲区已通过VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT特性创建并映射
        // Vulkan会自动保证内存可见性，无需手动刷新（因创建时指定了VK_MEMORY_PROPERTY_HOST_COHERENT_BIT）
        memcpy(
                graphics.uniformBuffer.mapped,         // 目标：GPU映射内存指针
                &graphics.uniformData,                    // 源：CPU端数据结构
                sizeof(Graphics::UniformData)
        );           // 数据大小：严格匹配着色器端的UBO结构
    }

    // 准备所有Vulkan资源
    void prepare() override {
        // [1] 调用基类准备方法（初始化交换链、命令池等基础资源）
        VulkanExampleBase::prepare();

        // [2] 获取队列家族索引
        // ---------------------------------------------------------------
        // 记录图形和计算队列家族索引，用于后续判断是否需要跨队列同步
        // 注意：如果图形和计算队列属于不同家族，需要特殊处理资源所有权转移
        graphics.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics; // 图形队列家族索引
        compute.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;   // 计算队列家族索引

        // [3] 资源加载与初始化流程
        // ---------------------------------------------------------------
        loadAssets();            // 加载纹理等外部资源（粒子纹理、渐变纹理等）
        prepareStorageBuffers(); // 创建并初始化粒子存储缓冲区（SSBO）
        prepareGraphics();       // 初始化图形管线相关资源（着色器、描述符、渲染管线等）
        prepareCompute();        // 初始化计算管线相关资源（计算着色器、命令缓冲等）

        // [4] 标记准备完成
        prepared = true; // 用于后续渲染循环的状态检查
    }

    // 执行帧渲染（协调计算与图形队列的提交顺序及同步）
    void draw() {
        // [1] 等待图形队列完成渲染（通过信号量同步）
        // ---------------------------------------------------------------
        // 计算队列将在COMPUTE_SHADER阶段等待图形信号量
        VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        // [2] 配置并提交计算命令
        // ---------------------------------------------------------------
        VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
        computeSubmitInfo.commandBufferCount = 1;                    // 计算命令缓冲区
        computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;  // 指向预录制的计算命令缓冲
        computeSubmitInfo.waitSemaphoreCount = 1;                     // 需要等待1个信号量
        computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;       // 等待图形队列的上次渲染完成
        computeSubmitInfo.pWaitDstStageMask = &waitStageMask;         // 在计算着色器阶段等待
        computeSubmitInfo.signalSemaphoreCount = 1;                   // 完成时触发1个信号量
        computeSubmitInfo.pSignalSemaphores = &compute.semaphore;     // 计算完成通知信号量
        VK_CHECK_RESULT(
                vkQueueSubmit(
                        compute.queue,
                        1,
                        &computeSubmitInfo,
                        VK_NULL_HANDLE
                )
        )

        // [3] 准备交换链资源
        // ---------------------------------------------------------------
        VulkanExampleBase::prepareFrame(); // 获取下一帧交换链图像，更新semaphores.presentComplete

        // [4] 配置图形渲染的同步依赖
        // ---------------------------------------------------------------
        // 等待阶段配置（两个条件需同时满足）：
        VkPipelineStageFlags graphicsWaitStageMasks[] = {
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,          // 顶点输入阶段等待计算完成（粒子数据就绪）
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT // 颜色附件阶段等待图像可用
        };
        // 等待的信号量：
        VkSemaphore graphicsWaitSemaphores[] = {
                compute.semaphore,         // 计算命令完成信号
                semaphores.presentComplete // 交换链图像准备就绪信号
        };
        // 触发信号量：
        VkSemaphore graphicsSignalSemaphores[] = {
                graphics.semaphore,       // 图形渲染完成信号（用于下一帧计算等待）
                semaphores.renderComplete // 渲染完成信号（用于后续呈现）
        };

        // [5] 配置并提交图形渲染命令
        // ---------------------------------------------------------------
        submitInfo.commandBufferCount = 1;                          // 当前帧的图形命令缓冲
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];// 指向当前交换链图像的命令缓冲
        submitInfo.waitSemaphoreCount = 2;                           // 需要等待两个信号量
        submitInfo.pWaitSemaphores = graphicsWaitSemaphores;          // 计算完成 + 图像就绪
        submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;       // 对应两个等待阶段
        submitInfo.signalSemaphoreCount = 2;                         // 触发两个信号量
        submitInfo.pSignalSemaphores = graphicsSignalSemaphores;     // 图形完成 + 渲染完成
        VK_CHECK_RESULT(
                vkQueueSubmit(
                        queue,
                        1,
                        &submitInfo,
                        VK_NULL_HANDLE
                )
        )

        // [6] 提交渲染结果到呈现引擎
        // ---------------------------------------------------------------
        VulkanExampleBase::submitFrame(); // 将渲染结果提交给交换链，处理垂直同步
    }

    // 每帧渲染逻辑（重写基类虚函数）
    void render() override {
        // [1] 检查资源是否准备就绪
        if (!prepared)
            return; // 如果资源未初始化，跳过渲染

        // [2] 更新计算管线的Uniform缓冲数据
        // 包含时间步长(deltaT)、粒子系统参数等
        updateComputeUniformBuffers();

        // [3] 更新图形管线的Uniform缓冲数据
        // 包含视图矩阵、投影矩阵、屏幕尺寸等
        updateGraphicsUniformBuffers();

        // [4] 提交渲染命令
        // 执行以下操作：
        // - 提交计算命令更新粒子位置
        // - 提交图形命令渲染粒子
        // - 处理跨队列同步
        // - 提交呈现请求
        draw();
    }
};

VULKAN_EXAMPLE_MAIN()