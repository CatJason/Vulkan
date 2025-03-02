#include <cstdio>                                           // 标准输入输出库，用于文件操作和调试输出
#include <cstdlib>                                          // 标准库，包含内存管理、程序控制等功能
#include <cstring>                                          // 字符串操作库，用于内存和字符串操作
#include <cassert>                                          // 断言库，用于调试时检查条件
#include <fstream>                                          // 文件流库，用于文件读写操作
#include <vector>                                           // 向量容器库，用于动态数组管理
#include <exception>                                        // 异常处理库，用于捕获和处理异常

#define GLM_FORCE_RADIANS                                   // 强制 GLM 使用弧度制
#define GLM_FORCE_DEPTH_ZERO_TO_ONE                         // 强制 GLM 使用深度范围 [0, 1]（Vulkan 默认）

#include <glm/glm.hpp>                                      // GLM 核心库，提供向量、矩阵等数学工具
#include <glm/gtc/matrix_transform.hpp>                     // GLM 矩阵变换库，提供矩阵变换函数（如平移、旋转、缩放等）

#include <vulkan/vulkan.h>                                  // Vulkan 核心库，提供 Vulkan API 的定义
#include "vulkanexamplebase.h"                              // Vulkan 示例基类，提供通用的 Vulkan 初始化、渲染等功能

// 我们希望让 GPU 和 CPU 保持忙碌。为此，我们可以在前一个命令缓冲区仍在执行时开始构建一个新的命令缓冲区
// 这个数字定义了可以同时处理的帧数的最大值
// 增加这个数字可能会提高性能，但也会引入额外的延迟
#define MAX_CONCURRENT_FRAMES 2

class VulkanExample : public VulkanExampleBase {
public:
    // 本示例中使用的顶点布局
    struct Vertex {
        float position[3]; // 顶点位置
        float color[3];    // 顶点颜色
    };

    // 顶点缓冲区及其属性
    struct {
        VkDeviceMemory memory{VK_NULL_HANDLE}; // 该缓冲区的设备内存句柄
        VkBuffer buffer{VK_NULL_HANDLE};       // 内存绑定到的 Vulkan 缓冲区对象句柄
    } vertices;

    // 索引缓冲区
    struct {
        VkDeviceMemory memory{VK_NULL_HANDLE}; // 设备内存句柄
        VkBuffer buffer{VK_NULL_HANDLE};       // 缓冲区对象句柄
        uint32_t count{0};                     // 索引数量
    } indices;

    // 统一缓冲区块对象
    struct UniformBuffer {
        VkDeviceMemory memory{VK_NULL_HANDLE}; // 设备内存句柄
        VkBuffer buffer{VK_NULL_HANDLE};       // 缓冲区对象句柄
        // 描述符集存储绑定到着色器绑定点的资源
        // 它将不同着色器的绑定点与用于这些绑定的缓冲区和图像连接起来
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
        // 我们保留一个指向映射缓冲区的指针，以便通过 memcpy 轻松更新其内容
        uint8_t *mapped{nullptr};
    };

    // 我们为每一帧使用一个 UBO，以便实现帧重叠，并确保在统一缓冲区仍在使用时不会更新它
    std::array<UniformBuffer, MAX_CONCURRENT_FRAMES> uniformBuffers;

    // 为了简化，我们使用与着色器中相同的统一区块布局：
    //
    //	layout(set = 0, binding = 0) uniform UBO
    //	{
    //		mat4 projectionMatrix;
    //		mat4 modelMatrix;
    //		mat4 viewMatrix;
    //	} ubo;
    //
    // 这样我们可以直接将 ubo 数据通过 memcpy 复制到 ubo 中
    // 注意：您应该使用与 GPU 对齐的数据类型，以避免手动填充（例如 vec4、mat4）
    struct ShaderData {
        glm::mat4 projectionMatrix; // 投影矩阵
        glm::mat4 modelMatrix;      // 模型矩阵
        glm::mat4 viewMatrix;       // 视图矩阵
    };

    // 管线布局（Pipeline Layout）被管线用于访问描述符集
    // 它定义了管线所使用的着色器阶段与着色器资源之间的接口（无需绑定实际数据）
    // 只要接口匹配，管线布局可以在多个管线之间共享
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};

    // 管线（通常称为“管线状态对象”）用于打包所有影响管线的状态
    // 在 OpenGL 中，几乎所有状态都可以（几乎）随时更改，而 Vulkan 要求提前定义图形（和计算）管线的状态
    // 因此，对于每种非动态管线状态的组合，都需要一个新的管线（这里不讨论一些例外情况）
    // 尽管这增加了提前规划的维度，但它为驱动程序提供了性能优化的绝佳机会
    VkPipeline pipeline{VK_NULL_HANDLE};

    // 描述符集布局描述了着色器绑定布局（无需实际引用描述符）
    // 与管线布局类似，它更像是一个蓝图，只要布局匹配，就可以与不同的描述符集一起使用
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};

    // 同步原语
    // 同步是 Vulkan 的一个重要概念，而 OpenGL 则大多将其隐藏。正确理解和使用同步对于 Vulkan 至关重要。

    // 信号量（Semaphores）用于协调图形队列中的操作，并确保命令的正确顺序
    std::array<VkSemaphore, MAX_CONCURRENT_FRAMES> presentCompleteSemaphores{}; // 用于表示呈现完成的信号量
    std::array<VkSemaphore, MAX_CONCURRENT_FRAMES> renderCompleteSemaphores{}; // 用于表示渲染完成的信号量

    VkCommandPool commandPool{VK_NULL_HANDLE}; // 命令池，用于分配命令缓冲区
    std::array<VkCommandBuffer, MAX_CONCURRENT_FRAMES> commandBuffers{}; // 命令缓冲区数组，每帧对应一个
    std::array<VkFence, MAX_CONCURRENT_FRAMES> waitFences{}; // 栅栏（Fences）数组，用于等待帧完成

    // 为了选择正确的同步对象，我们需要跟踪当前帧
    uint32_t currentFrame{0};

    VulkanExample() : VulkanExampleBase() {
        title = "Vulkan 示例 - 基本索引三角形";
        // 为了简化，我们不使用框架中的 UI 叠加层
        settings.overlay = false;                                        // 设置一个默认的 look-at 相机
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -2.5f));       // 设置相机位置
        camera.setRotation(glm::vec3(0.0f));                       // 设置相机旋转
        camera.setPerspective(60.0f, (float) width / (float) height, 1.0f, 256.0f);     // 设置透视投影参数
        // 未在此设置的参数将在基类构造函数中初始化
    }

    ~VulkanExample() override {
        // 清理使用的 Vulkan 资源
        // 注意：继承的析构函数会清理基类中存储的资源

        // 销毁管线
        vkDestroyPipeline(device, pipeline, nullptr);

        // 销毁管线布局
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        // 销毁描述符集布局
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        // 销毁顶点缓冲区及其内存
        vkDestroyBuffer(device, vertices.buffer, nullptr);
        vkFreeMemory(device, vertices.memory, nullptr);

        // 销毁索引缓冲区及其内存
        vkDestroyBuffer(device, indices.buffer, nullptr);
        vkFreeMemory(device, indices.memory, nullptr);

        // 销毁命令池
        vkDestroyCommandPool(device, commandPool, nullptr);

        // 循环清理每一帧的同步对象和统一缓冲区
        for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
            vkDestroyFence(device, waitFences[i], nullptr);                         // 销毁栅栏
            vkDestroySemaphore(device, presentCompleteSemaphores[i], nullptr);  // 销毁呈现完成信号量
            vkDestroySemaphore(device, renderCompleteSemaphores[i], nullptr);   // 销毁渲染完成信号量
            vkDestroyBuffer(device, uniformBuffers[i].buffer, nullptr);                   // 销毁统一缓冲区
            vkFreeMemory(device, uniformBuffers[i].memory,
                         nullptr);                      // 释放统一缓冲区内存
        }
    }

    // 此函数用于请求支持我们所需的所有属性标志的设备内存类型（例如设备本地内存、主机可见内存）
    // 成功时，它将返回符合我们请求的内存属性的内存类型索引
    // 这是必要的，因为实现可以提供任意数量的具有不同内存属性的内存类型
    // 您可以在 https://vulkan.gpuinfo.org/ 上查看不同内存配置的详细信息
    uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties) {
        // 遍历本示例中使用的设备的所有可用内存类型
        for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
            if ((typeBits & 1) == 1) { // 检查当前内存类型是否在请求的类型位中
                if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) ==
                    properties) { // 检查内存属性是否匹配
                    return i; // 返回匹配的内存类型索引
                }
            }
            typeBits >>= 1; // 检查下一个内存类型
        }

        throw "Could not find a suitable memory type!"; // 如果没有找到合适的内存类型，抛出异常
    }

    // 创建本示例中使用的每帧（in-flight）Vulkan 同步原语
    void createSynchronizationPrimitives() {
        // 信号量用于确保队列中命令的正确顺序
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // 栅栏用于在主机端检查绘制命令缓冲区是否完成
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // 将栅栏创建为已触发状态（这样我们不会在首次渲染每个命令缓冲区时等待）
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
            // 信号量用于确保图像呈现完成后再开始提交命令
            VK_CHECK_RESULT(
                    vkCreateSemaphore(
                            device,
                            &semaphoreCI,
                            nullptr,
                            &presentCompleteSemaphores[i]
                    )
            )
            // 信号量用于确保所有提交的命令完成后，再将图像提交到队列
            VK_CHECK_RESULT(
                    vkCreateSemaphore(
                            device,
                            &semaphoreCI,
                            nullptr,
                            &renderCompleteSemaphores[i]
                    )
            )

            // 栅栏用于确保命令缓冲区执行完成后再重新使用它
            VK_CHECK_RESULT(
                    vkCreateFence(
                            device,
                            &fenceCI,
                            nullptr,
                            &waitFences[i]
                    )
            )
        }
    }

    void createCommandBuffers() {
        // 所有命令缓冲区都是从命令池中分配的
        VkCommandPoolCreateInfo commandPoolCI{};
        commandPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCI.queueFamilyIndex = swapChain.queueNodeIndex; // 指定命令池所属的队列族
        commandPoolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // 允许重置命令缓冲区
        VK_CHECK_RESULT(
                vkCreateCommandPool(
                        device,
                        &commandPoolCI,
                        nullptr,
                        &commandPool
                )
        ) // 创建命令池

        // 从上述命令池中为每帧分配一个命令缓冲区
        VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(
                commandPool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                MAX_CONCURRENT_FRAMES
        ); // 分配命令缓冲区
        VK_CHECK_RESULT(
                vkAllocateCommandBuffers(
                        device,
                        &cmdBufAllocateInfo,
                        commandBuffers.data()
                )
        ) // 分配命令缓冲区
    }

// 准备用于索引三角形的顶点和索引缓冲区
// 同时使用暂存缓冲区将它们上传到设备本地内存，并初始化顶点输入和属性绑定以匹配顶点着色器
    void createVertexBuffer() {
        // 关于 Vulkan 内存管理的一般说明：
        // 这是一个非常复杂的话题，虽然示例应用程序可以简单地使用单独的小内存分配，但这并不是
        // 真实应用程序中应该采用的方式，在真实应用程序中，你应该一次性分配大块内存。

        // 设置顶点数据
        std::vector<Vertex> vertexBuffer{
                {{1.0f,  1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}}, // 顶点 1：位置和颜色
                {{-1.0f, 1.0f,  0.0f}, {0.0f, 1.0f, 0.0f}}, // 顶点 2：位置和颜色
                {{0.0f,  -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}  // 顶点 3：位置和颜色
        };
        uint32_t vertexBufferSize =
                static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex); // 计算顶点缓冲区大小

        // 设置索引数据
        std::vector<uint32_t> indexBuffer{0, 1, 2}; // 索引数据
        indices.count = static_cast<uint32_t>(indexBuffer.size()); // 索引数量
        uint32_t indexBufferSize = indices.count * sizeof(uint32_t); // 计算索引缓冲区大小

        VkMemoryAllocateInfo memAlloc{};
        memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; // 内存分配信息
        VkMemoryRequirements memReqs; // 内存需求

        // 静态数据（如顶点和索引缓冲区）应存储在设备内存中，以便 GPU 以最优（且最快）的方式访问
        //
        // 为此，我们使用所谓的“暂存缓冲区”：
        // - 创建一个对主机可见（并且可以映射）的缓冲区
        // - 将数据复制到此缓冲区
        // - 在设备（VRAM）上创建另一个大小相同的本地缓冲区
        // - 使用命令缓冲区将数据从主机复制到设备
        // - 删除主机可见的（暂存）缓冲区
        // - 使用设备本地缓冲区进行渲染
        //
        // 注意：在主机（CPU）和 GPU 共享同一内存的统一内存架构上，暂存缓冲区不是必需的
        // 为了保持本示例的简洁性，这里没有进行此类检查
        struct StagingBuffer {
            VkDeviceMemory memory;
            VkBuffer buffer;
        };

        struct {
            StagingBuffer vertices;
            StagingBuffer indices;
        } stagingBuffers{};

        void *data;

        // 顶点缓冲区
        VkBufferCreateInfo vertexBufferInfoCI{};
        vertexBufferInfoCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; // 缓冲区创建信息
        vertexBufferInfoCI.size = vertexBufferSize; // 缓冲区大小
        // 缓冲区用作复制源
        vertexBufferInfoCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; // 缓冲区用途
        // 创建一个主机可见的缓冲区以复制顶点数据（暂存缓冲区）
        VK_CHECK_RESULT(
                vkCreateBuffer(
                        device,
                        &vertexBufferInfoCI,
                        nullptr,
                        &stagingBuffers.vertices.buffer
                )
        ) // 创建缓冲区
        vkGetBufferMemoryRequirements(
                device,
                stagingBuffers.vertices.buffer,
                &memReqs
        ); // 获取内存需求

        memAlloc.allocationSize = memReqs.size; // 设置内存分配大小

        // 请求一个主机可见的内存类型，用于将数据复制到其中
        // 同时请求内存为一致的（coherent），以便在取消映射缓冲区后，写入的内容对 GPU 立即可见
        memAlloc.memoryTypeIndex = getMemoryTypeIndex(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        ); // 获取内存类型索引
        VK_CHECK_RESULT(
                vkAllocateMemory(
                        device,
                        &memAlloc,
                        nullptr,
                        &stagingBuffers.vertices.memory
                )
        ) // 分配内存

        // 映射并复制数据
        VK_CHECK_RESULT(
                vkMapMemory(
                        device,
                        stagingBuffers.vertices.memory,
                        0,
                        memAlloc.allocationSize,
                        0,
                        &data
                )
        ) // 映射内存

        memcpy(data, vertexBuffer.data(), vertexBufferSize); // 复制数据

        vkUnmapMemory(device, stagingBuffers.vertices.memory); // 取消映射内存

        VK_CHECK_RESULT(
                vkBindBufferMemory(
                        device,
                        stagingBuffers.vertices.buffer,
                        stagingBuffers.vertices.memory,
                        0
                )
        ) // 绑定内存到缓冲区

        // 创建一个设备本地缓冲区，用于将（主机本地）顶点数据复制到其中，并用于渲染
        vertexBufferInfoCI.usage =
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // 缓冲区用途
        VK_CHECK_RESULT(
                vkCreateBuffer(
                        device,
                        &vertexBufferInfoCI,
                        nullptr,
                        &vertices.buffer
                )
        ) // 创建缓冲区
        vkGetBufferMemoryRequirements(device, vertices.buffer, &memReqs); // 获取内存需求
        memAlloc.allocationSize = memReqs.size; // 设置内存分配大小
        // 获取内存类型索引
        memAlloc.memoryTypeIndex = getMemoryTypeIndex(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // 分配内存
        VK_CHECK_RESULT(
                vkAllocateMemory(
                        device,
                        &memAlloc,
                        nullptr,
                        &vertices.memory
                )
        )

        // 绑定内存到缓冲区
        VK_CHECK_RESULT(
                vkBindBufferMemory(
                        device,
                        vertices.buffer,
                        vertices.memory,
                        0
                )
        )

        // 索引缓冲区
        VkBufferCreateInfo indexBufferCI{};
        indexBufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; // 缓冲区创建信息
        indexBufferCI.size = indexBufferSize; // 缓冲区大小
        indexBufferCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; // 缓冲区用途

        // 将索引数据复制到主机可见的缓冲区（暂存缓冲区）
        // 创建缓冲区
        VK_CHECK_RESULT(
                vkCreateBuffer(
                        device,
                        &indexBufferCI,
                        nullptr,
                        &stagingBuffers.indices.buffer
                )
        )
        // 获取内存需求
        vkGetBufferMemoryRequirements(
                device,
                stagingBuffers.indices.buffer,
                &memReqs
        );
        // 设置内存分配大小
        memAlloc.allocationSize = memReqs.size;
        // 获取内存类型索引
        memAlloc.memoryTypeIndex = getMemoryTypeIndex(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        // 分配内存给索引暂存缓冲区
        VK_CHECK_RESULT(
                vkAllocateMemory(
                        device,                       // Vulkan 设备句柄
                        &memAlloc,                    // 内存分配信息（如大小、内存类型等）
                        nullptr,                      // 自定义分配器（通常为 nullptr）
                        &stagingBuffers.indices.memory // 指向分配的内存句柄的指针
                )
        )
        // 将索引暂存缓冲区的内存映射到主机（CPU）可访问的地址空间
        VK_CHECK_RESULT(
                vkMapMemory(
                        device,                       // Vulkan 设备句柄
                        stagingBuffers.indices.memory, // 要映射的内存句柄
                        0,                            // 映射的偏移量（从内存起始位置开始）
                        indexBufferSize,              // 映射的内存大小
                        0,                            // 映射标志（通常为 0）
                        &data                         // 指向映射内存的指针
                )
        )

        // 将索引数据从主机内存复制到映射的 GPU 内存中
        memcpy(data, indexBuffer.data(), indexBufferSize);

        // 取消映射 GPU 内存，使主机无法再访问该内存区域
        vkUnmapMemory(device, stagingBuffers.indices.memory);

        // 将索引缓冲区与分配的内存绑定
        VK_CHECK_RESULT(
                vkBindBufferMemory(
                        device,                       // Vulkan 设备句柄
                        stagingBuffers.indices.buffer, // 要绑定的缓冲区句柄
                        stagingBuffers.indices.memory, // 要绑定的内存句柄
                        0                             // 内存偏移量（从内存起始位置开始）
                )
        )

        // 创建一个仅对设备可见的目标缓冲区
        indexBufferCI.usage =
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // 缓冲区用途

        // 创建缓冲区
        VK_CHECK_RESULT(
                vkCreateBuffer(
                        device,
                        &indexBufferCI,
                        nullptr,
                        &indices.buffer
                )
        )
        // 获取内存需求
        vkGetBufferMemoryRequirements(device, indices.buffer, &memReqs);
        // 设置内存分配大小
        memAlloc.allocationSize = memReqs.size;
        // 获取内存类型索引
        memAlloc.memoryTypeIndex = getMemoryTypeIndex(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        // 分配内存
        VK_CHECK_RESULT(
                vkAllocateMemory(
                        device,
                        &memAlloc,
                        nullptr,
                        &indices.memory
                )
        )

        // 绑定内存到缓冲区
        VK_CHECK_RESULT(
                vkBindBufferMemory(
                        device,
                        indices.buffer,
                        indices.memory,
                        0
                )
        )

        // 缓冲区复制操作需要提交到队列，因此我们需要一个命令缓冲区来执行这些操作
        // 注意：某些设备提供了一个专用的传输队列（仅设置了传输位），在进行大量复制操作时可能会更快
        VkCommandBuffer copyCmd; // 用于执行复制操作的命令缓冲区句柄

        // 设置命令缓冲区分配信息
        VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
        cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; // 结构体类型
        cmdBufAllocateInfo.commandPool = commandPool; // 命令池，用于分配命令缓冲区
        cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // 命令缓冲区级别（主命令缓冲区）
        cmdBufAllocateInfo.commandBufferCount = 1; // 分配的命令缓冲区数量

        // 分配命令缓冲区
        VK_CHECK_RESULT(
                vkAllocateCommandBuffers(
                        device,            // Vulkan 设备句柄
                        &cmdBufAllocateInfo, // 命令缓冲区分配信息
                        &copyCmd           // 指向分配的命缓冲区句柄的指针
                )
        )

        // 命令缓冲区开始信息
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        // 开始记录命令缓冲区
        VK_CHECK_RESULT(
                vkBeginCommandBuffer(
                        copyCmd,  // 要记录的命令缓冲区句柄
                        &cmdBufInfo // 命令缓冲区开始记录的信息（如使用标志等）
                )
        )

        // 将缓冲区区域复制操作放入命令缓冲区
        VkBufferCopy copyRegion{}; // 缓冲区复制区域信息

        // 顶点缓冲区复制
        copyRegion.size = vertexBufferSize; // 设置复制区域的大小为顶点缓冲区的大小
        vkCmdCopyBuffer(
                copyCmd,                          // 命令缓冲区句柄
                stagingBuffers.vertices.buffer,    // 源缓冲区（暂存缓冲区）
                vertices.buffer,                  // 目标缓冲区（设备本地缓冲区）
                1,                                // 复制的区域数量
                &copyRegion                       // 复制区域信息
        ); // 执行顶点缓冲区的复制操作

        // 索引缓冲区复制
        copyRegion.size = indexBufferSize; // 设置复制区域的大小为索引缓冲区的大小
        vkCmdCopyBuffer(
                copyCmd,                         // 命令缓冲区句柄
                stagingBuffers.indices.buffer,    // 源缓冲区（暂存缓冲区）
                indices.buffer,                   // 目标缓冲区（设备本地缓冲区）
                1,                               // 复制的区域数量
                &copyRegion                      // 复制区域信息
        ); // 执行索引缓冲区的复制操作

        VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd)) // 结束命令缓冲区

        // 将命令缓冲区提交到队列以完成复制操作
        VkSubmitInfo submitInfo{}; // 提交信息
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; // 提交信息类型
        submitInfo.commandBufferCount = 1; // 命令缓冲区数量
        submitInfo.pCommandBuffers = &copyCmd; // 命令缓冲区指针

        // 创建一个栅栏以确保命令缓冲区已完成执行
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; // 栅栏创建信息的结构体类型
        fenceCI.flags = 0; // 栅栏标志（默认为 0，表示未触发状态）
        VkFence fence; // 栅栏句柄
        VK_CHECK_RESULT(
                vkCreateFence(
                        device,   // Vulkan 设备句柄
                        &fenceCI, // 栅栏创建信息
                        nullptr,  // 自定义分配器（通常为 nullptr）
                        &fence    // 指向创建的栅栏句柄的指针
                )
        ); // 创建栅栏

        // 提交命令缓冲区到队列
        VK_CHECK_RESULT(
                vkQueueSubmit(
                        queue,      // 目标队列句柄
                        1,         // 提交的命令缓冲区数量
                        &submitInfo, // 提交信息（包含命令缓冲区、信号量等）
                        fence      // 栅栏句柄，用于同步命令缓冲区执行完成
                )
        ); // 提交命令缓冲区

        // 等待栅栏信号，表示命令缓冲区已完成执行
        VK_CHECK_RESULT(
                vkWaitForFences(
                        device,   // Vulkan 设备句柄
                        1,        // 等待的栅栏数量
                        &fence,   // 栅栏句柄
                        VK_TRUE,  // 等待所有栅栏信号
                        DEFAULT_FENCE_TIMEOUT // 超时时间（如果栅栏未触发，等待的最长时间）
                )
        ); // 等待栅栏

        vkDestroyFence(device, fence, nullptr); // 销毁栅栏
        vkFreeCommandBuffers(device, commandPool, 1, &copyCmd); // 释放命令缓冲区
        // 销毁暂存缓冲区
        // 注意：暂存缓冲区必须在复制操作提交并执行完成后才能删除
        vkDestroyBuffer(device, stagingBuffers.vertices.buffer, nullptr); // 销毁顶点暂存缓冲区
        vkFreeMemory(device, stagingBuffers.vertices.memory, nullptr);   // 释放顶点暂存缓冲区内存
        vkDestroyBuffer(device, stagingBuffers.indices.buffer, nullptr);  // 销毁索引暂存缓冲区
        vkFreeMemory(device, stagingBuffers.indices.memory, nullptr);    // 释放索引暂存缓冲区内存
    }

    // 描述符是从池中分配的，池会告诉实现我们最多要使用多少个以及什么类型的描述符
    void createDescriptorPool() {
        // 我们需要告诉 API 每种类型所需的最大描述符数量
        VkDescriptorPoolSize descriptorTypeCounts[1]{}; // 描述符类型数量数组
        // 本示例仅使用一种描述符类型（统一缓冲区）
        descriptorTypeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // 描述符类型为统一缓冲区
        // 每帧有一个缓冲区（因此有一个描述符）
        descriptorTypeCounts[0].descriptorCount = MAX_CONCURRENT_FRAMES; // 描述符数量
        // 对于其他类型，您需要在类型数量列表中添加新条目
        // 例如，对于两个组合图像采样器：
        // typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        // typeCounts[1].descriptorCount = 2;

        // 创建全局描述符池
        // 本示例中使用的所有描述符都从该池中分配
        VkDescriptorPoolCreateInfo descriptorPoolCI{}; // 描述符池创建信息
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; // 结构体类型
        descriptorPoolCI.pNext = nullptr; // 下一个指针
        descriptorPoolCI.poolSizeCount = 1; // 描述符类型数量
        descriptorPoolCI.pPoolSizes = descriptorTypeCounts; // 描述符类型数量数组
        // 设置可以从该池中请求的描述符集的最大数量（超出此限制的请求将导致错误）
        // 我们的示例将为每帧的每个统一缓冲区创建一个描述符集
        descriptorPoolCI.maxSets = MAX_CONCURRENT_FRAMES; // 最大描述符集数量
        VK_CHECK_RESULT(
                vkCreateDescriptorPool(
                        device,
                        &descriptorPoolCI,
                        nullptr,
                        &descriptorPool
                )
        ) // 创建描述符池
    }

    // 描述符集布局定义了应用程序与着色器之间的接口
    // 基本上将不同的着色器阶段连接到用于绑定统一缓冲区、图像采样器等资源的描述符
    // 因此，每个着色器绑定都应映射到一个描述符集布局绑定
    void createDescriptorSetLayout() {
        // 绑定 0：统一缓冲区（顶点着色器）
        VkDescriptorSetLayoutBinding layoutBinding{}; // 描述符集布局绑定
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // 描述符类型为统一缓冲区
        layoutBinding.descriptorCount = 1; // 描述符数量
        layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; // 着色器阶段为顶点着色器
        layoutBinding.pImmutableSamplers = nullptr; // 不可变采样器（未使用）

        VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{}; // 描述符集布局创建信息
        descriptorLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; // 结构体类型
        descriptorLayoutCI.pNext = nullptr; // 下一个指针
        descriptorLayoutCI.bindingCount = 1; // 绑定数量
        descriptorLayoutCI.pBindings = &layoutBinding; // 绑定信息
        VK_CHECK_RESULT(
                vkCreateDescriptorSetLayout(
                        device,
                        &descriptorLayoutCI,
                        nullptr,
                        &descriptorSetLayout
                )
        ) // 创建描述符集布局
    }

    // 着色器通过描述符集访问数据，描述符集“指向”我们的统一缓冲区
    // 描述符集使用上面创建的描述符集布局
    void createDescriptorSets() {
        // 从全局描述符池中为每帧分配一个描述符集
        for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
            VkDescriptorSetAllocateInfo allocInfo{}; // 描述符集分配信息
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; // 结构体类型
            allocInfo.descriptorPool = descriptorPool; // 描述符池
            allocInfo.descriptorSetCount = 1; // 描述符集数量
            allocInfo.pSetLayouts = &descriptorSetLayout; // 描述符集布局
            VK_CHECK_RESULT(
                    vkAllocateDescriptorSets(
                            device,
                            &allocInfo,
                            &uniformBuffers[i].descriptorSet
                    )
            ) // 分配描述符集

            // 更新描述符集，确定着色器绑定点
            // 对于着色器中使用的每个绑定点，都需要有一个匹配的描述符集
            VkWriteDescriptorSet writeDescriptorSet{}; // 写入描述符集

            // 缓冲区的信息通过描述符信息结构传递
            VkDescriptorBufferInfo bufferInfo{}; // 缓冲区信息
            bufferInfo.buffer = uniformBuffers[i].buffer; // 缓冲区句柄
            bufferInfo.range = sizeof(ShaderData); // 缓冲区大小

            // 绑定 0：统一缓冲区
            writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; // 结构体类型
            writeDescriptorSet.dstSet = uniformBuffers[i].descriptorSet; // 目标描述符集
            writeDescriptorSet.descriptorCount = 1; // 描述符数量
            writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // 描述符类型
            writeDescriptorSet.pBufferInfo = &bufferInfo; // 缓冲区信息
            writeDescriptorSet.dstBinding = 0; // 目标绑定点
            vkUpdateDescriptorSets(
                    device,
                    1,
                    &writeDescriptorSet,
                    0,
                    nullptr
            ); // 更新描述符集
        }
    }

    // 创建用于帧缓冲区的深度（和模板）缓冲区附件
    // 注意：这是基类中虚函数的重写，从 VulkanExampleBase::prepare 中调用
    void setupDepthStencil() override {
        // 创建一个用作深度模板附件的最佳图像
        VkImageCreateInfo imageCI{}; // 图像创建信息
        imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; // 结构体类型
        imageCI.imageType = VK_IMAGE_TYPE_2D; // 图像类型为 2D
        imageCI.format = depthFormat; // 图像格式为深度格式
        // 使用示例的高度和宽度
        imageCI.extent = {width, height, 1}; // 图像范围
        imageCI.mipLevels = 1; // Mipmap 层级数
        imageCI.arrayLayers = 1; // 数组层数
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT; // 采样数
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL; // 图像布局为最佳
        imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; // 图像用途为深度模板附件
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // 初始布局为未定义
        VK_CHECK_RESULT(
                vkCreateImage(
                        device,
                        &imageCI,
                        nullptr,
                        &depthStencil.image
                )
        ) // 创建图像

        // 为图像分配内存（设备本地）并将其绑定到我们的图像
        VkMemoryAllocateInfo memAlloc{}; // 内存分配信息
        memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; // 结构体类型
        VkMemoryRequirements memReqs; // 内存需求
        vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs); // 获取图像内存需求
        memAlloc.allocationSize = memReqs.size; // 分配内存大小
        memAlloc.memoryTypeIndex = getMemoryTypeIndex(
                memReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ); // 获取内存类型索引
        VK_CHECK_RESULT(
                vkAllocateMemory(
                        device,
                        &memAlloc,
                        nullptr,
                        &depthStencil.memory
                )
        ) // 分配内存
        VK_CHECK_RESULT(
                vkBindImageMemory(
                        device,
                        depthStencil.image,
                        depthStencil.memory,
                        0
                )
        )// 绑定内存到图像

        // 为深度模板图像创建视图
        // 在 Vulkan 中，图像不能直接访问，而是通过描述子资源范围的视图访问
        // 这允许对同一图像创建多个视图，具有不同的范围（例如用于不同的层）
        VkImageViewCreateInfo depthStencilViewCI{}; // 图像视图创建信息
        depthStencilViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; // 结构体类型
        depthStencilViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D; // 视图类型为 2D
        depthStencilViewCI.format = depthFormat; // 视图格式为深度格式
        depthStencilViewCI.subresourceRange = {}; // 子资源范围
        depthStencilViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; // 子资源范围的方面为深度
        // 模板方面应仅在深度 + 模板格式上设置（VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT）
        if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
            depthStencilViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT; // 添加模板方面
        }
        depthStencilViewCI.subresourceRange.baseMipLevel = 0; // 基础 Mipmap 层级
        depthStencilViewCI.subresourceRange.levelCount = 1; // Mipmap 层级数
        depthStencilViewCI.subresourceRange.baseArrayLayer = 0; // 基础数组层
        depthStencilViewCI.subresourceRange.layerCount = 1; // 数组层数
        depthStencilViewCI.image = depthStencil.image; // 图像句柄
        VK_CHECK_RESULT(
                vkCreateImageView(device, &depthStencilViewCI, nullptr,
                                  &depthStencil.view)) // 创建图像视图
    }

    // 为交换链中的每个图像创建帧缓冲区
    // 注意：这是基类中虚函数的重写，从 VulkanExampleBase::prepare 中调用
    void setupFrameBuffer() override {
        // 为交换链中的每个图像创建帧缓冲区
        frameBuffers.resize(swapChain.images.size()); // 调整帧缓冲区大小以匹配交换链图像数量
        for (size_t i = 0; i < frameBuffers.size(); i++) {
            std::array<VkImageView, 2> attachments{}; // 附件数组
            // 颜色附件是交换链图像的视图
            attachments[0] = swapChain.imageViews[i];
            // 深度/模板附件对于所有帧缓冲区都是相同的，因为深度在当前 GPU 中的工作方式
            attachments[1] = depthStencil.view;

            VkFramebufferCreateInfo frameBufferCI{}; // 帧缓冲区创建信息
            frameBufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO; // 结构体类型
            // 所有帧缓冲区使用相同的渲染通道设置
            frameBufferCI.renderPass = renderPass; // 渲染通道
            frameBufferCI.attachmentCount = static_cast<uint32_t>(attachments.size()); // 附件数量
            frameBufferCI.pAttachments = attachments.data(); // 附件数组
            frameBufferCI.width = width; // 帧缓冲区宽度
            frameBufferCI.height = height; // 帧缓冲区高度
            frameBufferCI.layers = 1; // 帧缓冲区层数
            // 创建帧缓冲区
            VK_CHECK_RESULT(
                    vkCreateFramebuffer(
                            device,
                            &frameBufferCI,
                            nullptr,
                            &frameBuffers[i]
                    )
            ) // 创建帧缓冲区
        }
    }

    // 渲染通道设置
    // 渲染通道是 Vulkan 中的一个新概念。它们描述了渲染过程中使用的附件，并可能包含多个子通道及其附件依赖关系
    // 这使得驱动程序可以提前知道渲染的情况，并是一个优化的好机会，尤其是在基于瓦片的渲染器上（使用多个子通道时）
    // 使用子通道依赖关系还会为使用的附件添加隐式的布局转换，因此我们不需要显式地添加图像内存屏障来转换它们
    // 注意：这是基类中虚函数的重写，从 VulkanExampleBase::prepare 中调用
    void setupRenderPass() override {
        // 本示例将使用一个包含一个子通道的渲染通道

        // 渲染通道使用的附件描述
        std::array<VkAttachmentDescription, 2> attachments{};

        // 颜色附件
        attachments[0].format = swapChain.colorFormat;                                  // 使用交换链选择的颜色格式
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;                                 // 本示例中不使用多重采样
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                            // 在渲染通道开始时清除此附件
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;                          // 在渲染通道结束后保留其内容（用于显示）
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                 // 我们不使用模板，因此不关心加载操作
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;               // 同样不关心存储操作
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                       // 渲染通道开始时的布局。初始布局不重要，因此我们使用未定义
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;                   // 渲染通道结束后附件转换到的布局
        // 因为我们希望将颜色缓冲区呈现到交换链，所以将其转换为 PRESENT_KHR
        // 深度附件
        attachments[1].format = depthFormat;                                           // 在示例基类中选择了合适的深度格式
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                           // 在第一个子通道开始时清除深度
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;                     // 渲染通道结束后我们不需要深度（DONT_CARE 可能会带来更好的性能）
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                // 不使用模板
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;              // 不使用模板
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                      // 渲染通道开始时的布局。初始布局不重要，因此我们使用未定义
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // 转换为深度/模板附件

        // 设置附件引用
        VkAttachmentReference colorReference{};
        colorReference.attachment = 0;                                    // 附件 0 是颜色
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // 子通道期间用作颜色的附件布局

        VkAttachmentReference depthReference{};
        depthReference.attachment = 1;                                            // 附件 1 是深度
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // 子通道期间用作深度/模板的附件布局

        // 设置单个子通道引用
        VkSubpassDescription subpassDescription{};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;                            // 子通道使用一个颜色附件
        subpassDescription.pColorAttachments = &colorReference;                 // 引用插槽 0 中的颜色附件
        subpassDescription.pDepthStencilAttachment = &depthReference;           // 引用插槽 1 中的深度附件
        subpassDescription.inputAttachmentCount = 0;                            // 输入附件可用于从前一个子通道的内容中采样
        subpassDescription.pInputAttachments = nullptr;                         // （本示例未使用输入附件）
        subpassDescription.preserveAttachmentCount = 0;                         // 保留附件可用于在子通道之间循环（并保留）附件
        subpassDescription.pPreserveAttachments = nullptr;                      // （本示例未使用保留附件）
        subpassDescription.pResolveAttachments = nullptr;                       // 解析附件在子通道结束时解析，可用于例如多重采样

        // 设置子通道依赖关系
        // 这些将添加由附件描述指定的隐式附件布局转换
        // 实际使用布局通过附件引用中指定的布局保留
        // 每个子通道依赖关系将在源子通道和目标子通道之间引入内存和执行依赖关系，由
        // srcStageMask、dstStageMask、srcAccessMask、dstAccessMask（并设置 dependencyFlags）描述
        // 注意：VK_SUBPASS_EXTERNAL 是一个特殊常量，指代在实际渲染通道之外执行的所有命令
        std::array<VkSubpassDependency, 2> dependencies{};

        // 为深度和颜色附件执行从最终布局到初始布局的转换
        // 深度附件
        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependencies[0].dependencyFlags = 0;
        // 颜色附件
        dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].dstSubpass = 0;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].srcAccessMask = 0;
        dependencies[1].dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        dependencies[1].dependencyFlags = 0;

        // 创建实际的渲染通道
        VkRenderPassCreateInfo renderPassCI{};
        renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCI.attachmentCount = static_cast<uint32_t>(attachments.size());  // 渲染通道使用的附件数量
        renderPassCI.pAttachments = attachments.data();                            // 渲染通道使用的附件描述
        renderPassCI.subpassCount = 1;                                             // 本示例仅使用一个子通道
        renderPassCI.pSubpasses = &subpassDescription;                             // 该子通道的描述
        renderPassCI.dependencyCount = static_cast<uint32_t>(dependencies.size()); // 子通道依赖关系数量
        renderPassCI.pDependencies = dependencies.data();                          // 渲染通道使用的子通道依赖关系
        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCI, nullptr, &renderPass))
    }

    // Vulkan 从一种称为 SPIR-V 的即时二进制表示形式加载其着色器
    // 着色器是离线编译的，例如使用 GLSL 并通过参考 glslang 编译器进行编译
    // 此函数从二进制文件加载这样的着色器并返回一个着色器模块结构
    VkShaderModule loadSPIRVShader(std::string filename) {
        size_t shaderSize; // 着色器文件的大小
        char *shaderCode{nullptr}; // 着色器代码的指针

#if defined(__ANDROID__)
        // Load shader from compressed asset
        AAsset *asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(),
                                           AASSET_MODE_STREAMING);
        assert(asset);
        shaderSize = AAsset_getLength(asset);
        assert(shaderSize > 0);

        shaderCode = new char[shaderSize];
        AAsset_read(asset, shaderCode, shaderSize);
        AAsset_close(asset);
#else
        std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

        if (is.is_open())
        {
            shaderSize = is.tellg();
            is.seekg(0, std::ios::beg);
            // Copy file contents into a buffer
            shaderCode = new char[shaderSize];
            is.read(shaderCode, shaderSize);
            is.close();
            assert(shaderSize > 0);
        }
#endif
        if (shaderCode) {
            // Create a new shader module that will be used for pipeline creation
            VkShaderModuleCreateInfo shaderModuleCI{};
            shaderModuleCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderModuleCI.codeSize = shaderSize;
            shaderModuleCI.pCode = (uint32_t *) shaderCode;

            VkShaderModule shaderModule;
            VK_CHECK_RESULT(vkCreateShaderModule(device, &shaderModuleCI, nullptr, &shaderModule))

            delete[] shaderCode;

            return shaderModule;
        } else {
            std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
            return VK_NULL_HANDLE;
        }
    }

    void createPipelines() {
        // 创建用于生成基于此描述符集布局的渲染管线的管线布局
        // 在更复杂的场景中，您可以为不同的描述符集布局创建不同的管线布局，以便重用
        VkPipelineLayoutCreateInfo pipelineLayoutCI{}; // 管线布局创建信息结构体
        pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; // 结构体类型
        pipelineLayoutCI.pNext = nullptr; // 下一个指针（通常为 nullptr）
        pipelineLayoutCI.setLayoutCount = 1; // 描述符集布局的数量
        pipelineLayoutCI.pSetLayouts = &descriptorSetLayout; // 描述符集布局数组

        // 创建管线布局
        VK_CHECK_RESULT(
                vkCreatePipelineLayout(
                        device, // Vulkan 设备句柄
                        &pipelineLayoutCI, // 管线布局创建信息
                        nullptr, // 自定义分配器（通常为 nullptr）
                        &pipelineLayout // 指向创建的管线布局句柄的指针
                )
        );

        // 创建本示例中使用的图形管线
        // Vulkan 使用渲染管线的概念来封装固定状态，取代了 OpenGL 的复杂状态机
        // 管线随后存储在 GPU 上并进行哈希处理，使得管线切换非常快速
        // 注意：仍有一些动态状态不直接属于管线（但使用它们的信息是）

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        // 此管线使用的布局（可以在使用相同布局的多个管线之间共享）
        pipelineCI.layout = pipelineLayout;
        // 此管线附加到的渲染通道
        pipelineCI.renderPass = renderPass;

        // 构建组成管线的不同状态

        // 输入组装状态描述了如何组装图元
        // 此管线将顶点数据组装为三角形列表（尽管我们只使用一个三角形）
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI{};
        inputAssemblyStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; // 结构体类型
        inputAssemblyStateCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // 图元拓扑类型（三角形列表）
        // 此管线将顶点数据组装为三角形列表（尽管我们只使用一个三角形）

        // 配置光栅化状态
        VkPipelineRasterizationStateCreateInfo rasterizationStateCI{};
        rasterizationStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; // 结构体类型
        rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL; // 多边形填充模式（填充整个多边形）
        rasterizationStateCI.cullMode = VK_CULL_MODE_NONE; // 剔除模式（不剔除任何面）
        rasterizationStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // 正面定义（逆时针为正面）
        rasterizationStateCI.depthClampEnable = VK_FALSE; // 是否启用深度截断（禁用）
        rasterizationStateCI.rasterizerDiscardEnable = VK_FALSE; // 是否禁用光栅化（禁用，即启用光栅化）
        rasterizationStateCI.depthBiasEnable = VK_FALSE; // 是否启用深度偏移（禁用）
        rasterizationStateCI.lineWidth = 1.0f; // 线宽（用于线框模式，默认为 1.0f）

        // 配置颜色混合状态
        // 颜色混合状态描述了如何计算混合因子（如果使用）
        // 每个颜色附件需要一个混合附件状态（即使不使用混合）
        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.colorWriteMask = 0xf; // 颜色写入掩码（允许写入所有颜色通道：R、G、B、A）
        blendAttachmentState.blendEnable = VK_FALSE; // 是否启用混合（禁用）

        VkPipelineColorBlendStateCreateInfo colorBlendStateCI{};
        colorBlendStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; // 结构体类型
        colorBlendStateCI.attachmentCount = 1; // 颜色附件数量
        colorBlendStateCI.pAttachments = &blendAttachmentState; // 指向颜色混合附件状态的指针

        // 配置视口状态
        // 视口状态设置此管线中使用的视口和裁剪器的数量
        // 注意：这实际上被动态状态覆盖（见下文）
        VkPipelineViewportStateCreateInfo viewportStateCI{};
        viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; // 结构体类型
        viewportStateCI.viewportCount = 1; // 视口数量（设置为 1 表示使用一个视口）
        viewportStateCI.scissorCount = 1;  // 裁剪器数量（设置为 1 表示使用一个裁剪器）

        // 启用动态状态
        // 大多数状态被烘焙到管线中，但仍有一些动态状态可以在命令缓冲区中更改
        // 为了能够更改这些状态，我们需要指定将使用此管线更改哪些动态状态。它们的实际状态稍后在命令缓冲区中设置。
        // 对于此示例，我们将使用动态状态设置视口和裁剪器
        std::vector<VkDynamicState> dynamicStateEnables; // 动态状态列表
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT); // 动态视口状态
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);   // 动态裁剪器状态

        VkPipelineDynamicStateCreateInfo dynamicStateCI{};
        dynamicStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; // 结构体类型
        dynamicStateCI.pDynamicStates = dynamicStateEnables.data(); // 动态状态数组
        dynamicStateCI.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size()); // 动态状态数量

        // 配置深度和模板状态
        // 深度和模板状态，包含深度和模板的比较和测试操作
        // 我们仅使用深度测试，并希望启用深度测试和写入，并使用小于或等于进行比较
        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
        depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO; // 结构体类型
        depthStencilStateCI.depthTestEnable = VK_TRUE; // 启用深度测试
        depthStencilStateCI.depthWriteEnable = VK_TRUE; // 启用深度写入
        depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; // 深度比较操作（小于或等于通过）
        depthStencilStateCI.depthBoundsTestEnable = VK_FALSE; // 禁用深度边界测试
        depthStencilStateCI.back.failOp = VK_STENCIL_OP_KEEP; // 模板测试失败时保留当前值
        depthStencilStateCI.back.passOp = VK_STENCIL_OP_KEEP; // 模板测试通过时保留当前值
        depthStencilStateCI.back.compareOp = VK_COMPARE_OP_ALWAYS; // 模板比较操作（始终通过）
        depthStencilStateCI.stencilTestEnable = VK_FALSE; // 禁用模板测试
        depthStencilStateCI.front = depthStencilStateCI.back; // 正面模板状态与背面相同

        // 配置多重采样状态
        // 多重采样状态用于抗锯齿，但此示例不使用多重采样
        // 尽管如此，仍必须设置状态并将其传递给管线
        VkPipelineMultisampleStateCreateInfo multisampleStateCI{};
        multisampleStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; // 结构体类型
        multisampleStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // 光栅化采样数（1 表示禁用多重采样）
        multisampleStateCI.pSampleMask = nullptr; // 采样掩码（未使用，设置为 nullptr）

        // 顶点输入描述
        // 指定管线的顶点输入参数

        // 配置顶点输入绑定
        // 顶点输入绑定描述了顶点数据的来源和布局
        // 此示例在绑定点 0 处使用单个顶点输入绑定（参见 vkCmdBindVertexBuffers）
        VkVertexInputBindingDescription vertexInputBinding{};
        vertexInputBinding.binding = 0; // 绑定索引（指定绑定点）
        vertexInputBinding.stride = sizeof(Vertex); // 顶点数据之间的步幅（每个顶点数据的大小）
        vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // 输入速率（按顶点数据步进）

        // 输入属性绑定描述着色器属性位置和内存布局
        std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs{};
        // 这些与以下着色器布局匹配（参见 triangle.vert）：
        //	layout (location = 0) in vec3 inPos;
        //	layout (location = 1) in vec3 inColor;
        // 属性位置 0：位置
        vertexInputAttributs[0].binding = 0;
        vertexInputAttributs[0].location = 0;
        // 位置属性是三个 32 位有符号浮点数（R32 G32 B32）
        vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttributs[0].offset = offsetof(Vertex, position);
        // 属性位置 1：颜色
        vertexInputAttributs[1].binding = 0;
        vertexInputAttributs[1].location = 1;
        // 颜色属性是三个 32 位有符号浮点数（R32 G32 B32）
        vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexInputAttributs[1].offset = offsetof(Vertex, color);

        // 配置顶点输入状态
        // 顶点输入状态用于描述管线的顶点输入绑定和属性
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO; // 结构体类型
        vertexInputStateCI.vertexBindingDescriptionCount = 1; // 顶点绑定描述的数量
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding; // 指向顶点绑定描述的指针
        vertexInputStateCI.vertexAttributeDescriptionCount = 2; // 顶点属性描述的数量
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributs.data(); // 指向顶点属性描述的指针

        // 着色器
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

        // 顶点着色器
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        // 设置此着色器的管线阶段
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        // 加载二进制 SPIR-V 着色器
        shaderStages[0].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.vert.spv");
        // 着色器的主入口点
        shaderStages[0].pName = "main";
        assert(shaderStages[0].module != VK_NULL_HANDLE);

        // 片段着色器
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        // 设置此着色器的管线阶段
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        // 加载二进制 SPIR-V 着色器
        shaderStages[1].module = loadSPIRVShader(getShadersPath() + "triangle/triangle.frag.spv");
        // 着色器的主入口点
        shaderStages[1].pName = "main";
        assert(shaderStages[1].module != VK_NULL_HANDLE);

        // 设置管线着色器阶段信息
        pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages = shaderStages.data();

        // 将管线状态分配给管线创建信息结构
        pipelineCI.pVertexInputState = &vertexInputStateCI; // 顶点输入状态
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI; // 输入组装状态
        pipelineCI.pRasterizationState = &rasterizationStateCI; // 光栅化状态
        pipelineCI.pColorBlendState = &colorBlendStateCI; // 颜色混合状态
        pipelineCI.pMultisampleState = &multisampleStateCI; // 多重采样状态
        pipelineCI.pViewportState = &viewportStateCI; // 视口状态
        pipelineCI.pDepthStencilState = &depthStencilStateCI; // 深度和模板状态
        pipelineCI.pDynamicState = &dynamicStateCI; // 动态状态

        // 使用指定的状态创建渲染管线
        VK_CHECK_RESULT(
                vkCreateGraphicsPipelines(
                        device,
                        pipelineCache,
                        1,
                        &pipelineCI,
                        nullptr,
                        &pipeline
                )
        )

        // 一旦图形管线创建完成，着色器模块就不再需要
        vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
        vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
    }

    void createUniformBuffers() {
        // 准备并初始化包含着色器 uniform 的每帧 uniform 缓冲区块
        // 像 OpenGL 中的单个 uniform 在 Vulkan 中不再存在。所有着色器 uniform 都通过 uniform 缓冲区块传递
        VkMemoryRequirements memReqs; // 用于存储缓冲区的内存需求

        // 顶点着色器 uniform 缓冲区块
        VkBufferCreateInfo bufferInfo{}; // 缓冲区创建信息
        VkMemoryAllocateInfo allocInfo{}; // 内存分配信息
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; // 结构体类型
        allocInfo.pNext = nullptr; // 下一个指针（通常为 nullptr）
        allocInfo.allocationSize = 0; // 分配内存大小（初始为 0，后续根据需求设置）
        allocInfo.memoryTypeIndex = 0; // 内存类型索引（初始为 0，后续根据需求设置）

        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; // 结构体类型
        bufferInfo.size = sizeof(ShaderData); // 缓冲区大小（根据 ShaderData 结构体的大小）
        // 此缓冲区将用作 uniform 缓冲区
        bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; // 缓冲区用途（uniform 缓冲区）

        // 创建缓冲区
        for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
            VK_CHECK_RESULT(
                    vkCreateBuffer(
                            device, // Vulkan 设备句柄
                            &bufferInfo, // 缓冲区创建信息
                            nullptr, // 自定义分配器（通常为 nullptr）
                            &uniformBuffers[i].buffer // 指向创建的缓冲区句柄的指针
                    )
            ) // 创建 uniform 缓冲区

            // 获取内存需求，包括大小、对齐方式和内存类型
            vkGetBufferMemoryRequirements(device, uniformBuffers[i].buffer, &memReqs);
            allocInfo.allocationSize = memReqs.size; // 设置分配内存的大小

            // 获取支持主机可见内存访问的内存类型索引
            // 大多数实现提供多种内存类型，选择正确的内存类型来分配内存至关重要
            // 我们还希望缓冲区是主机一致的，这样我们就不必在每次更新后刷新（或同步）
            // 注意：这可能会影响性能，因此在定期更新缓冲区的实际应用程序中，您可能不希望这样做
            allocInfo.memoryTypeIndex = getMemoryTypeIndex(
                    memReqs.memoryTypeBits, // 内存类型位掩码
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | // 主机可见内存
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  // 主机一致内存
            );

            // 为 uniform 缓冲区分配内存
            VK_CHECK_RESULT(
                    vkAllocateMemory(
                            device, // Vulkan 设备句柄
                            &allocInfo, // 内存分配信息
                            nullptr, // 自定义分配器（通常为 nullptr）
                            &(uniformBuffers[i].memory) // 指向分配的内存句柄的指针
                    )
            ) // 分配内存

            // 将内存绑定到缓冲区
            VK_CHECK_RESULT(
                    vkBindBufferMemory(
                            device, // Vulkan 设备句柄
                            uniformBuffers[i].buffer, // 缓冲区句柄
                            uniformBuffers[i].memory, // 内存句柄
                            0 // 内存偏移量（从内存起始位置开始）
                    )
            ) // 绑定内存到缓冲区

            // 我们映射缓冲区一次，这样我们可以在不重新映射的情况下更新它
            VK_CHECK_RESULT(
                    vkMapMemory(
                            device, // Vulkan 设备句柄
                            uniformBuffers[i].memory, // 内存句柄
                            0, // 映射的偏移量（从内存起始位置开始）
                            sizeof(ShaderData), // 映射的内存大小
                            0, // 映射标志（通常为 0）
                            (void **) &uniformBuffers[i].mapped // 指向映射内存的指针
                    )
            ) // 映射内存
        }
    }

    void prepare() override {
        // 调用基类的 prepare 方法，完成 Vulkan 示例的通用初始化
        VulkanExampleBase::prepare();

        // 创建同步原语（如信号量和栅栏）
        createSynchronizationPrimitives();

        // 创建命令缓冲区
        createCommandBuffers();

        // 创建顶点缓冲区
        createVertexBuffer();

        // 创建 uniform 缓冲区
        createUniformBuffers();

        // 创建描述符集布局
        createDescriptorSetLayout();

        // 创建描述符池
        createDescriptorPool();

        // 创建描述符集
        createDescriptorSets();

        // 创建图形管线
        createPipelines();

        // 标记准备完成
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;

        // 使用栅栏等待命令缓冲区完成执行后再重新使用它
        vkWaitForFences(device, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);
        VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentFrame]))

        // 从实现中获取下一个交换链图像
        // 注意：实现可以自由地以任何顺序返回图像，因此我们必须使用获取函数，而不能简单地循环使用图像/图像索引
        uint32_t imageIndex;
        // 从交换链中获取下一个图像
        VkResult result = vkAcquireNextImageKHR(
                device, // Vulkan 设备句柄
                swapChain.swapChain, // 交换链句柄
                UINT64_MAX, // 超时时间（无限等待）
                presentCompleteSemaphores[currentFrame], // 信号量，用于同步图像获取完成
                VK_NULL_HANDLE, // 栅栏句柄（未使用，设置为 nullptr）
                &imageIndex // 指向获取的图像索引的指针
        );

        // 检查获取图像的结果
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // 如果交换链已过期（例如窗口大小改变），重新创建交换链
            windowResize();
            return;
        } else if ((result != VK_SUCCESS) && (result != VK_SUBOPTIMAL_KHR)) {
            // 如果获取图像失败且不是次优状态，抛出异常
            throw "无法获取下一个交换链图像！";
        }

        // 更新下一帧的 uniform 缓冲区
        ShaderData shaderData{};
        shaderData.projectionMatrix = camera.matrices.perspective;
        shaderData.viewMatrix = camera.matrices.view;
        shaderData.modelMatrix = glm::mat4(1.0f);

        // 将当前矩阵复制到当前帧的 uniform 缓冲区
        // 注意：由于我们为 uniform 缓冲区请求了主机一致的内存类型，写入操作会立即对 GPU 可见
        memcpy(uniformBuffers[currentFrame].mapped, &shaderData, sizeof(ShaderData));

        // 构建命令缓冲区
        // 与 OpenGL 不同，所有渲染命令都被记录到命令缓冲区中，然后提交到队列
        // 这允许在单独的线程中提前生成工作
        // 对于基本的命令缓冲区（如本示例），记录速度非常快，因此不需要将其卸载

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);

        VkCommandBufferBeginInfo cmdBufInfo{};
        cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // 为所有帧缓冲区附件设置清除值，这些附件的 loadOp 设置为清除
        // 我们使用两个附件（颜色和深度），它们在子通道开始时被清除，因此我们需要为两者设置清除值
        VkClearValue clearValues[2]{}; // 清除值数组
        clearValues[0].color = {{0.0f, 0.0f, 0.2f, 1.0f}}; // 颜色附件的清除值（RGBA，此处为深蓝色）
        clearValues[1].depthStencil = {1.0f, 0}; // 深度和模板附件的清除值（深度为 1.0，模板为 0）

        // 配置渲染通道开始信息
        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; // 结构体类型
        renderPassBeginInfo.pNext = nullptr; // 下一个指针（通常为 nullptr）
        renderPassBeginInfo.renderPass = renderPass; // 渲染通道句柄
        renderPassBeginInfo.renderArea.offset.x = 0; // 渲染区域的起始 X 坐标
        renderPassBeginInfo.renderArea.offset.y = 0; // 渲染区域的起始 Y 坐标
        renderPassBeginInfo.renderArea.extent.width = width; // 渲染区域的宽度
        renderPassBeginInfo.renderArea.extent.height = height; // 渲染区域的高度
        renderPassBeginInfo.clearValueCount = 2; // 清除值的数量
        renderPassBeginInfo.pClearValues = clearValues; // 指向清除值数组的指针
        renderPassBeginInfo.framebuffer = frameBuffers[imageIndex]; // 帧缓冲区句柄（当前交换链图像对应的帧缓冲区）
        // 获取当前帧的命令缓冲区
        const VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
        // 开始记录命令缓冲区
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

        // 启动基类在默认渲染通道设置中指定的第一个子通道
        // 这将清除颜色和深度附件
        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // 更新动态视口状态
        VkViewport viewport{};
        viewport.height = (float) height; // 视口高度
        viewport.width = (float) width;  // 视口宽度
        viewport.minDepth = (float) 0.0f; // 最小深度值
        viewport.maxDepth = (float) 1.0f; // 最大深度值
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport); // 设置视口

        // 更新动态裁剪器状态
        VkRect2D scissor{};
        scissor.extent.width = width;  // 裁剪器宽度
        scissor.extent.height = height; // 裁剪器高度
        scissor.offset.x = 0; // 裁剪器起始 X 坐标
        scissor.offset.y = 0; // 裁剪器起始 Y 坐标
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor); // 设置裁剪器

        // 绑定当前帧的 uniform 缓冲区的描述符集，以便着色器在此次绘制中使用该缓冲区的数据
        vkCmdBindDescriptorSets(
                commandBuffer, // 命令缓冲区句柄
                VK_PIPELINE_BIND_POINT_GRAPHICS, // 管线绑定点（图形管线）
                pipelineLayout, // 管线布局句柄
                0, // 第一个描述符集的索引
                1, // 描述符集的数量
                &uniformBuffers[currentFrame].descriptorSet, // 描述符集数组
                0, // 动态偏移量的数量
                nullptr // 动态偏移量数组
        );
        // 绑定渲染管线
        // 管线（状态对象）包含渲染管线的所有状态，绑定它将设置管线创建时指定的所有状态
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        // 绑定三角形顶点缓冲区（包含位置和颜色）
        VkDeviceSize offsets[1]{0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
        // 绑定三角形索引缓冲区
        vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        // 绘制索引三角形
        vkCmdDrawIndexed(commandBuffer, indices.count, 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        // 结束渲染通道将添加一个隐式屏障，将帧缓冲区颜色附件转换为
        // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR，以便将其呈现到窗口系统
        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer))

        // 将命令缓冲区提交到图形队列

        // 队列提交将等待的管线阶段（通过 pWaitSemaphores）
        VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // 提交信息结构指定命令缓冲区队列提交批次
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pWaitDstStageMask = &waitStageMask;      // 指向信号量等待发生的管线阶段列表
        submitInfo.pCommandBuffers = &commandBuffer;        // 在此批次（提交）中执行的命令缓冲区
        submitInfo.commandBufferCount = 1;                  // 我们提交一个命令缓冲区

        // 在提交的命令缓冲区开始执行之前等待的信号量
        submitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
        submitInfo.waitSemaphoreCount = 1;
        // 当命令缓冲区完成时发出信号的信号量
        submitInfo.pSignalSemaphores = &renderCompleteSemaphores[currentFrame];
        submitInfo.signalSemaphoreCount = 1;

        // 提交到图形队列并传递一个等待栅栏
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentFrame]))

        // 将当前帧缓冲区呈现到交换链
        // 将提交信息中由命令缓冲区提交发出的信号量作为交换链呈现的等待信号量传递
        // 这确保在提交所有命令之前不会将图像呈现到窗口系统

        VkPresentInfoKHR presentInfo{}; // 呈现信息结构体
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; // 结构体类型
        presentInfo.waitSemaphoreCount = 1; // 等待的信号量数量
        presentInfo.pWaitSemaphores = &renderCompleteSemaphores[currentFrame]; // 等待的信号量数组（渲染完成信号量）
        presentInfo.swapchainCount = 1; // 交换链数量
        presentInfo.pSwapchains = &swapChain.swapChain; // 交换链数组
        presentInfo.pImageIndices = &imageIndex; // 要呈现的图像索引数组
        result = vkQueuePresentKHR(queue, &presentInfo); // 将图像呈现到交换链

        if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR)) {
            windowResize();
        } else if (result != VK_SUCCESS) {
            throw "无法将图像呈现到交换链！";
        }

        // 根据最大并发帧数选择下一帧进行渲染
        currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
    }
};

// 操作系统特定的主入口点
// 大部分代码库在支持的不同操作系统之间是共享的，但像消息处理这样的内容有所不同

#if defined(_WIN32)
// Windows entry point
VulkanExample *vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (vulkanExample != NULL)
    {
        vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
    }
    return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(_In_ HINSTANCE hInstance, _In_opt_  HINSTANCE hPrevInstance, _In_ LPSTR, _In_ int)
{
    for (size_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->setupWindow(hInstance, WndProc);
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}

#elif defined(__ANDROID__)
// Android entry point
VulkanExample *vulkanExample;

void android_main(android_app *state) {
    vulkanExample = new VulkanExample();
    state->userData = vulkanExample;
    state->onAppCmd = VulkanExample::handleAppCommand;
    state->onInputEvent = VulkanExample::handleAppInput;
    androidApp = state;
    vulkanExample->renderLoop();
    delete (vulkanExample);
}

#elif defined(_DIRECT2DISPLAY)

// Linux entry point with direct to display wsi
// Direct to Displays (D2D) is used on embedded platforms
VulkanExample *vulkanExample;
static void handleEvent()
{
}
int main(const int argc, const char *argv[])
{
    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
VulkanExample *vulkanExample;
static void handleEvent(const DFBWindowEvent *event)
{
    if (vulkanExample != NULL)
    {
        vulkanExample->handleEvent(event);
    }
}
int main(const int argc, const char *argv[])
{
    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->setupWindow();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
{
    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->setupWindow();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}
#elif defined(__linux__) || defined(__FreeBSD__)

// Linux entry point
VulkanExample *vulkanExample;
#if defined(VK_USE_PLATFORM_XCB_KHR)
static void handleEvent(const xcb_generic_event_t *event)
{
    if (vulkanExample != NULL)
    {
        vulkanExample->handleEvent(event);
    }
}
#else
static void handleEvent()
{
}
#endif
int main(const int argc, const char *argv[])
{
    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->setupWindow();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}
#elif (defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT)) && defined(VK_EXAMPLE_XCODE_GENERATED)
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
{
    @autoreleasepool
    {
        for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
        vulkanExample = new VulkanExample();
        vulkanExample->initVulkan();
        vulkanExample->setupWindow(nullptr);
        vulkanExample->prepare();
        vulkanExample->renderLoop();
        delete(vulkanExample);
    }
    return 0;
}
#elif defined(VK_USE_PLATFORM_SCREEN_QNX)
VULKAN_EXAMPLE_MAIN()
#endif
