/*
* Vulkan Example base class
*
* Copyright (C) 2016-2025 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#ifdef _WIN32
#pragma comment(linker, "/subsystem:windows")
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <ShellScalingAPI.h>
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
#include <android/native_activity.h>
#include <android/asset_manager.h>
#include <android_native_app_glue.h>
#include <sys/system_properties.h>
#include "VulkanAndroid.h"
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
#include <directfb.h>
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
#include <wayland-client.h>
#include "xdg-shell-client-protocol.h"
#elif defined(_DIRECT2DISPLAY)
//
#elif defined(VK_USE_PLATFORM_XCB_KHR)
#include <xcb/xcb.h>
#endif

// 标准库头文件
#include <cstdio>      // 标准输入输出库，提供文件操作和格式化输入输出功能
#include <cstdlib>     // 标准库，提供内存管理、程序控制等功能
#include <cstring>     // 字符串处理库，提供字符串操作函数
#include <cassert>     // 断言库，用于调试时检查条件是否成立
#include <vector>       // 动态数组容器，提供高效的动态数组操作
#include <array>        // 固定大小数组容器，提供静态数组功能
#include <unordered_map> // 无序映射容器，基于哈希表实现，提供高效的键值对存储和查找
#include <numeric>      // 数值算法库，提供数值计算相关函数（如求和、累加等）
#include <ctime>        // 时间库，提供日期和时间处理功能
#include <iostream>     // 输入输出流库，提供标准输入输出功能（如 cout、cin）
#include <chrono>       // 时间库，提供高精度计时和时钟功能
#include <random>       // 随机数库，提供随机数生成器及相关功能
#include <algorithm>    // 算法库，提供常用算法（如排序、查找、遍历等）
#include <sys/stat.h>   // 系统状态库，提供文件状态查询功能（如文件是否存在、权限等）

#define GLM_FORCE_RADIANS       // 强制 GLM 使用弧度制而非角度制，确保数学计算的统一性
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // 强制 GLM 使用深度范围为 [0, 1]，而非默认的 [-1, 1]
#define GLM_ENABLE_EXPERIMENTAL // 启用 GLM 的实验性功能（如额外的数学工具）

#include <glm/glm.hpp>         // GLM 核心库，提供向量、矩阵等基本数学结构
#include <glm/gtc/matrix_transform.hpp> // 矩阵变换库，提供平移、旋转、缩放等矩阵操作
#include <glm/gtc/matrix_inverse.hpp> // 矩阵逆运算库，提供矩阵求逆功能
#include <glm/gtc/type_ptr.hpp> // 类型指针库，提供将 GLM 类型转换为原始指针的功能

#include <string>             // 字符串库，提供字符串操作功能
#include <numeric>            // 数值算法库，提供数值计算相关函数（如求和、累加等）
#include <array>              // 固定大小数组容器，提供静态数组功能

// Vulkan 头文件
#include "vulkan/vulkan.h" // Vulkan API

// 项目自定义头文件
#include "CommandLineParser.hpp" // 命令行参数解析器，用于解析和处理命令行输入
#include "keycodes.hpp"          // 键码定义文件，提供键盘按键的常量定义

// Vulkan 工具和辅助模块
#include "VulkanTools.h"        // Vulkan 工具函数，提供常用的 Vulkan 辅助功能
#include "VulkanDebug.h"        // Vulkan 调试工具，用于启用和配置 Vulkan 调试层和回调
#include "VulkanUIOverlay.h"    // Vulkan UI 叠加层，用于集成 ImGui 并渲染调试 UI
#include "VulkanSwapChain.h"    // Vulkan 交换链管理，负责创建和管理交换链
#include "VulkanBuffer.h"       // Vulkan 缓冲区管理，用于创建和管理 Vulkan 缓冲区
#include "VulkanDevice.h"       // Vulkan 设备管理，封装了物理设备和逻辑设备的创建与操作
#include "VulkanTexture.h"      // Vulkan 纹理管理，用于创建和管理 Vulkan 纹理资源

#include "VulkanInitializers.hpp" // Vulkan 初始化工具，提供简化 Vulkan 对象创建的辅助函数
#include "camera.hpp"           // 摄像机模块，用于实现 3D 场景中的视角控制和投影计算
#include "benchmark.hpp"        // 性能基准测试工具，用于测量和分析应用程序的性能

class VulkanExampleBase {
private:
    // 窗口相关
    std::string getWindowTitle() const; // 获取窗口标题
    uint32_t destWidth;                 // 目标宽度（用于窗口调整）
    uint32_t destHeight;                // 目标高度
    bool resizing = false;              // 窗口调整中标志

    // 输入处理
    void handleMouseMove(int32_t x, int32_t y); // 处理鼠标移动事件

    // 帧更新
    void nextFrame(); // 进入下一帧，更新逻辑和渲染状态

    // UI 叠加层
    void updateOverlay(); // 更新 UI 叠加层内容

    // Vulkan 资源管理
    void createPipelineCache(); // 创建管线缓存，用于加速管线创建
    void createCommandPool();   // 创建命令池，用于分配命令缓冲区
    void createSynchronizationPrimitives(); // 创建同步原语（信号量、栅栏等）
    void createSurface();       // 创建 Vulkan 表面，用于与窗口系统交互
    void createSwapChain();     // 创建交换链，管理图像呈现
    void createCommandBuffers(); // 创建命令缓冲区，用于记录渲染命令
    void destroyCommandBuffers(); // 销毁命令缓冲区，释放资源

    // 着色器目录
    std::string shaderDir = "glsl"; // 着色器文件目录，默认为 "glsl"
protected:
    // 返回 GLSL、HLSL 或 Slang 着色器目录的根路径
    std::string getShadersPath() const;

    // 帧计数器，用于显示 FPS
    uint32_t frameCounter = 0;
    uint32_t lastFPS = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp, tPrevEnd;

    // Vulkan 核心对象
    VkInstance instance{VK_NULL_HANDLE}; // Vulkan 实例
    std::vector<std::string> supportedInstanceExtensions; // 支持的实例扩展列表
    // Vulkan 使用的物理设备（GPU）
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    // 存储物理设备的属性（例如检查设备限制）
    VkPhysicalDeviceProperties deviceProperties{};
    // 存储所选物理设备上可用的特性（例如检查某个特性是否可用）
    VkPhysicalDeviceFeatures deviceFeatures{};
    // 存储物理设备上所有可用的内存（类型）属性
    VkPhysicalDeviceMemoryProperties deviceMemoryProperties{};
    /** @brief 为此示例启用的物理设备特性集合（必须在派生类的构造函数中设置） */
    VkPhysicalDeviceFeatures enabledFeatures{};
    /** @brief 为此示例启用的设备扩展集合（必须在派生类的构造函数中设置） */
    std::vector<const char *> enabledDeviceExtensions;
    std::vector<const char *> enabledInstanceExtensions;
    /** @brief 可选的 pNext 结构，用于传递扩展结构到设备创建过程中 */
    void *deviceCreatepNextChain = nullptr;
    /** @brief 逻辑设备，应用程序对物理设备（GPU）的视图 */
    VkDevice device{VK_NULL_HANDLE};
    // 设备图形队列的句柄，命令缓冲区将提交到此队列
    VkQueue queue{VK_NULL_HANDLE};
    // 深度缓冲格式（在 Vulkan 初始化期间选择）
    VkFormat depthFormat;
    // 命令缓冲区池
    VkCommandPool cmdPool{VK_NULL_HANDLE};
    /** @brief 用于图形队列提交时等待的管线阶段 */
    VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // 包含要提交到队列的命令缓冲区和信号量
    VkSubmitInfo submitInfo;
    // 用于渲染的命令缓冲区
    std::vector<VkCommandBuffer> drawCmdBuffers;
    // 全局渲染流程，用于帧缓冲写入
    VkRenderPass renderPass{VK_NULL_HANDLE};
    // 可用的帧缓冲区列表（与交换链图像数量相同）
    std::vector<VkFramebuffer> frameBuffers;
    // 当前活动的帧缓冲区索引
    uint32_t currentBuffer = 0;
    // 描述符集池
    VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
    // 创建的着色器模块列表（存储用于清理）
    std::vector<VkShaderModule> shaderModules;
    // 管线缓存对象
    VkPipelineCache pipelineCache{VK_NULL_HANDLE};
    // 封装交换链，用于将图像（帧缓冲区）呈现到窗口系统
    VulkanSwapChain swapChain;
    // 同步信号量
    struct {
        // 交换链图像呈现完成信号量
        VkSemaphore presentComplete;
        // 命令缓冲区提交和执行完成信号量
        VkSemaphore renderComplete;
    } semaphores;
    std::vector<VkFence> waitFences; // 等待栅栏列表
    bool requiresStencil{false}; // 是否需要模板测试
public:
    bool prepared = false; // 标记 Vulkan 资源是否已准备就绪
    bool resized = false;  // 标记窗口是否已调整大小
    bool viewUpdated = false; // 标记视图是否已更新（例如摄像机移动）
    uint32_t width = 1280; // 窗口宽度
    uint32_t height = 720; // 窗口高度

    vks::UIOverlay ui; // ImGui UI 叠加层对象
    CommandLineParser commandLineParser; // 命令行参数解析器

    /** @brief 使用高性能计时器（如果可用）测量的上一帧时间 */
    float frameTimer = 1.0f; // 帧时间（秒）

    vks::Benchmark benchmark; // 性能基准测试工具

    /** @brief 封装的物理和逻辑 Vulkan 设备 */
    vks::VulkanDevice *vulkanDevice; // Vulkan 设备管理对象

    /** @brief 示例设置，可以通过命令行参数修改 */
    struct Settings {
        /** @brief 如果设置为 true，则启用验证层（并输出消息） */
        bool validation = false;
        /** @brief 如果通过命令行请求全屏模式，则设置为 true */
        bool fullscreen = false;
        /** @brief 如果交换链强制启用垂直同步（V-Sync），则设置为 true */
        bool vsync = false;
        /** @brief 启用 UI 叠加层 */
        bool overlay = true;
    } settings;

    /** @brief 游戏手柄输入状态（仅在 Android 上使用） */
    struct {
        glm::vec2 axisLeft = glm::vec2(0.0f);  // 左摇杆状态
        glm::vec2 axisRight = glm::vec2(0.0f); // 右摇杆状态
    } gamePadState;

    /** @brief 鼠标/触摸输入状态 */
    struct {
        struct {
            bool left = false;   // 左键是否按下
            bool right = false;  // 右键是否按下
            bool middle = false; // 中键是否按下
        } buttons;
        glm::vec2 position; // 鼠标/触摸位置
    } mouseState;

    VkClearColorValue defaultClearColor = {{0.025f, 0.025f, 0.025f, 1.0f}};

    static std::vector<const char *> args; // 命令行参数列表

    // 定义一个与帧率无关的计时器值，范围限制在 -1.0 到 1.0 之间
    // 用于动画、旋转等场景
    float timer = 0.0f;
    // 全局计时器的速度乘数，用于加速或减慢计时器
    float timerSpeed = 0.25f;
    bool paused = false; // 标记是否暂停

    Camera camera; // 摄像机对象，用于控制视图和投影

    std::string title = "Vulkan Example"; // 窗口标题
    std::string name = "vulkanExample";   // 示例名称
    uint32_t apiVersion = VK_API_VERSION_1_0; // 使用的 Vulkan API 版本

    /** @brief 默认渲染流程使用的深度模板附件 */
    struct {
        VkImage image;        // 深度模板图像
        VkDeviceMemory memory; // 图像内存
        VkImageView view;     // 图像视图
    } depthStencil{};

    // OS specific
#if defined(_WIN32)
    HWND window;
	HINSTANCE windowInstance;
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
    // true if application has focused, false if moved to background
    bool focused = false;
    struct TouchPos {
        int32_t x;
        int32_t y;
    } touchPos;
    bool touchDown = false;
    double touchTimer = 0.0;
    int64_t lastTapTime = 0;
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT))
    void* view;
#if defined(VK_USE_PLATFORM_METAL_EXT)
	CAMetalLayer* metalLayer;
#endif
#if defined(VK_EXAMPLE_XCODE_GENERATED)
	bool quit = false;
#endif
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
	bool quit = false;
	IDirectFB *dfb = nullptr;
	IDirectFBDisplayLayer *layer = nullptr;
	IDirectFBWindow *window = nullptr;
	IDirectFBSurface *surface = nullptr;
	IDirectFBEventBuffer *event_buffer = nullptr;
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	wl_display *display = nullptr;
	wl_registry *registry = nullptr;
	wl_compositor *compositor = nullptr;
	struct xdg_wm_base *shell = nullptr;
	wl_seat *seat = nullptr;
	wl_pointer *pointer = nullptr;
	wl_keyboard *keyboard = nullptr;
	wl_surface *surface = nullptr;
	struct xdg_surface *xdg_surface;
	struct xdg_toplevel *xdg_toplevel;
	bool quit = false;
	bool configured = false;

#elif defined(_DIRECT2DISPLAY)
	bool quit = false;
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	bool quit = false;
	xcb_connection_t *connection;
	xcb_screen_t *screen;
	xcb_window_t window;
	xcb_intern_atom_reply_t *atom_wm_delete_window;
#elif defined(VK_USE_PLATFORM_HEADLESS_EXT)
	bool quit = false;
#elif defined(VK_USE_PLATFORM_SCREEN_QNX)
	screen_context_t screen_context = nullptr;
	screen_window_t screen_window = nullptr;
	screen_event_t screen_event = nullptr;
	bool quit = false;
#endif

    /** @brief Default base class constructor */
    VulkanExampleBase();

    /** @brief 虚析构函数，用于释放资源 */
    virtual ~VulkanExampleBase();

    /** @brief 设置 Vulkan 实例，启用所需的扩展并连接到物理设备（GPU） */
    bool initVulkan();

#if defined(_WIN32)
    void setupConsole(std::string title);
	void setupDPIAwareness();
	HWND setupWindow(HINSTANCE hinstance, WNDPROC wndproc);
	void handleMessages(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
    static int32_t handleAppInput(struct android_app* app, AInputEvent* event);
    static void handleAppCommand(android_app* app, int32_t cmd);
#elif (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK) || defined(VK_USE_PLATFORM_METAL_EXT))
    void* setupWindow(void* view);
	void displayLinkOutputCb();
	void mouseDragged(float x, float y);
	void windowWillResize(float x, float y);
	void windowDidResize();
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
	IDirectFBSurface *setupWindow();
	void handleEvent(const DFBWindowEvent *event);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
	struct xdg_surface *setupWindow();
	void initWaylandConnection();
	void setSize(int width, int height);
	static void registryGlobalCb(void *data, struct wl_registry *registry,
			uint32_t name, const char *interface, uint32_t version);
	void registryGlobal(struct wl_registry *registry, uint32_t name,
			const char *interface, uint32_t version);
	static void registryGlobalRemoveCb(void *data, struct wl_registry *registry,
			uint32_t name);
	static void seatCapabilitiesCb(void *data, wl_seat *seat, uint32_t caps);
	void seatCapabilities(wl_seat *seat, uint32_t caps);
	static void pointerEnterCb(void *data, struct wl_pointer *pointer,
			uint32_t serial, struct wl_surface *surface, wl_fixed_t sx,
			wl_fixed_t sy);
	static void pointerLeaveCb(void *data, struct wl_pointer *pointer,
			uint32_t serial, struct wl_surface *surface);
	static void pointerMotionCb(void *data, struct wl_pointer *pointer,
			uint32_t time, wl_fixed_t sx, wl_fixed_t sy);
	void pointerMotion(struct wl_pointer *pointer,
			uint32_t time, wl_fixed_t sx, wl_fixed_t sy);
	static void pointerButtonCb(void *data, struct wl_pointer *wl_pointer,
			uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
	void pointerButton(struct wl_pointer *wl_pointer,
			uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
	static void pointerAxisCb(void *data, struct wl_pointer *wl_pointer,
			uint32_t time, uint32_t axis, wl_fixed_t value);
	void pointerAxis(struct wl_pointer *wl_pointer,
			uint32_t time, uint32_t axis, wl_fixed_t value);
	static void keyboardKeymapCb(void *data, struct wl_keyboard *keyboard,
			uint32_t format, int fd, uint32_t size);
	static void keyboardEnterCb(void *data, struct wl_keyboard *keyboard,
			uint32_t serial, struct wl_surface *surface, struct wl_array *keys);
	static void keyboardLeaveCb(void *data, struct wl_keyboard *keyboard,
			uint32_t serial, struct wl_surface *surface);
	static void keyboardKeyCb(void *data, struct wl_keyboard *keyboard,
			uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
	void keyboardKey(struct wl_keyboard *keyboard,
			uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
	static void keyboardModifiersCb(void *data, struct wl_keyboard *keyboard,
			uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched,
			uint32_t mods_locked, uint32_t group);

#elif defined(_DIRECT2DISPLAY)
//
#elif defined(VK_USE_PLATFORM_XCB_KHR)
	xcb_window_t setupWindow();
	void initxcbConnection();
	void handleEvent(const xcb_generic_event_t *event);
#elif defined(VK_USE_PLATFORM_SCREEN_QNX)
	void setupWindow();
	void handleEvent();
#else
	void setupWindow();
#endif

    /** @brief （虚函数）创建应用程序范围的 Vulkan 实例 */
    virtual VkResult createInstance();

    /** @brief （纯虚函数）渲染函数，需要由示例应用程序实现 */
    virtual void render() = 0;

    /** @brief （虚函数）在按键按下后调用，可用于自定义按键处理 */
    virtual void keyPressed(uint32_t);

    /** @brief （虚函数）在鼠标移动后调用，在内部事件（如摄像机旋转）处理之前执行 */
    virtual void mouseMoved(double x, double y, bool &handled);

    /** @brief （虚函数）在窗口大小调整后调用，示例应用程序可用来重新创建资源 */
    virtual void windowResized();

    /** @brief （虚函数）在资源重新创建后调用，需要重建命令缓冲区（例如帧缓冲区），由示例应用程序实现 */
    virtual void buildCommandBuffers();

    /** @brief （虚函数）设置默认的深度和模板视图 */
    virtual void setupDepthStencil();

    /** @brief （虚函数）为所有请求的交换链图像设置默认的帧缓冲区 */
    virtual void setupFrameBuffer();

    /** @brief （虚函数）设置默认的渲染流程 */
    virtual void setupRenderPass();

    /** @brief （虚函数）在读取物理设备特性后调用，可用于设置要在设备上启用的特性 */
    virtual void getEnabledFeatures();

    /** @brief （虚函数）在读取物理设备扩展后调用，可根据支持的扩展列表启用扩展 */
    virtual void getEnabledExtensions();

    /** @brief 准备运行示例所需的所有 Vulkan 资源和功能 */
    virtual void prepare();

    /** @brief 加载指定着色器阶段的 SPIR-V 着色器文件 */
    VkPipelineShaderStageCreateInfo loadShader(const std::string& fileName, VkShaderStageFlagBits stage);

    /** @brief 处理窗口大小调整 */
    void windowResize();

    /** @brief 主渲染循环的入口点 */
    void renderLoop();

    /** @brief 将 ImGui 叠加层的绘制命令添加到指定的命令缓冲区 */
    void drawUI(VkCommandBuffer commandBuffer);

    /** @brief 通过获取下一个交换链图像，为提交工作负载准备下一帧 */
    void prepareFrame();

    /** @brief 将当前图像提交到交换链 */
    void submitFrame();

    /** @brief （虚函数）默认的图像获取、提交和命令缓冲区提交函数 */
    virtual void renderFrame();

    /** @brief （虚函数）在 UI 叠加层更新时调用，可用于向叠加层添加自定义元素 */
    virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay);

#if defined(_WIN32)
    virtual void OnHandleMessage(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
#endif
};

#include "Entrypoints.h"