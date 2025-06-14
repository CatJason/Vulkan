/*
* Vulkan 示例 - 全屏径向模糊（单次离屏渲染后处理效果）
*
* 这个示例展示了如何实现一个简单的后处理效果
*
* 版权所有 (C) 2016-2024 Sascha Willems - www.saschawillems.de
*
* 代码遵循 MIT 许可证 (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include "VulkanglTFModel.h"

class VulkanExample : public VulkanExampleBase {
public:
    bool blur = true;               // 是否启用模糊效果
    bool displayTexture = false;    // 是否只显示渲染目标纹理

    vks::Texture2D gradientTexture; // 渐变纹理
    vkglTF::Model scene;            // 3D场景模型

    // 场景渲染的统一缓冲区数据
    struct UniformDataScene {
        glm::mat4 projection;       // 投影矩阵
        glm::mat4 modelView;        // 模型视图矩阵
        float gradientPos = 0.0f;   // 渐变位置（用于动画效果）
    } uniformDataScene;

    // 模糊参数的统一缓冲区数据
    struct UniformDataBlurParams {
        float radialBlurScale = 0.35f;      // 模糊缩放因子
        float radialBlurStrength = 0.75f;   // 模糊强度
        glm::vec2 radialOrigin = glm::vec2(0.5f, 0.5f); // 模糊原点（屏幕空间坐标）
    } uniformDataBlurParams;

    // 统一缓冲区对象
    struct {
        vks::Buffer scene;          // 场景渲染的缓冲区
        vks::Buffer blurParams;     // 模糊参数的缓冲区
    } uniformBuffers;

    // 管线对象
    struct {
        VkPipeline radialBlur{VK_NULL_HANDLE};        // 径向模糊管线
        VkPipeline colorPass{VK_NULL_HANDLE};         // 颜色渲染管线（离屏）
        VkPipeline phongPass{VK_NULL_HANDLE};         // Phong光照管线
        VkPipeline offscreenDisplay{VK_NULL_HANDLE};   // 离屏纹理显示管线
    } pipelines;

    // 管线布局
    struct {
        VkPipelineLayout radialBlur{VK_NULL_HANDLE};  // 径向模糊管线布局
        VkPipelineLayout scene{VK_NULL_HANDLE};       // 场景渲染管线布局
    } pipelineLayouts;

    // 描述符集
    struct {
        VkDescriptorSet scene{VK_NULL_HANDLE};        // 场景渲染描述符集
        VkDescriptorSet radialBlur{VK_NULL_HANDLE};   // 径向模糊描述符集
    } descriptorSets;

    // 描述符集布局
    struct {
        VkDescriptorSetLayout scene{VK_NULL_HANDLE};      // 场景渲染描述符集布局
        VkDescriptorSetLayout radialBlur{VK_NULL_HANDLE}; // 径向模糊描述符集布局
    } descriptorSetLayouts;

    // 离屏渲染的帧缓冲区附件
    struct FrameBufferAttachment {
        VkImage image;              // 图像对象
        VkDeviceMemory mem;         // 设备内存
        VkImageView view;           // 图像视图
    };

    // 离屏渲染通道
    struct OffscreenPass {
        int32_t width, height;      // 离屏图像尺寸
        VkFramebuffer frameBuffer;   // 帧缓冲区
        FrameBufferAttachment color, depth; // 颜色和深度附件
        VkRenderPass renderPass;     // 渲染通道
        VkSampler sampler;           // 采样器
        VkDescriptorImageInfo descriptor; // 描述符信息
    } offscreenPass{};

    // 离屏图像尺寸
    const uint32_t offscreenImageSize{512};
    // 离屏图像格式（8位RGBA）
    const VkFormat offscreenImageFormat{VK_FORMAT_R8G8B8A8_UNORM};

    VulkanExample() : VulkanExampleBase() {
        title = "全屏径向模糊效果";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -17.5f));
        camera.setRotation(glm::vec3(-16.25f, -28.75f, 0.0f));
        camera.setPerspective(45.0f, (float) width / (float) height, 1.0f, 256.0f);
        timerSpeed *= 0.5f;  // 减慢动画速度
    }

    ~VulkanExample() {
        // 清理所有Vulkan资源
        if (device) {
            // 清理离屏渲染相关资源
            vkDestroyImageView(device, offscreenPass.color.view, nullptr);
            vkDestroyImage(device, offscreenPass.color.image, nullptr);
            vkFreeMemory(device, offscreenPass.color.mem, nullptr);

            vkDestroyImageView(device, offscreenPass.depth.view, nullptr);
            vkDestroyImage(device, offscreenPass.depth.image, nullptr);
            vkFreeMemory(device, offscreenPass.depth.mem, nullptr);

            vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);
            vkDestroySampler(device, offscreenPass.sampler, nullptr);
            vkDestroyFramebuffer(device, offscreenPass.frameBuffer, nullptr);

            // 清理管线
            vkDestroyPipeline(device, pipelines.radialBlur, nullptr);
            vkDestroyPipeline(device, pipelines.phongPass, nullptr);
            vkDestroyPipeline(device, pipelines.colorPass, nullptr);
            vkDestroyPipeline(device, pipelines.offscreenDisplay, nullptr);

            // 清理管线布局
            vkDestroyPipelineLayout(device, pipelineLayouts.radialBlur, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);

            // 清理描述符集布局
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.radialBlur, nullptr);

            // 清理缓冲区
            uniformBuffers.scene.destroy();
            uniformBuffers.blurParams.destroy();

            // 清理纹理
            gradientTexture.destroy();
        }
    }

    // 准备离屏渲染
    void prepareOffscreen() {
        offscreenPass.width = offscreenImageSize;
        offscreenPass.height = offscreenImageSize;

        // 寻找合适的深度格式
        VkFormat fbDepthFormat;
        VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice,
                                                                        &fbDepthFormat);
        assert(validDepthFormat);

        // 创建颜色附件
        VkImageCreateInfo image = vks::initializers::imageCreateInfo();
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = offscreenImageFormat;
        image.extent.width = offscreenPass.width;
        image.extent.height = offscreenPass.height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        // 图像将用作颜色附件和采样器
        image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
        VkMemoryRequirements memReqs;

        VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &offscreenPass.color.image));
        vkGetImageMemoryRequirements(device, offscreenPass.color.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreenPass.color.mem));
        VK_CHECK_RESULT(
                vkBindImageMemory(device, offscreenPass.color.image, offscreenPass.color.mem, 0));

        // 创建颜色附件的图像视图
        VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
        colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorImageView.format = offscreenImageFormat;
        colorImageView.subresourceRange = {};
        colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorImageView.subresourceRange.baseMipLevel = 0;
        colorImageView.subresourceRange.levelCount = 1;
        colorImageView.subresourceRange.baseArrayLayer = 0;
        colorImageView.subresourceRange.layerCount = 1;
        colorImageView.image = offscreenPass.color.image;
        VK_CHECK_RESULT(
                vkCreateImageView(device, &colorImageView, nullptr, &offscreenPass.color.view));

        // 创建采样器用于片段着色器中采样附件
        VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = samplerInfo.addressModeU;
        samplerInfo.addressModeW = samplerInfo.addressModeU;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &offscreenPass.sampler));

        // 创建深度附件
        image.format = fbDepthFormat;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &offscreenPass.depth.image));
        vkGetImageMemoryRequirements(device, offscreenPass.depth.image, &memReqs);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits,
                                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &offscreenPass.depth.mem));
        VK_CHECK_RESULT(
                vkBindImageMemory(device, offscreenPass.depth.image, offscreenPass.depth.mem, 0));

        // 创建深度附件的图像视图
        VkImageViewCreateInfo depthStencilView = vks::initializers::imageViewCreateInfo();
        depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthStencilView.format = fbDepthFormat;
        depthStencilView.flags = 0;
        depthStencilView.subresourceRange = {};
        depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (fbDepthFormat >= VK_FORMAT_D16_UNORM_S8_UINT)
            depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        depthStencilView.subresourceRange.baseMipLevel = 0;
        depthStencilView.subresourceRange.levelCount = 1;
        depthStencilView.subresourceRange.baseArrayLayer = 0;
        depthStencilView.subresourceRange.layerCount = 1;
        depthStencilView.image = offscreenPass.depth.image;
        VK_CHECK_RESULT(
                vkCreateImageView(device, &depthStencilView, nullptr, &offscreenPass.depth.view));

        // 为离屏渲染创建单独的渲染通道
        std::array<VkAttachmentDescription, 2> attchmentDescriptions = {};
        // 颜色附件描述
        attchmentDescriptions[0].format = offscreenImageFormat;
        attchmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attchmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attchmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attchmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attchmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attchmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attchmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // 深度附件描述
        attchmentDescriptions[1].format = fbDepthFormat;
        attchmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attchmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attchmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attchmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attchmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attchmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attchmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorReference = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthReference = {1,
                                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;
        subpassDescription.pDepthStencilAttachment = &depthReference;

        // 使用子通道依赖进行布局转换
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // 创建实际的渲染通道
        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
        renderPassInfo.pAttachments = attchmentDescriptions.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        VK_CHECK_RESULT(
                vkCreateRenderPass(device, &renderPassInfo, nullptr, &offscreenPass.renderPass));

        // 创建帧缓冲区
        VkImageView attachments[2];
        attachments[0] = offscreenPass.color.view;
        attachments[1] = offscreenPass.depth.view;

        VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
        fbufCreateInfo.renderPass = offscreenPass.renderPass;
        fbufCreateInfo.attachmentCount = 2;
        fbufCreateInfo.pAttachments = attachments;
        fbufCreateInfo.width = offscreenPass.width;
        fbufCreateInfo.height = offscreenPass.height;
        fbufCreateInfo.layers = 1;

        VK_CHECK_RESULT(
                vkCreateFramebuffer(device, &fbufCreateInfo, nullptr, &offscreenPass.frameBuffer));

        // 填充描述符信息供后续在描述符集中使用
        offscreenPass.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        offscreenPass.descriptor.imageView = offscreenPass.color.view;
        offscreenPass.descriptor.sampler = offscreenPass.sampler;
    }

    // 构建命令缓冲区
    void buildCommandBuffers() {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        VkViewport viewport;
        VkRect2D scissor;

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            /*
                第一渲染通道：离屏渲染
            */
            {
                clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
                clearValues[1].depthStencil = {1.0f, 0};

                VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = offscreenPass.renderPass;
                renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
                renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
                renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
                renderPassBeginInfo.clearValueCount = 2;
                renderPassBeginInfo.pClearValues = clearValues;

                viewport = vks::initializers::viewport((float) offscreenPass.width,
                                                       (float) offscreenPass.height, 0.0f, 1.0f);
                vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);
                scissor = vks::initializers::rect2D(offscreenPass.width, offscreenPass.height, 0,
                                                    0);
                vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);
                vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo,
                                     VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0,
                                        NULL);
                vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.colorPass);
                scene.draw(drawCmdBuffers[i]);
                vkCmdEndRenderPass(drawCmdBuffers[i]);
            }

            /*
                注意：渲染通道之间不需要显式同步，因为这是通过子通道依赖隐式完成的
            */

            /*
                第二渲染通道：应用径向模糊的场景渲染
            */
            {
                clearValues[0].color = defaultClearColor;
                clearValues[1].depthStencil = {1.0f, 0};

                VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = renderPass;
                renderPassBeginInfo.framebuffer = frameBuffers[i];
                renderPassBeginInfo.renderArea.extent.width = width;
                renderPassBeginInfo.renderArea.extent.height = height;
                renderPassBeginInfo.clearValueCount = 2;
                renderPassBeginInfo.pClearValues = clearValues;

                vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo,
                                     VK_SUBPASS_CONTENTS_INLINE);

                viewport = vks::initializers::viewport((float) width, (float) height, 0.0f, 1.0f);
                vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

                scissor = vks::initializers::rect2D(width, height, 0, 0);
                vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

                // 3D场景
                vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0,
                                        NULL);
                vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  pipelines.phongPass);
                scene.draw(drawCmdBuffers[i]);

                // 全屏三角形（裁剪为四边形）应用径向模糊
                if (blur) {
                    vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            pipelineLayouts.radialBlur, 0, 1,
                                            &descriptorSets.radialBlur, 0, NULL);
                    vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      (displayTexture) ? pipelines.offscreenDisplay
                                                       : pipelines.radialBlur);
                    vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
                }

                drawUI(drawCmdBuffers[i]);

                vkCmdEndRenderPass(drawCmdBuffers[i]);
            }

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    // 加载资源
    void loadAssets() {
        // 加载3D模型和渐变纹理
        scene.loadFromFile(getAssetPath() + "models/glowsphere.gltf", vulkanDevice, queue,
                           vkglTF::FileLoadingFlags::PreTransformVertices |
                           vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
                           vkglTF::FileLoadingFlags::FlipY);
        gradientTexture.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx",
                                     VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
    }

    // 设置描述符
    void setupDescriptors() {
        // 描述符池
        std::vector<VkDescriptorPoolSize> poolSizes = {
                vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4),
                vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6)
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(
                poolSizes, 2);
        VK_CHECK_RESULT(
                vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // 描述符集布局
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
        VkDescriptorSetLayoutCreateInfo descriptorLayout;

        // 场景渲染描述符集布局
        setLayoutBindings = {
                // 绑定0：顶点着色器统一缓冲区
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                              VK_SHADER_STAGE_VERTEX_BIT, 0),
                // 绑定1：片段着色器图像采样器
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
                // 绑定2：片段着色器统一缓冲区
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                              VK_SHADER_STAGE_FRAGMENT_BIT, 2)
        };
        descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                                    &descriptorSetLayouts.scene));

        // 全屏径向模糊描述符集布局
        setLayoutBindings = {
                // 绑定0：片段着色器统一缓冲区
                vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                              VK_SHADER_STAGE_FRAGMENT_BIT, 0),
                // 绑定1：片段着色器图像采样器
                vks::initializers::descriptorSetLayoutBinding(
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1)
        };
        descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
                setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr,
                                                    &descriptorSetLayouts.radialBlur));

        // 分配描述符集
        VkDescriptorSetAllocateInfo descriptorSetAllocInfo;

        // 场景渲染描述符集
        descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool,
                                                                              &descriptorSetLayouts.scene,
                                                                              1);
        VK_CHECK_RESULT(
                vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSets.scene));

        std::vector<VkWriteDescriptorSet> offScreenWriteDescriptorSets = {
                // 绑定0：顶点着色器统一缓冲区
                vks::initializers::writeDescriptorSet(descriptorSets.scene,
                                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
                                                      &uniformBuffers.scene.descriptor),
                // 绑定1：颜色渐变采样器
                vks::initializers::writeDescriptorSet(descriptorSets.scene,
                                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                      &gradientTexture.descriptor),
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(offScreenWriteDescriptorSets.size()),
                               offScreenWriteDescriptorSets.data(), 0, nullptr);

        // 全屏径向模糊描述符集
        descriptorSetAllocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool,
                                                                              &descriptorSetLayouts.radialBlur,
                                                                              1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocInfo,
                                                 &descriptorSets.radialBlur));

        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                // 绑定0：片段着色器统一缓冲区
                vks::initializers::writeDescriptorSet(descriptorSets.radialBlur,
                                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
                                                      &uniformBuffers.blurParams.descriptor),
                // 绑定1：片段着色器纹理采样器
                vks::initializers::writeDescriptorSet(descriptorSets.radialBlur,
                                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                      &offscreenPass.descriptor),
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()),
                               writeDescriptorSets.data(), 0, nullptr);
    }

    // 准备管线
    void preparePipelines() {
        // 管线布局
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
                &descriptorSetLayouts.scene, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,
                                               &pipelineLayouts.scene));

        pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(
                &descriptorSetLayouts.radialBlur, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr,
                                               &pipelineLayouts.radialBlur));

        // 管线状态
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(
                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(
                VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(
                0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(
                1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(
                VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(
                1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(
                VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> dynamicStateEnables = {VK_DYNAMIC_STATE_VIEWPORT,
                                                           VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(
                dynamicStateEnables);
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        VkGraphicsPipelineCreateInfo pipelineCI = vks::initializers::pipelineCreateInfo(
                pipelineLayouts.radialBlur, renderPass, 0);
        pipelineCI.pInputAssemblyState = &inputAssemblyStateCI;
        pipelineCI.pRasterizationState = &rasterizationStateCI;
        pipelineCI.pColorBlendState = &colorBlendStateCI;
        pipelineCI.pMultisampleState = &multisampleStateCI;
        pipelineCI.pViewportState = &viewportStateCI;
        pipelineCI.pDepthStencilState = &depthStencilStateCI;
        pipelineCI.pDynamicState = &dynamicStateCI;
        pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCI.pStages = shaderStages.data();

        // 径向模糊管线
        shaderStages[0] = loadShader(getShadersPath() + "radialblur/radialblur.vert.spv",
                                     VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "radialblur/radialblur.frag.spv",
                                     VK_SHADER_STAGE_FRAGMENT_BIT);
        // 空顶点输入状态（顶点着色器生成全屏三角形）
        VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        pipelineCI.pVertexInputState = &emptyInputState;
        pipelineCI.layout = pipelineLayouts.radialBlur;
        // 加法混合
        blendAttachmentState.colorWriteMask = 0xF;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr,
                                                  &pipelines.radialBlur));

        // 无混合（用于调试显示）
        blendAttachmentState.blendEnable = VK_FALSE;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr,
                                                  &pipelines.offscreenDisplay));

        // Phong光照管线
        pipelineCI.layout = pipelineLayouts.scene;
        shaderStages[0] = loadShader(getShadersPath() + "radialblur/phongpass.vert.spv",
                                     VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "radialblur/phongpass.frag.spv",
                                     VK_SHADER_STAGE_FRAGMENT_BIT);
        blendAttachmentState.blendEnable = VK_FALSE;
        depthStencilStateCI.depthWriteEnable = VK_TRUE;
        pipelineCI.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState(
                {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV,
                 vkglTF::VertexComponent::Color, vkglTF::VertexComponent::Normal});;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr,
                                                  &pipelines.phongPass));

        // 仅颜色渲染管线（离屏模糊基础）
        shaderStages[0] = loadShader(getShadersPath() + "radialblur/colorpass.vert.spv",
                                     VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "radialblur/colorpass.frag.spv",
                                     VK_SHADER_STAGE_FRAGMENT_BIT);
        pipelineCI.renderPass = offscreenPass.renderPass;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr,
                                                  &pipelines.colorPass));
    }

    // 准备统一缓冲区
    void prepareUniformBuffers() {
        // Phong and color pass vertex shader uniform buffer
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                   &uniformBuffers.scene,
                                                   sizeof(UniformDataScene)));
        // Fullscreen radial blur parameters
        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                   &uniformBuffers.blurParams,
                                                   sizeof(UniformDataBlurParams),
                                                   &uniformDataBlurParams));
        // Map persistent
        VK_CHECK_RESULT(uniformBuffers.scene.map());
        VK_CHECK_RESULT(uniformBuffers.blurParams.map());
    }

// 更新径向模糊通道的参数
// 注意：这里仅执行数据拷贝操作，实际参数值是通过UI界面设置的
    void updateUniformBuffersBlurParams() {
        // 将CPU端的模糊参数结构体数据拷贝到GPU端的统一缓冲区
        memcpy(uniformBuffers.blurParams.mapped, &uniformDataBlurParams,
               sizeof(UniformDataBlurParams));
    }

// 更新用于渲染3D场景的统一缓冲区
    void updateUniformBuffers() {
        // 设置初始透视投影矩阵（45度视野，基于窗口宽高比，近平面1.0f，远平面256.0f）
        uniformDataScene.projection = glm::perspective(glm::radians(45.0f),
                                                       (float) width / (float) height, 1.0f,
                                                       256.0f);

        // 更新相机旋转（每帧绕Y轴旋转10度）
        camera.setRotation(camera.rotation + glm::vec3(0.0f, frameTimer * 10.0f, 0.0f));

        // 使用相机生成的投影和视图矩阵
        uniformDataScene.projection = camera.matrices.perspective;
        uniformDataScene.modelView = camera.matrices.view;

        // 如果未暂停，为后处理效果添加动画：通过改变颜色渐变位置
        if (!paused) {
            uniformDataScene.gradientPos += frameTimer * 0.1f; // 每帧移动渐变位置
        }

        // 将场景数据拷贝到GPU统一缓冲区
        memcpy(uniformBuffers.scene.mapped, &uniformDataScene, sizeof(UniformDataScene));
    }

// 准备函数：初始化所有Vulkan资源
    void prepare() {
        VulkanExampleBase::prepare();  // 调用基类准备函数
        loadAssets();                  // 加载3D模型和纹理资源
        prepareOffscreen();            // 设置离屏渲染缓冲区
        prepareUniformBuffers();       // 创建并映射统一缓冲区
        setupDescriptors();            // 设置描述符和描述符集
        preparePipelines();            // 创建图形管线
        buildCommandBuffers();         // 构建命令缓冲区
        prepared = true;               // 标记准备完成
    }

// 绘制函数：提交命令缓冲区进行渲染
    void draw() {
        VulkanExampleBase::prepareFrame();  // 准备帧（获取交换链图像等）

        // 设置提交信息：使用当前缓冲区的命令缓冲区
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

        // 提交命令缓冲区到图形队列
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

        VulkanExampleBase::submitFrame();  // 提交帧（呈现交换链图像）
    }

// 渲染循环主函数
    virtual void render() {
        if (!prepared) return;  // 如果未初始化完成则直接返回

        updateUniformBuffers();  // 更新统一缓冲区数据
        draw();                  // 执行绘制
    }

// UI界面更新函数
    virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay) {
        if (overlay->header("Settings")) {  // "设置"标题栏
            // 径向模糊开关复选框
            if (overlay->checkBox("Radial blur", &blur)) {
                buildCommandBuffers();  // 如果值改变，重建命令缓冲区
            }

            // 仅显示渲染目标复选框
            if (overlay->checkBox("Display render target only", &displayTexture)) {
                buildCommandBuffers();  // 如果值改变，重建命令缓冲区
            }

            // 如果启用了径向模糊，显示参数控制
            if (blur) {
                if (overlay->header("Blur parameters")) {  // "模糊参数"标题栏
                    bool updateParams = false;  // 标记参数是否被修改

                    // 模糊比例滑块（0.1到1.0）
                    updateParams |= overlay->sliderFloat("Scale",
                                                         &uniformDataBlurParams.radialBlurScale,
                                                         0.1f, 1.0f);
                    // 模糊强度滑块（0.1到2.0）
                    updateParams |= overlay->sliderFloat("Strength",
                                                         &uniformDataBlurParams.radialBlurStrength,
                                                         0.1f, 2.0f);
                    // 水平原点滑块（0.0到1.0）
                    updateParams |= overlay->sliderFloat("Horiz. origin",
                                                         &uniformDataBlurParams.radialOrigin.x,
                                                         0.0f, 1.0f);
                    // 垂直原点滑块（0.0到1.0）
                    updateParams |= overlay->sliderFloat("Vert. origin",
                                                         &uniformDataBlurParams.radialOrigin.y,
                                                         0.0f, 1.0f);

                    // 如果有参数被修改，更新GPU缓冲区
                    if (updateParams) {
                        updateUniformBuffersBlurParams();
                    }
                }
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
