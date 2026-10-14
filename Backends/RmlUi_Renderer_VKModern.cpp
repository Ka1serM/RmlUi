#include "RmlUi_Renderer_VKModern.h"
#include "RmlUi_VKModern/ShadersCompiledSPV.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Math.h>
#include <RmlUi/Core/SystemInterface.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// Include the implementation of the bundled glad Vulkan loader and VulkanMemoryAllocator in this translation unit. The
// first include pulled in the declarations, these re-includes expand only the implementation blocks.
#define GLAD_VULKAN_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#if defined _MSC_VER
	#pragma warning(push, 0)
#elif defined __clang__
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wall"
	#pragma clang diagnostic ignored "-Wextra"
	#pragma clang diagnostic ignored "-Wnullability-extension"
	#pragma clang diagnostic ignored "-Wnullability-completeness"
#elif defined __GNUC__
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wall"
	#pragma GCC diagnostic ignored "-Wextra"
	#pragma GCC diagnostic ignored "-Wpedantic"
	#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
	#pragma GCC diagnostic ignored "-Wunused-parameter"
	#pragma GCC diagnostic ignored "-Wunused-variable"
	#pragma GCC diagnostic ignored "-Wunused-function"
	#pragma GCC diagnostic ignored "-Wparentheses"
	#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#include "RmlUi_VKModern/vulkan.h"
#include "RmlUi_Vulkan/vk_mem_alloc.h"
#if defined _MSC_VER
	#pragma warning(pop)
#elif defined __clang__
	#pragma clang diagnostic pop
#elif defined __GNUC__
	#pragma GCC diagnostic pop
#endif

static constexpr uint32_t MaxTextures = 1 << 16;
static constexpr uint32_t PushConstantsSize = 128;
static constexpr uint32_t GeometryBlockVertices = 1 << 17;
static constexpr uint32_t GeometryBlockIndices = 1 << 18;
static constexpr VkDeviceSize StagingBufferSize = 4 << 20;
static constexpr VkDeviceSize StagingAlignment = 16;
// Render targets are allocated in steps of this size, so that resizing the window rarely reallocates them.
static constexpr int RenderTargetGranularity = 256;
static constexpr uint32_t FlagStraightAlpha = 1;
static constexpr VkImageUsageFlags PostprocessUsage =
	VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
static constexpr bool RowMajorMatrices = std::is_same<Rml::Matrix4f, Rml::RowMajorMatrix4f>::value;

#define MAX_NUM_STOPS 16
#define BLUR_SIZE 7
#define BLUR_NUM_WEIGHTS ((BLUR_SIZE + 1) / 2)

enum class ShaderGradientFunction { Linear, Radial, Conic, RepeatingLinear, RepeatingRadial, RepeatingConic }; // Must match shader definitions.

enum class FilterType { Invalid = 0, Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };
struct CompiledFilter {
	FilterType type;

	// Passthrough
	float blend_factor;

	// Blur
	float sigma;

	// Drop shadow
	Rml::Vector2f offset;
	Rml::ColourbPremultiplied color;

	// ColorMatrix
	Rml::Matrix4f color_matrix;
};

enum class CompiledShaderType { Invalid = 0, Gradient, Creation };
struct CompiledShader {
	CompiledShaderType type;

	// Gradient
	ShaderGradientFunction gradient_function;
	Rml::Vector2f p;
	Rml::Vector2f v;
	Rml::Vector<float> stop_positions;
	Rml::Vector<Rml::Colourf> stop_colors;

	// Shader
	Rml::Vector2f dimensions;
};

// Push constant offsets, see the shaders. Geometry pushes the transform and translation, followed by any constants of
// its fragment shader. Quads push their rectangles and textures, followed by any constants of their fragment shader.
static constexpr uint32_t TranslationOffset = 64;
static constexpr uint32_t GeometryExtraOffset = 72;
static constexpr uint32_t QuadExtraOffset = 48;

struct BlurConstants {
	Rml::Vector2f texel_offset;
	Rml::Vector2f tex_coord_min;
	Rml::Vector2f tex_coord_max;
	float padding[2];
	float weights[BLUR_NUM_WEIGHTS];
};

struct DropShadowConstants {
	Rml::Vector2f tex_coord_min;
	Rml::Vector2f tex_coord_max;
	Rml::Colourf color;
};

struct CreationConstants {
	Rml::Vector2f dimensions;
	float time;
};

// std430 layout of the gradient buffer in shader_frag_gradient.frag.
struct GradientData {
	int func;
	int num_stops;
	Rml::Vector2f p;
	Rml::Vector2f v;
	float padding[2];
	Rml::Colourf stop_colors[MAX_NUM_STOPS];
	float stop_positions[MAX_NUM_STOPS];
};

static_assert(QuadExtraOffset + sizeof(BlurConstants) <= PushConstantsSize, "Push constants exceed the guaranteed size.");
static_assert(QuadExtraOffset + sizeof(Rml::Matrix4f) <= PushConstantsSize, "Push constants exceed the guaranteed size.");
static_assert(sizeof(GradientData) == 352, "Must match the std430 layout in shader_frag_gradient.frag.");

// Errors after initialization, such as a lost device or running out of memory, are not recoverable.
static void Check(VkResult result, const char* operation)
{
	if (result == VK_SUCCESS)
		return;
	Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: %s failed with error %d.", operation, int(result));
	std::abort();
}

// AlignUp(314, 256) = 512
static VkDeviceSize AlignUp(VkDeviceSize val, VkDeviceSize alignment)
{
	return (val + alignment - 1) & ~(alignment - 1);
}

static Rml::Colourf ConvertToColorf(Rml::ColourbPremultiplied c0)
{
	Rml::Colourf result;
	for (int i = 0; i < 4; i++)
		result[i] = (1.f / 255.f) * float(c0[i]);
	return result;
}

static void SigmaToParameters(const float desired_sigma, int& out_pass_level, float& out_sigma)
{
	constexpr int max_num_passes = 10;
	static_assert(max_num_passes < 31, "");
	constexpr float max_single_pass_sigma = 3.0f;
	out_pass_level = Rml::Math::Clamp(Rml::Math::Log2(int(desired_sigma * (2.f / max_single_pass_sigma))), 0, max_num_passes);
	out_sigma = Rml::Math::Clamp(desired_sigma / float(1 << out_pass_level), 0.0f, max_single_pass_sigma);
}

static void SetBlurWeights(float (&weights)[BLUR_NUM_WEIGHTS], float sigma)
{
	float normalization = 0.0f;
	for (int i = 0; i < BLUR_NUM_WEIGHTS; i++)
	{
		if (Rml::Math::Absolute(sigma) < 0.1f)
			weights[i] = float(i == 0);
		else
			weights[i] = Rml::Math::Exp(-float(i * i) / (2.0f * sigma * sigma)) / (Rml::Math::SquareRoot(2.f * Rml::Math::RMLUI_PI) * sigma);

		normalization += (i == 0 ? 1.f : 2.0f) * weights[i];
	}
	for (int i = 0; i < BLUR_NUM_WEIGHTS; i++)
		weights[i] /= normalization;
}

static void SetTexCoordLimits(Rml::Vector2f& tex_coord_min, Rml::Vector2f& tex_coord_max, Rml::Rectanglei rectangle, Rml::Vector2i framebuffer_size)
{
	// Offset by half-texel values so that texture lookups are clamped to fragment centers, thereby avoiding color
	// bleeding from neighboring texels due to bilinear interpolation.
	tex_coord_min = (Rml::Vector2f(rectangle.p0) + Rml::Vector2f(0.5f)) / Rml::Vector2f(framebuffer_size);
	tex_coord_max = (Rml::Vector2f(rectangle.p1) - Rml::Vector2f(0.5f)) / Rml::Vector2f(framebuffer_size);
}

// A matrix in the column-major order of GLSL.
struct ShaderMatrix {
	float values[16];
};

static ShaderMatrix ToShaderMatrix(const Rml::Matrix4f& matrix)
{
	ShaderMatrix result;
	const float* data = matrix.data();
	for (int i = 0; i < 16; i++)
		result.values[i] = RowMajorMatrices ? data[(i % 4) * 4 + i / 4] : data[i];
	return result;
}

static VkRect2D ToRect(Rml::Rectanglei region)
{
	return VkRect2D{{region.Left(), region.Top()}, {uint32_t(region.Width()), uint32_t(region.Height())}};
}

static VkImageAspectFlags AspectOf(VkFormat format)
{
	switch (format)
	{
	case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
	case VK_FORMAT_D16_UNORM_S8_UINT:
	case VK_FORMAT_D24_UNORM_S8_UINT:
	case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
	default: return VK_IMAGE_ASPECT_COLOR_BIT;
	}
}

// All images stay in the general layout, so a single memory barrier is enough to order the passes of a frame: attachment
// writes, transfers, and sampling.
static void FullBarrier(VkCommandBuffer commands)
{
	constexpr VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
		VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;

	VkMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
	barrier.srcStageMask = stages;
	barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
	barrier.dstStageMask = stages;
	barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
		VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(commands, &dependency);
}

static void ImageBarrier(VkCommandBuffer commands, VkImage image, VkImageAspectFlags aspect, VkImageLayout old_layout, VkImageLayout new_layout,
	VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
	VkImageMemoryBarrier2 barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	barrier.srcStageMask = src_stage;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask = dst_stage;
	barrier.dstAccessMask = dst_access;
	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, 1};

	VkDependencyInfo dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers = &barrier;
	vkCmdPipelineBarrier2(commands, &dependency);
}

// Copies a region of a layer into a single-sampled image, resolving it if the layer is multisampled.
static void ResolveOrCopy(VkCommandBuffer commands, VkImage source, VkImage destination, bool multisampled, Rml::Rectanglei region)
{
	const VkImageSubresourceLayers subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	const VkOffset3D offset = {region.Left(), region.Top(), 0};
	const VkExtent3D extent = {uint32_t(region.Width()), uint32_t(region.Height()), 1};
	if (multisampled)
	{
		const VkImageResolve resolve = {subresource, offset, subresource, offset, extent};
		vkCmdResolveImage(commands, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &resolve);
		return;
	}
	const VkImageCopy copy = {subresource, offset, subresource, offset, extent};
	vkCmdCopyImage(commands, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
}

// Set to byte packing, or the compiler will expand our struct, which means it won't read correctly from file
#pragma pack(1)
struct TGAHeader {
	char idLength;
	char colourMapType;
	char dataType;
	short int colourMapOrigin;
	short int colourMapLength;
	char colourMapDepth;
	short int xOrigin;
	short int yOrigin;
	short int width;
	short int height;
	char bitsPerPixel;
	char imageDescriptor;
};
// Restore packing
#pragma pack()

RenderInterface_VKModern::RenderInterface_VKModern() = default;
RenderInterface_VKModern::~RenderInterface_VKModern() = default;

bool RenderInterface_VKModern::Initialize(Rml::Vector<const char*> required_extensions, CreateSurfaceCallback create_surface_callback,
	Rml::Vector<const char*> optional_device_extensions)
{
	if (!gladLoaderLoadVulkan(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The Vulkan library could not be loaded.");
		return false;
	}
	if (!CreateInstance(std::move(required_extensions)))
		return false;
	if (!create_surface_callback(m_instance, &m_surface))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The window surface could not be created.");
		return false;
	}
	if (!ChoosePhysicalDevice() || !CreateDevice(optional_device_extensions) || !ChooseFormats())
		return false;

	CreateDescriptors();
	CreatePipelines();
	CreateFrames();
	return true;
}

bool RenderInterface_VKModern::CreateInstance(Rml::Vector<const char*> required_extensions)
{
	uint32_t version = 0;
	if (!vkEnumerateInstanceVersion || vkEnumerateInstanceVersion(&version) != VK_SUCCESS || version < VK_API_VERSION_1_3)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The Vulkan loader does not support Vulkan 1.3.");
		return false;
	}

	uint32_t count = 0;
	Check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr), "vkEnumerateInstanceExtensionProperties");
	Rml::Vector<VkExtensionProperties> available(count);
	Check(vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data()), "vkEnumerateInstanceExtensionProperties");

	// Required to list devices that implement Vulkan on top of another API, such as MoltenVK.
	VkInstanceCreateFlags flags = 0;
	for (const VkExtensionProperties& extension : available)
	{
		if (std::strcmp(extension.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != 0)
			continue;
		required_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}

	VkApplicationInfo application = {};
	application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	application.pEngineName = "RmlUi";
	application.apiVersion = VK_API_VERSION_1_3;

	VkInstanceCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	info.flags = flags;
	info.pApplicationInfo = &application;
	info.enabledExtensionCount = uint32_t(required_extensions.size());
	info.ppEnabledExtensionNames = required_extensions.data();
	if (vkCreateInstance(&info, nullptr, &m_instance) != VK_SUCCESS)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The Vulkan instance could not be created.");
		return false;
	}

	gladLoaderLoadVulkan(m_instance, VK_NULL_HANDLE, VK_NULL_HANDLE);
	return true;
}

bool RenderInterface_VKModern::ChoosePhysicalDevice()
{
	uint32_t count = 0;
	Check(vkEnumeratePhysicalDevices(m_instance, &count, nullptr), "vkEnumeratePhysicalDevices");
	Rml::Vector<VkPhysicalDevice> devices(count);
	Check(vkEnumeratePhysicalDevices(m_instance, &count, devices.data()), "vkEnumeratePhysicalDevices");

	int best_score = -1;
	for (VkPhysicalDevice device : devices)
	{
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		if (properties.apiVersion < VK_API_VERSION_1_3)
			continue;

		VkPhysicalDeviceVulkan13Features features13 = {};
		features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		VkPhysicalDeviceVulkan12Features features12 = {};
		features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		features12.pNext = &features13;
		VkPhysicalDeviceFeatures2 features = {};
		features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features.pNext = &features12;
		vkGetPhysicalDeviceFeatures2(device, &features);
		if (!features.features.shaderSampledImageArrayDynamicIndexing || !features12.runtimeDescriptorArray ||
			!features12.descriptorBindingPartiallyBound || !features12.descriptorBindingSampledImageUpdateAfterBind ||
			!features12.descriptorBindingUpdateUnusedWhilePending || !features12.bufferDeviceAddress || !features13.dynamicRendering ||
			!features13.synchronization2)
			continue;

		const Rml::Vector<VkExtensionProperties> extensions = [device] {
			uint32_t extension_count = 0;
			Check(vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr), "vkEnumerateDeviceExtensionProperties");
			Rml::Vector<VkExtensionProperties> result(extension_count);
			Check(vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, result.data()), "vkEnumerateDeviceExtensionProperties");
			return result;
		}();
		const bool has_swapchain = std::any_of(extensions.begin(), extensions.end(),
			[](const VkExtensionProperties& extension) { return std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0; });
		if (!has_swapchain)
			continue;

		uint32_t family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
		Rml::Vector<VkQueueFamilyProperties> families(family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

		for (uint32_t family = 0; family < family_count; family++)
		{
			VkBool32 present = VK_FALSE;
			Check(vkGetPhysicalDeviceSurfaceSupportKHR(device, family, m_surface, &present), "vkGetPhysicalDeviceSurfaceSupportKHR");
			if (!(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present)
				continue;

			const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2
				: properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU           ? 1
																							: 0;
			if (score > best_score)
			{
				best_score = score;
				m_physical_device = device;
				m_queue_family = family;
			}
			break;
		}
	}

	if (!m_physical_device)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR,
			"Vulkan: No GPU supports Vulkan 1.3 with dynamic rendering, synchronization2, descriptor indexing, and buffer device addresses, and "
			"can present to the window.");
		return false;
	}
	return true;
}

bool RenderInterface_VKModern::CreateDevice(const Rml::Vector<const char*>& optional_extensions)
{
	VkPhysicalDeviceVulkan11Features supported11 = {};
	supported11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	VkPhysicalDeviceFeatures2 supported_features = {};
	supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	supported_features.pNext = &supported11;
	vkGetPhysicalDeviceFeatures2(m_physical_device, &supported_features);
	const VkPhysicalDeviceFeatures& supported = supported_features.features;

	m_features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	m_features13.dynamicRendering = VK_TRUE;
	m_features13.synchronization2 = VK_TRUE;

	m_features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	m_features12.pNext = &m_features13;
	m_features12.runtimeDescriptorArray = VK_TRUE;
	m_features12.descriptorBindingPartiallyBound = VK_TRUE;
	m_features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
	m_features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
	m_features12.bufferDeviceAddress = VK_TRUE;
	// Required by Vulkan 1.2, for applications that synchronize their own work with timeline semaphores.
	m_features12.timelineSemaphore = VK_TRUE;

	m_features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	m_features11.pNext = &m_features12;
	m_features11.samplerYcbcrConversion = supported11.samplerYcbcrConversion;

	m_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	m_features.pNext = &m_features11;
	m_features.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
	m_features.features.sampleRateShading = supported.sampleRateShading;
	m_sample_shading = (supported.sampleRateShading == VK_TRUE);
	// Enable compressed formats when available, so that CreateTexture() can upload them.
	m_features.features.textureCompressionBC = supported.textureCompressionBC;
	m_features.features.textureCompressionETC2 = supported.textureCompressionETC2;
	m_features.features.textureCompressionASTC_LDR = supported.textureCompressionASTC_LDR;

	m_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	uint32_t count = 0;
	Check(vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
	Rml::Vector<VkExtensionProperties> available(count);
	Check(vkEnumerateDeviceExtensionProperties(m_physical_device, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
	// Must be enabled when available, see VK_KHR_portability_subset.
	constexpr const char* portability_subset = "VK_KHR_portability_subset";
	for (const VkExtensionProperties& extension : available)
	{
		if (std::strcmp(extension.extensionName, portability_subset) == 0)
			m_device_extensions.push_back(portability_subset);
		for (const char* optional : optional_extensions)
			if (std::strcmp(extension.extensionName, optional) == 0)
				m_device_extensions.push_back(optional);
	}

	uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &family_count, nullptr);
	const float priority = 1.f;
	Rml::Vector<VkDeviceQueueCreateInfo> queues(family_count);
	for (uint32_t family = 0; family < family_count; family++)
	{
		queues[family].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queues[family].queueFamilyIndex = family;
		queues[family].queueCount = 1;
		queues[family].pQueuePriorities = &priority;
	}

	VkDeviceCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	info.pNext = &m_features;
	info.queueCreateInfoCount = family_count;
	info.pQueueCreateInfos = queues.data();
	info.enabledExtensionCount = uint32_t(m_device_extensions.size());
	info.ppEnabledExtensionNames = m_device_extensions.data();
	if (vkCreateDevice(m_physical_device, &info, nullptr, &m_device) != VK_SUCCESS)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The Vulkan device could not be created.");
		return false;
	}

	gladLoaderLoadVulkan(m_instance, m_physical_device, m_device);
	vkGetDeviceQueue(m_device, m_queue_family, 0, &m_queue);

	VmaVulkanFunctions functions = {};
	functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo allocator = {};
	allocator.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	allocator.vulkanApiVersion = VK_API_VERSION_1_3;
	allocator.physicalDevice = m_physical_device;
	allocator.device = m_device;
	allocator.instance = m_instance;
	allocator.pVulkanFunctions = &functions;
	Check(vmaCreateAllocator(&allocator, &m_allocator), "vmaCreateAllocator");
	return true;
}

bool RenderInterface_VKModern::ChooseFormats()
{
	// Layers are resolved or copied directly into the swapchain images, so they must share the same format. Use a linear
	// 8-bit format, since RmlUi blends colors in sRGB space.
	uint32_t count = 0;
	Check(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &count, nullptr), "vkGetPhysicalDeviceSurfaceFormatsKHR");
	Rml::Vector<VkSurfaceFormatKHR> formats(count);
	Check(vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical_device, m_surface, &count, formats.data()), "vkGetPhysicalDeviceSurfaceFormatsKHR");
	for (const VkSurfaceFormatKHR& format : formats)
	{
		if ((format.format != VK_FORMAT_B8G8R8A8_UNORM && format.format != VK_FORMAT_R8G8B8A8_UNORM) ||
			format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			continue;
		m_color_format = format.format;
		m_color_space = format.colorSpace;
		break;
	}
	if (m_color_format == VK_FORMAT_UNDEFINED)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The window surface supports neither B8G8R8A8_UNORM nor R8G8B8A8_UNORM.");
		return false;
	}

	// The final frame is resolved or copied into the swapchain image, and presented without blending.
	VkSurfaceCapabilitiesKHR capabilities;
	Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
	if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
		!(capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The window surface does not support opaque swapchain images that are transfer destinations.");
		return false;
	}

	// FIFO is the only present mode guaranteed to be supported.
	uint32_t mode_count = 0;
	Check(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &mode_count, nullptr), "vkGetPhysicalDeviceSurfacePresentModesKHR");
	Rml::Vector<VkPresentModeKHR> modes(mode_count);
	Check(vkGetPhysicalDeviceSurfacePresentModesKHR(m_physical_device, m_surface, &mode_count, modes.data()),
		"vkGetPhysicalDeviceSurfacePresentModesKHR");
	m_present_mode = RMLUI_RENDER_BACKEND_FIELD_PRESENT_MODE;
	if (std::find(modes.begin(), modes.end(), m_present_mode) == modes.end())
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Vulkan: The surface does not support the present mode %d, using FIFO.", int(m_present_mode));
		m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
	}

	// Only the stencil is used, for clip masks.
	for (VkFormat format : {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT})
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(m_physical_device, format, &properties);
		if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
			continue;
		m_stencil_format = format;
		break;
	}
	if (m_stencil_format == VK_FORMAT_UNDEFINED)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The GPU supports no stencil format.");
		return false;
	}

	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(m_physical_device, &properties);
	const VkSampleCountFlags supported = properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts &
		properties.limits.framebufferStencilSampleCounts;
	for (int samples = 64; samples > 1; samples /= 2)
	{
		if (samples > RMLUI_RENDER_BACKEND_FIELD_MSAA_SAMPLE_COUNT || !(supported & VkSampleCountFlags(samples)))
			continue;
		m_samples = VkSampleCountFlagBits(samples);
		break;
	}
	return true;
}

void RenderInterface_VKModern::CreateDescriptors()
{
	VkSamplerCreateInfo sampler = {};
	sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler.magFilter = VK_FILTER_LINEAR;
	sampler.minFilter = VK_FILTER_LINEAR;
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler.maxLod = VK_LOD_CLAMP_NONE;
	Check(vkCreateSampler(m_device, &sampler, nullptr, &m_sampler), "vkCreateSampler");

	VkPhysicalDeviceVulkan12Properties properties12 = {};
	properties12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
	VkPhysicalDeviceProperties2 properties = {};
	properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	properties.pNext = &properties12;
	vkGetPhysicalDeviceProperties2(m_physical_device, &properties);
	m_texture_capacity = std::min({MaxTextures, properties12.maxDescriptorSetUpdateAfterBindSampledImages,
		properties12.maxPerStageDescriptorUpdateAfterBindSampledImages, properties12.maxPerStageUpdateAfterBindResources - 1});

	VkDescriptorSetLayoutBinding bindings[2] = {};
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	bindings[0].descriptorCount = m_texture_capacity;
	bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	bindings[1].pImmutableSamplers = &m_sampler;

	// Textures are added and removed while frames using the table are recorded and executed. A slot is only reused after
	// all frames that could sample its previous texture have finished.
	const VkDescriptorBindingFlags binding_flags[2] = {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
			VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
		0};
	VkDescriptorSetLayoutBindingFlagsCreateInfo flags = {};
	flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
	flags.bindingCount = 2;
	flags.pBindingFlags = binding_flags;

	VkDescriptorSetLayoutCreateInfo layout = {};
	layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout.pNext = &flags;
	layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
	layout.bindingCount = 2;
	layout.pBindings = bindings;
	Check(vkCreateDescriptorSetLayout(m_device, &layout, nullptr, &m_descriptor_set_layout), "vkCreateDescriptorSetLayout");

	const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_texture_capacity}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
	VkDescriptorPoolCreateInfo pool = {};
	pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
	pool.maxSets = 1;
	pool.poolSizeCount = 2;
	pool.pPoolSizes = sizes;
	Check(vkCreateDescriptorPool(m_device, &pool, nullptr, &m_descriptor_pool), "vkCreateDescriptorPool");

	VkDescriptorSetAllocateInfo allocate = {};
	allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate.descriptorPool = m_descriptor_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts = &m_descriptor_set_layout;
	Check(vkAllocateDescriptorSets(m_device, &allocate, &m_descriptor_set), "vkAllocateDescriptorSets");

	const VkPushConstantRange push_constants = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, PushConstantsSize};
	VkPipelineLayoutCreateInfo pipeline_layout = {};
	pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout.setLayoutCount = 1;
	pipeline_layout.pSetLayouts = &m_descriptor_set_layout;
	pipeline_layout.pushConstantRangeCount = 1;
	pipeline_layout.pPushConstantRanges = &push_constants;
	Check(vkCreatePipelineLayout(m_device, &pipeline_layout, nullptr, &m_pipeline_layout), "vkCreatePipelineLayout");
}

void RenderInterface_VKModern::CreatePipelines()
{
	auto CreateModule = [this](const unsigned char* code, size_t size) {
		VkShaderModuleCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		info.codeSize = size;
		info.pCode = reinterpret_cast<const uint32_t*>(code);
		VkShaderModule module = VK_NULL_HANDLE;
		Check(vkCreateShaderModule(m_device, &info, nullptr, &module), "vkCreateShaderModule");
		return module;
	};
#define RMLUI_VK_MODULE(name) CreateModule(name, sizeof(name))
	const VkShaderModule geometry_vert = RMLUI_VK_MODULE(shader_vert_geometry);
	const VkShaderModule quad_vert = RMLUI_VK_MODULE(shader_vert_quad);
	const VkShaderModule blur_vert = RMLUI_VK_MODULE(shader_vert_blur);
	const VkShaderModule geometry_frag = RMLUI_VK_MODULE(shader_frag_geometry);
	const VkShaderModule gradient_frag = RMLUI_VK_MODULE(shader_frag_gradient);
	const VkShaderModule creation_frag = RMLUI_VK_MODULE(shader_frag_creation);
	const VkShaderModule passthrough_frag = RMLUI_VK_MODULE(shader_frag_passthrough);
	const VkShaderModule color_matrix_frag = RMLUI_VK_MODULE(shader_frag_color_matrix);
	const VkShaderModule blend_mask_frag = RMLUI_VK_MODULE(shader_frag_blend_mask);
	const VkShaderModule blur_frag = RMLUI_VK_MODULE(shader_frag_blur);
	const VkShaderModule drop_shadow_frag = RMLUI_VK_MODULE(shader_frag_drop_shadow);
#undef RMLUI_VK_MODULE

	enum class Blend { None, Premultiplied, Constant, NoColor };
	enum class Target { Layer, Postprocess };

	auto Create = [this](Pipeline pipeline, VkShaderModule vertex, VkShaderModule fragment, Target target, Blend blend) {
		const bool geometry = (target == Target::Layer && pipeline != Pipeline::LayerBlend && pipeline != Pipeline::LayerReplace);

		VkPipelineShaderStageCreateInfo stages[2] = {};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertex;
		stages[0].pName = "main";
		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragment;
		stages[1].pName = "main";

		const VkVertexInputBindingDescription binding = {0, sizeof(Rml::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
		const VkVertexInputAttributeDescription attributes[3] = {
			{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Rml::Vertex, position)},
			{1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(Rml::Vertex, colour)},
			{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Rml::Vertex, tex_coord)},
		};
		VkPipelineVertexInputStateCreateInfo vertex_input = {};
		vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		if (geometry)
		{
			vertex_input.vertexBindingDescriptionCount = 1;
			vertex_input.pVertexBindingDescriptions = &binding;
			vertex_input.vertexAttributeDescriptionCount = 3;
			vertex_input.pVertexAttributeDescriptions = attributes;
		}

		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = geometry ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

		VkPipelineViewportStateCreateInfo viewport = {};
		viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport.viewportCount = 1;
		viewport.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterization = {};
		rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization.cullMode = VK_CULL_MODE_NONE;
		rasterization.lineWidth = 1.f;

		VkPipelineMultisampleStateCreateInfo multisample = {};
		multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample.rasterizationSamples = (target == Target::Layer ? m_samples : VK_SAMPLE_COUNT_1_BIT);
		// Shade each sample when supported, this also antialiases the interior of triangles: textures, gradients, and shaders.
		multisample.sampleShadingEnable = (m_sample_shading && geometry && fragment ? VK_TRUE : VK_FALSE);
		multisample.minSampleShading = 1.f;

		// The stencil test, operation, and reference are set as dynamic state.
		VkPipelineDepthStencilStateCreateInfo depth_stencil = {};
		depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil.front.compareMask = 0xFF;
		depth_stencil.front.writeMask = 0xFF;
		depth_stencil.back = depth_stencil.front;

		VkPipelineColorBlendAttachmentState attachment = {};
		attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		attachment.colorBlendOp = VK_BLEND_OP_ADD;
		attachment.alphaBlendOp = VK_BLEND_OP_ADD;
		switch (blend)
		{
		case Blend::None: break;
		case Blend::NoColor: attachment.colorWriteMask = 0; break;
		case Blend::Premultiplied:
			attachment.blendEnable = VK_TRUE;
			attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			break;
		case Blend::Constant:
			attachment.blendEnable = VK_TRUE;
			attachment.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
			attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
			attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
			attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			break;
		}
		VkPipelineColorBlendStateCreateInfo color_blend = {};
		color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend.attachmentCount = 1;
		color_blend.pAttachments = &attachment;

		const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
			VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
		VkPipelineDynamicStateCreateInfo dynamic = {};
		dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic.dynamicStateCount = uint32_t(sizeof(dynamic_states) / sizeof(dynamic_states[0]));
		dynamic.pDynamicStates = dynamic_states;

		VkPipelineRenderingCreateInfo rendering = {};
		rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachmentFormats = &m_color_format;
		rendering.stencilAttachmentFormat = (target == Target::Layer ? m_stencil_format : VK_FORMAT_UNDEFINED);

		VkGraphicsPipelineCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		info.pNext = &rendering;
		// The clip mask only writes to the stencil buffer, so it needs no fragment shader.
		info.stageCount = (fragment ? 2 : 1);
		info.pStages = stages;
		info.pVertexInputState = &vertex_input;
		info.pInputAssemblyState = &input_assembly;
		info.pViewportState = &viewport;
		info.pRasterizationState = &rasterization;
		info.pMultisampleState = &multisample;
		info.pDepthStencilState = &depth_stencil;
		info.pColorBlendState = &color_blend;
		info.pDynamicState = &dynamic;
		info.layout = m_pipeline_layout;
		Check(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &info, nullptr, &m_pipelines[size_t(pipeline)]), "vkCreateGraphicsPipelines");
	};

	Create(Pipeline::Geometry, geometry_vert, geometry_frag, Target::Layer, Blend::Premultiplied);
	Create(Pipeline::Gradient, geometry_vert, gradient_frag, Target::Layer, Blend::Premultiplied);
	Create(Pipeline::Creation, geometry_vert, creation_frag, Target::Layer, Blend::Premultiplied);
	Create(Pipeline::ClipMask, geometry_vert, VK_NULL_HANDLE, Target::Layer, Blend::NoColor);
	Create(Pipeline::LayerBlend, quad_vert, passthrough_frag, Target::Layer, Blend::Premultiplied);
	Create(Pipeline::LayerReplace, quad_vert, passthrough_frag, Target::Layer, Blend::None);
	Create(Pipeline::Copy, quad_vert, passthrough_frag, Target::Postprocess, Blend::None);
	Create(Pipeline::Blend, quad_vert, passthrough_frag, Target::Postprocess, Blend::Premultiplied);
	Create(Pipeline::Opacity, quad_vert, passthrough_frag, Target::Postprocess, Blend::Constant);
	Create(Pipeline::ColorMatrix, quad_vert, color_matrix_frag, Target::Postprocess, Blend::None);
	Create(Pipeline::BlendMask, quad_vert, blend_mask_frag, Target::Postprocess, Blend::None);
	Create(Pipeline::Blur, blur_vert, blur_frag, Target::Postprocess, Blend::None);
	Create(Pipeline::DropShadow, quad_vert, drop_shadow_frag, Target::Postprocess, Blend::None);

	for (VkShaderModule module : {geometry_vert, quad_vert, blur_vert, geometry_frag, gradient_frag, creation_frag, passthrough_frag,
			 color_matrix_frag, blend_mask_frag, blur_frag, drop_shadow_frag})
		vkDestroyShaderModule(m_device, module, nullptr);
}

void RenderInterface_VKModern::CreateFrames()
{
	for (Frame& frame : m_frames)
	{
		VkCommandPoolCreateInfo pool = {};
		pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		pool.queueFamilyIndex = m_queue_family;
		Check(vkCreateCommandPool(m_device, &pool, nullptr, &frame.command_pool), "vkCreateCommandPool");

		VkCommandBuffer buffers[2];
		VkCommandBufferAllocateInfo allocate = {};
		allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocate.commandPool = frame.command_pool;
		allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocate.commandBufferCount = 2;
		Check(vkAllocateCommandBuffers(m_device, &allocate, buffers), "vkAllocateCommandBuffers");
		frame.commands = buffers[0];
		frame.uploads = buffers[1];

		VkFenceCreateInfo fence = {};
		fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		Check(vkCreateFence(m_device, &fence, nullptr, &frame.fence), "vkCreateFence");

		VkSemaphoreCreateInfo semaphore = {};
		semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		Check(vkCreateSemaphore(m_device, &semaphore, nullptr, &frame.image_acquired), "vkCreateSemaphore");
	}
}

void RenderInterface_VKModern::Shutdown()
{
	if (!m_device)
	{
		if (m_surface)
			vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
		if (m_instance)
			vkDestroyInstance(m_instance, nullptr);
		gladLoaderUnloadVulkan();
		return;
	}

	{
		std::lock_guard<std::mutex> lock(m_queue_mutex);
		Check(vkDeviceWaitIdle(m_device), "vkDeviceWaitIdle");
	}
	CollectGarbage(true);

	for (Image& image : m_textures)
		DestroyImage(image);
	for (Image& image : m_targets.layers)
		DestroyImage(image);
	for (Image& image : m_targets.postprocess)
		DestroyImage(image);
	DestroyImage(m_targets.stencil);

	for (GeometryBlock& block : m_geometry_blocks)
	{
		if (!block.buffer)
			continue;
		vmaClearVirtualBlock(block.vertices);
		vmaClearVirtualBlock(block.indices);
		DestroyGeometryBlock(block);
	}

	for (Frame& frame : m_frames)
	{
		for (StagingBuffer& buffer : frame.staging)
			vmaDestroyBuffer(m_allocator, buffer.buffer, buffer.allocation);
		vkDestroyCommandPool(m_device, frame.command_pool, nullptr);
		vkDestroyFence(m_device, frame.fence, nullptr);
		vkDestroySemaphore(m_device, frame.image_acquired, nullptr);
	}

	RetireSwapchain();
	CollectGarbage(true);

	for (VkPipeline pipeline : m_pipelines)
		vkDestroyPipeline(m_device, pipeline, nullptr);
	vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
	vkDestroySampler(m_device, m_sampler, nullptr);
	vmaDestroyAllocator(m_allocator);
	vkDestroyDevice(m_device, nullptr);
	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	vkDestroyInstance(m_instance, nullptr);
	gladLoaderUnloadVulkan();
}

void RenderInterface_VKModern::SetViewport(int width, int height)
{
	if (width > 0 && height > 0)
		m_requested_extent = {uint32_t(width), uint32_t(height)};

	VkSurfaceCapabilitiesKHR capabilities;
	Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical_device, m_surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
	VkExtent2D extent = capabilities.currentExtent;
	if (extent.width == UINT32_MAX)
	{
		extent.width = Rml::Math::Clamp(m_requested_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		extent.height = Rml::Math::Clamp(m_requested_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
	}
	// A minimized window has zero extent, we cannot have a swapchain until it is restored.
	if (extent.width == 0 || extent.height == 0)
	{
		RetireSwapchain();
		return;
	}

	CreateSwapchain(extent, capabilities);
}

void RenderInterface_VKModern::CreateSwapchain(VkExtent2D extent, const VkSurfaceCapabilitiesKHR& capabilities)
{
	// A maximum of zero means no limit.
	uint32_t image_count = FramesInFlight;
	if (image_count < capabilities.minImageCount || (capabilities.maxImageCount != 0 && image_count > capabilities.maxImageCount))
		image_count = capabilities.minImageCount;

	const VkSwapchainKHR old_swapchain = m_swapchain.swapchain;
	RetireSwapchain();

	VkSwapchainCreateInfoKHR info = {};
	info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	info.surface = m_surface;
	info.minImageCount = image_count;
	info.imageFormat = m_color_format;
	info.imageColorSpace = m_color_space;
	info.imageExtent = extent;
	info.imageArrayLayers = 1;
	info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	info.preTransform = (capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
																								   : capabilities.currentTransform;
	info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	info.presentMode = m_present_mode;
	info.clipped = VK_TRUE;
	// Lets the driver reuse resources of the old swapchain and finish its queued presents.
	info.oldSwapchain = old_swapchain;
	Check(vkCreateSwapchainKHR(m_device, &info, nullptr, &m_swapchain.swapchain), "vkCreateSwapchainKHR");
	m_swapchain.extent = extent;

	uint32_t count = 0;
	Check(vkGetSwapchainImagesKHR(m_device, m_swapchain.swapchain, &count, nullptr), "vkGetSwapchainImagesKHR");
	m_swapchain.images.resize(count);
	Check(vkGetSwapchainImagesKHR(m_device, m_swapchain.swapchain, &count, m_swapchain.images.data()), "vkGetSwapchainImagesKHR");

	m_swapchain.render_finished.resize(count);
	for (VkSemaphore& semaphore : m_swapchain.render_finished)
	{
		VkSemaphoreCreateInfo semaphore_info = {};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		Check(vkCreateSemaphore(m_device, &semaphore_info, nullptr, &semaphore), "vkCreateSemaphore");
	}
}

void RenderInterface_VKModern::RetireSwapchain()
{
	if (!m_swapchain.swapchain)
		return;
	// Frames in flight may still present to it, so destroy it later.
	m_garbage_swapchains.emplace_back(m_frame_counter, std::move(m_swapchain));
	m_swapchain = Swapchain();
}

bool RenderInterface_VKModern::IsSwapchainValid()
{
	return m_swapchain.swapchain != VK_NULL_HANDLE;
}

void RenderInterface_VKModern::RecreateSwapchain()
{
	SetViewport(int(m_requested_extent.width), int(m_requested_extent.height));
}

RenderInterface_VKModern::Frame& RenderInterface_VKModern::PrepareFrame()
{
	Frame& frame = m_frames[m_frame_counter % FramesInFlight];
	if (m_frame_prepared)
		return frame;

	// Wait until the GPU is done with the last frame that used these resources.
	Check(vkWaitForFences(m_device, 1, &frame.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
	CollectGarbage(false);
	Check(vkResetCommandPool(m_device, frame.command_pool, 0), "vkResetCommandPool");

	// Keep one buffer of the standard size between frames, release any larger buffers made for big uploads.
	const size_t kept = (!frame.staging.empty() && frame.staging[0].size == StagingBufferSize ? 1 : 0);
	for (size_t i = kept; i < frame.staging.size(); i++)
		vmaDestroyBuffer(m_allocator, frame.staging[i].buffer, frame.staging[i].allocation);
	frame.staging.resize(kept);
	frame.staging_used = 0;

	m_frame_prepared = true;
	return frame;
}

void RenderInterface_VKModern::CollectGarbage(bool everything)
{
	// Entries are ordered by their frame stamp. A resource released during frame N may still be used by frame N, which is
	// complete once the fence of frame N + FramesInFlight has been waited on.
	auto Expired = [this, everything](const auto& entry) { return !everything && m_frame_counter - entry.first < FramesInFlight; };

	auto images_end = std::find_if(m_garbage_images.begin(), m_garbage_images.end(), Expired);
	for (auto it = m_garbage_images.begin(); it != images_end; ++it)
		DestroyImage(it->second);
	m_garbage_images.erase(m_garbage_images.begin(), images_end);

	auto geometries_end = std::find_if(m_garbage_geometries.begin(), m_garbage_geometries.end(), Expired);
	for (auto it = m_garbage_geometries.begin(); it != geometries_end; ++it)
	{
		GeometryBlock& block = m_geometry_blocks[it->second.block];
		vmaVirtualFree(block.vertices, it->second.vertices);
		vmaVirtualFree(block.indices, it->second.indices);
		// An empty block is no longer used by any frame in flight. Always keep the first block around.
		if (it->second.block != 0 && vmaIsVirtualBlockEmpty(block.vertices))
			DestroyGeometryBlock(block);
	}
	m_garbage_geometries.erase(m_garbage_geometries.begin(), geometries_end);

	// Destroy older swapchains first, a swapchain must outlive the ones it replaced.
	auto swapchains_end = std::find_if(m_garbage_swapchains.begin(), m_garbage_swapchains.end(), Expired);
	for (auto it = m_garbage_swapchains.begin(); it != swapchains_end; ++it)
	{
		for (VkSemaphore semaphore : it->second.render_finished)
			vkDestroySemaphore(m_device, semaphore, nullptr);
		vkDestroySwapchainKHR(m_device, it->second.swapchain, nullptr);
	}
	m_garbage_swapchains.erase(m_garbage_swapchains.begin(), swapchains_end);
}

VkCommandBuffer RenderInterface_VKModern::UploadCommands()
{
	Frame& frame = PrepareFrame();
	if (frame.uploads_recording)
		return frame.uploads;

	VkCommandBufferBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	Check(vkBeginCommandBuffer(frame.uploads, &begin), "vkBeginCommandBuffer");
	frame.uploads_recording = true;
	return frame.uploads;
}

RenderInterface_VKModern::StagingAllocation RenderInterface_VKModern::AllocateStaging(VkDeviceSize size)
{
	Frame& frame = PrepareFrame();
	VkDeviceSize offset = AlignUp(frame.staging_used, StagingAlignment);
	if (frame.staging.empty() || offset + size > frame.staging.back().size)
	{
		VkBufferCreateInfo buffer = {};
		buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		buffer.size = std::max(StagingBufferSize, size);
		buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

		VmaAllocationCreateInfo allocation = {};
		allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		allocation.usage = VMA_MEMORY_USAGE_AUTO;
		allocation.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		StagingBuffer staging;
		staging.size = buffer.size;
		VmaAllocationInfo info;
		Check(vmaCreateBuffer(m_allocator, &buffer, &allocation, &staging.buffer, &staging.allocation, &info), "vmaCreateBuffer");
		staging.mapped = static_cast<Rml::byte*>(info.pMappedData);

		VkBufferDeviceAddressInfo address = {};
		address.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		address.buffer = staging.buffer;
		staging.address = vkGetBufferDeviceAddress(m_device, &address);

		frame.staging.push_back(staging);
		offset = 0;
	}
	frame.staging_used = offset + size;
	const StagingBuffer& staging = frame.staging.back();
	return StagingAllocation{staging.buffer, offset, staging.mapped + offset, staging.address + offset};
}

RenderInterface_VKModern::Image RenderInterface_VKModern::CreateImage(VkFormat format, Rml::Vector2i dimensions, uint32_t mip_levels,
	VkSampleCountFlagBits samples, VkImageUsageFlags usage)
{
	Image image;

	VkImageCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.format = format;
	info.extent = {uint32_t(dimensions.x), uint32_t(dimensions.y), 1};
	info.mipLevels = mip_levels;
	info.arrayLayers = 1;
	info.samples = samples;
	info.tiling = VK_IMAGE_TILING_OPTIMAL;
	info.usage = usage;
	info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocation = {};
	allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	Check(vmaCreateImage(m_allocator, &info, &allocation, &image.image, &image.allocation, nullptr), "vmaCreateImage");

	const VkImageAspectFlags aspect = AspectOf(format);
	VkImageViewCreateInfo view = {};
	view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view.image = image.image;
	view.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view.format = format;
	view.subresourceRange = {aspect, 0, mip_levels, 0, 1};
	Check(vkCreateImageView(m_device, &view, nullptr, &image.view), "vkCreateImageView");

	// Images stay in the general layout for their entire lifetime, this is their only layout transition.
	ImageBarrier(UploadCommands(), image.image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
		VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);

	if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
		AddToTextureTable(image);
	return image;
}

void RenderInterface_VKModern::AddToTextureTable(Image& image)
{
	if (m_free_texture_indices.empty() && m_next_texture_index == m_texture_capacity)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: All %u texture slots are in use.", m_texture_capacity);
		std::abort();
	}
	if (m_free_texture_indices.empty())
	{
		image.texture_index = m_next_texture_index++;
	}
	else
	{
		image.texture_index = m_free_texture_indices.back();
		m_free_texture_indices.pop_back();
	}

	const VkDescriptorImageInfo image_info = {VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL};
	VkWriteDescriptorSet write = {};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = m_descriptor_set;
	write.dstBinding = 0;
	write.dstArrayElement = image.texture_index;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	write.pImageInfo = &image_info;
	vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void RenderInterface_VKModern::DestroyGeometryBlock(GeometryBlock& block)
{
	vmaDestroyVirtualBlock(block.vertices);
	vmaDestroyVirtualBlock(block.indices);
	vmaDestroyBuffer(m_allocator, block.buffer, block.allocation);
	block = GeometryBlock();
}

void RenderInterface_VKModern::DestroyImage(Image& image)
{
	if (!image.image)
		return;
	if (image.texture_index != NoTexture)
		m_free_texture_indices.push_back(image.texture_index);
	vkDestroyImageView(m_device, image.view, nullptr);
	vmaDestroyImage(m_allocator, image.image, image.allocation);
	image = Image();
}

Rml::TextureHandle RenderInterface_VKModern::RegisterTexture(Image image)
{
	const uint32_t index = image.texture_index;
	if (index >= m_textures.size())
		m_textures.resize(index + 1);
	m_textures[index] = image;
	return Rml::TextureHandle(index) + 1;
}

const RenderInterface_VKModern::Image& RenderInterface_VKModern::GetTexture(Rml::TextureHandle handle) const
{
	return m_textures[size_t(handle - 1)];
}

Rml::TextureHandle RenderInterface_VKModern::UploadTexture(VkFormat format, Rml::Span<const TextureMip> mips, AlphaMode alpha_mode)
{
	VkDeviceSize size = 0;
	for (const TextureMip& mip : mips)
		size = AlignUp(size, StagingAlignment) + mip.data.size();
	const StagingAllocation staging = AllocateStaging(size);

	Rml::Vector<VkBufferImageCopy> regions(mips.size());
	VkDeviceSize offset = 0;
	for (size_t level = 0; level < mips.size(); level++)
	{
		const TextureMip& mip = mips[level];
		offset = AlignUp(offset, StagingAlignment);
		std::memcpy(staging.mapped + offset, mip.data.data(), mip.data.size());
		VkBufferImageCopy& region = regions[level];
		region.bufferOffset = staging.offset + offset;
		region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, uint32_t(level), 0, 1};
		region.imageExtent = {uint32_t(mip.dimensions.x), uint32_t(mip.dimensions.y), 1};
		offset += mip.data.size();
	}

	Image image = CreateImage(format, mips[0].dimensions, uint32_t(mips.size()), VK_SAMPLE_COUNT_1_BIT,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	image.alpha_mode = alpha_mode;
	vkCmdCopyBufferToImage(UploadCommands(), staging.buffer, image.image, VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
	return RegisterTexture(image);
}

Rml::TextureHandle RenderInterface_VKModern::CreateTexture(VkFormat format, Rml::Span<const TextureMip> mips, AlphaMode alpha_mode)
{
	RMLUI_ASSERT(!mips.empty());
	constexpr VkFormatFeatureFlags required =
		VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
	VkFormatProperties properties;
	vkGetPhysicalDeviceFormatProperties(m_physical_device, format, &properties);
	if ((properties.optimalTilingFeatures & required) != required)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: The GPU cannot sample textures of format %d.", int(format));
		return {};
	}
	return UploadTexture(format, mips, alpha_mode);
}

Rml::TextureHandle RenderInterface_VKModern::RegisterVulkanTexture(VkImage image, VmaAllocation allocation, VkImageView view)
{
	Image texture;
	texture.image = image;
	texture.allocation = allocation;
	texture.view = view;
	AddToTextureTable(texture);
	return RegisterTexture(texture);
}

VkInstance RenderInterface_VKModern::GetInstance() const
{
	return m_instance;
}

VkPhysicalDevice RenderInterface_VKModern::GetPhysicalDevice() const
{
	return m_physical_device;
}

VkDevice RenderInterface_VKModern::GetDevice() const
{
	return m_device;
}

VkQueue RenderInterface_VKModern::GetQueue() const
{
	return m_queue;
}

uint32_t RenderInterface_VKModern::GetQueueFamily() const
{
	return m_queue_family;
}

VmaAllocator RenderInterface_VKModern::GetAllocator() const
{
	return m_allocator;
}

const Rml::Vector<const char*>& RenderInterface_VKModern::GetDeviceExtensions() const
{
	return m_device_extensions;
}

const VkPhysicalDeviceFeatures2& RenderInterface_VKModern::GetDeviceFeatures() const
{
	return m_features;
}

std::mutex& RenderInterface_VKModern::GetQueueMutex()
{
	return m_queue_mutex;
}

Rml::TextureHandle RenderInterface_VKModern::GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions)
{
	RMLUI_ASSERT(source_data.size() == size_t(source_dimensions.x * source_dimensions.y * 4));
	const TextureMip mip = {source_dimensions, source_data};
	return UploadTexture(VK_FORMAT_R8G8B8A8_UNORM, {&mip, 1}, AlphaMode::Premultiplied);
}

Rml::TextureHandle RenderInterface_VKModern::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
{
	Rml::FileInterface* file_interface = Rml::GetFileInterface();
	Rml::FileHandle file_handle = file_interface->Open(source);
	if (!file_handle)
		return {};

	file_interface->Seek(file_handle, 0, SEEK_END);
	const size_t buffer_size = file_interface->Tell(file_handle);
	file_interface->Seek(file_handle, 0, SEEK_SET);

	if (buffer_size <= sizeof(TGAHeader))
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Texture file size is smaller than TGAHeader, file is not a valid TGA image.");
		file_interface->Close(file_handle);
		return {};
	}

	using Rml::byte;
	Rml::UniquePtr<byte[]> buffer(new byte[buffer_size]);
	file_interface->Read(buffer.get(), buffer_size, file_handle);
	file_interface->Close(file_handle);

	TGAHeader header;
	std::memcpy(&header, buffer.get(), sizeof(TGAHeader));

	const int color_mode = header.bitsPerPixel / 8;
	const size_t image_size = header.width * header.height * 4; // We always make 32bit textures

	if (header.dataType != 2)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24/32bit uncompressed TGAs are supported.");
		return {};
	}

	// Ensure we have at least 3 colors
	if (color_mode < 3)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24 and 32bit textures are supported.");
		return {};
	}

	const byte* image_src = buffer.get() + sizeof(TGAHeader);
	Rml::UniquePtr<byte[]> image_dest_buffer(new byte[image_size]);
	byte* image_dest = image_dest_buffer.get();

	// Targa is BGR, swap to RGB, flip Y axis, and convert to premultiplied alpha.
	for (long y = 0; y < header.height; y++)
	{
		long read_index = y * header.width * color_mode;
		long write_index = ((header.imageDescriptor & 32) != 0) ? read_index : (header.height - y - 1) * header.width * 4;
		for (long x = 0; x < header.width; x++)
		{
			image_dest[write_index] = image_src[read_index + 2];
			image_dest[write_index + 1] = image_src[read_index + 1];
			image_dest[write_index + 2] = image_src[read_index];
			if (color_mode == 4)
			{
				const byte alpha = image_src[read_index + 3];
				for (size_t j = 0; j < 3; j++)
					image_dest[write_index + j] = byte((image_dest[write_index + j] * alpha) / 255);
				image_dest[write_index + 3] = alpha;
			}
			else
				image_dest[write_index + 3] = 255;

			write_index += 4;
			read_index += color_mode;
		}
	}

	texture_dimensions.x = header.width;
	texture_dimensions.y = header.height;

	return GenerateTexture({image_dest, image_size}, texture_dimensions);
}

void RenderInterface_VKModern::ReleaseTexture(Rml::TextureHandle texture_handle)
{
	Image& image = m_textures[size_t(texture_handle - 1)];
	m_garbage_images.emplace_back(m_frame_counter, image);
	image = Image();
}

Rml::CompiledGeometryHandle RenderInterface_VKModern::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
{
	Geometry geometry;
	if (!vertices.empty() && !indices.empty())
	{
		VmaVirtualAllocationCreateInfo vertex_request = {};
		vertex_request.size = vertices.size();
		VmaVirtualAllocationCreateInfo index_request = {};
		index_request.size = indices.size();

		VkDeviceSize first_vertex = 0;
		VkDeviceSize first_index = 0;
		auto Allocate = [&](uint32_t block_index) {
			const GeometryBlock& block = m_geometry_blocks[block_index];
			if (!block.buffer)
				return false;
			if (vmaVirtualAllocate(block.vertices, &vertex_request, &geometry.vertices, &first_vertex) != VK_SUCCESS)
				return false;
			if (vmaVirtualAllocate(block.indices, &index_request, &geometry.indices, &first_index) != VK_SUCCESS)
			{
				vmaVirtualFree(block.vertices, geometry.vertices);
				return false;
			}
			geometry.block = block_index;
			return true;
		};

		bool allocated = false;
		for (uint32_t i = 0; i < uint32_t(m_geometry_blocks.size()) && !allocated; i++)
			allocated = Allocate(i);

		if (!allocated)
		{
			GeometryBlock block;
			const VkDeviceSize vertex_capacity = std::max(VkDeviceSize(GeometryBlockVertices), VkDeviceSize(vertices.size()));
			const VkDeviceSize index_capacity = std::max(VkDeviceSize(GeometryBlockIndices), VkDeviceSize(indices.size()));
			block.index_offset = vertex_capacity * sizeof(Rml::Vertex);

			VkBufferCreateInfo buffer = {};
			buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			buffer.size = block.index_offset + index_capacity * sizeof(int);
			buffer.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

			// Prefers device-local memory when it is host-visible, otherwise uses system memory.
			VmaAllocationCreateInfo allocation = {};
			allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
			allocation.usage = VMA_MEMORY_USAGE_AUTO;
			allocation.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			VmaAllocationInfo info;
			Check(vmaCreateBuffer(m_allocator, &buffer, &allocation, &block.buffer, &block.allocation, &info), "vmaCreateBuffer");
			block.mapped = static_cast<Rml::byte*>(info.pMappedData);

			VmaVirtualBlockCreateInfo virtual_block = {};
			virtual_block.size = vertex_capacity;
			Check(vmaCreateVirtualBlock(&virtual_block, &block.vertices), "vmaCreateVirtualBlock");
			virtual_block.size = index_capacity;
			Check(vmaCreateVirtualBlock(&virtual_block, &block.indices), "vmaCreateVirtualBlock");

			// Reuse the slot of a released block, so that the indices of the other blocks stay valid.
			const auto slot = std::find_if(m_geometry_blocks.begin(), m_geometry_blocks.end(), [](const GeometryBlock& b) { return !b.buffer; });
			const uint32_t block_index = uint32_t(slot - m_geometry_blocks.begin());
			if (slot == m_geometry_blocks.end())
				m_geometry_blocks.push_back(block);
			else
				*slot = block;
			allocated = Allocate(block_index);
			RMLUI_ASSERT(allocated);
		}

		const GeometryBlock& block = m_geometry_blocks[geometry.block];
		std::memcpy(block.mapped + first_vertex * sizeof(Rml::Vertex), vertices.data(), vertices.size() * sizeof(Rml::Vertex));
		std::memcpy(block.mapped + block.index_offset + first_index * sizeof(int), indices.data(), indices.size() * sizeof(int));
		geometry.first_vertex = uint32_t(first_vertex);
		geometry.first_index = uint32_t(first_index);
		geometry.num_indices = uint32_t(indices.size());
	}

	if (m_free_geometries.empty())
	{
		m_geometries.push_back(geometry);
		return Rml::CompiledGeometryHandle(m_geometries.size());
	}
	const uint32_t index = m_free_geometries.back();
	m_free_geometries.pop_back();
	m_geometries[index] = geometry;
	return Rml::CompiledGeometryHandle(index) + 1;
}

void RenderInterface_VKModern::ReleaseGeometry(Rml::CompiledGeometryHandle handle)
{
	const uint32_t index = uint32_t(handle - 1);
	// The vertex and index ranges may still be read by frames in flight, but the handle can be reused immediately.
	if (m_geometries[index].num_indices > 0)
		m_garbage_geometries.emplace_back(m_frame_counter, m_geometries[index]);
	m_free_geometries.push_back(index);
}

void RenderInterface_VKModern::EnsureRenderTargets()
{
	const Rml::Vector2i extent(int(m_swapchain.extent.width), int(m_swapchain.extent.height));
	m_render_region = Rml::Rectanglei::FromSize(extent);
	// Round up the size to reduce reallocations during resizing. The frame is rendered into the top-left region.
	const Rml::Vector2i dimensions = (extent + Rml::Vector2i(RenderTargetGranularity - 1)) / RenderTargetGranularity * RenderTargetGranularity;
	if (dimensions == m_targets.dimensions)
		return;

	// Frames in flight may still be using the old targets.
	for (Image& image : m_targets.layers)
		m_garbage_images.emplace_back(m_frame_counter, image);
	for (Image& image : m_targets.postprocess)
		if (image.image)
			m_garbage_images.emplace_back(m_frame_counter, image);
	if (m_targets.stencil.image)
		m_garbage_images.emplace_back(m_frame_counter, m_targets.stencil);
	m_targets = RenderTargets();

	m_targets.dimensions = dimensions;
	m_targets.stencil = CreateImage(m_stencil_format, dimensions, 1, m_samples, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

	// Transform from pixel coordinates with a top-left origin to Vulkan clip space.
	Rml::Matrix4f correction;
	correction.SetColumns({1.f, 0.f, 0.f, 0.f}, {0.f, -1.f, 0.f, 0.f}, {0.f, 0.f, 0.5f, 0.f}, {0.f, 0.f, 0.5f, 1.f});
	m_projection = correction * Rml::Matrix4f::ProjectOrtho(0.f, float(dimensions.x), float(dimensions.y), 0.f, -10000.f, 10000.f);
}

const RenderInterface_VKModern::Image& RenderInterface_VKModern::GetLayer(Rml::LayerHandle layer)
{
	while (m_targets.layers.size() <= size_t(layer))
		m_targets.layers.push_back(
			CreateImage(m_color_format, m_targets.dimensions, 1, m_samples, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
	return m_targets.layers[size_t(layer)];
}

const RenderInterface_VKModern::Image& RenderInterface_VKModern::GetPostprocess(Postprocess postprocess)
{
	Image& image = m_targets.postprocess[size_t(postprocess)];
	if (!image.image)
		image = CreateImage(m_color_format, m_targets.dimensions, 1, VK_SAMPLE_COUNT_1_BIT, PostprocessUsage);
	return image;
}

Rml::Rectanglef RenderInterface_VKModern::PixelsToDevice(Rml::Rectanglei region) const
{
	const Rml::Vector2f size(m_targets.dimensions);
	return Rml::Rectanglef::FromCorners(Rml::Vector2f(region.p0) / size * 2.f - Rml::Vector2f(1.f),
		Rml::Vector2f(region.p1) / size * 2.f - Rml::Vector2f(1.f));
}

void RenderInterface_VKModern::BeginFrame()
{
	if (!m_swapchain.swapchain)
	{
		Rml::Log::Message(Rml::Log::LT_ERROR, "Vulkan: BeginFrame() needs a valid swapchain, see IsSwapchainValid().");
		std::abort();
	}

	Frame& frame = PrepareFrame();
	EnsureRenderTargets();

	// If the swapchain is out of date, recreate it and try once more. Without an image, the frame is rendered but not
	// presented.
	m_image_index = UINT32_MAX;
	for (int attempt = 0; attempt < 2 && m_swapchain.swapchain; attempt++)
	{
		uint32_t index = 0;
		const VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain.swapchain, UINT64_MAX, frame.image_acquired, VK_NULL_HANDLE, &index);
		if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
		{
			m_image_index = index;
			break;
		}
		if (result != VK_ERROR_OUT_OF_DATE_KHR)
			Check(result, "vkAcquireNextImageKHR");
		RecreateSwapchain();
	}

	// The new swapchain may have a different size.
	if (m_swapchain.swapchain)
		EnsureRenderTargets();

	m_commands = frame.commands;
	VkCommandBufferBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	Check(vkBeginCommandBuffer(m_commands, &begin), "vkBeginCommandBuffer");

	// Set up the state shared by all pipelines: the texture table, the viewport, and the dynamic state.
	vkCmdBindDescriptorSets(m_commands, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0, 1, &m_descriptor_set, 0, nullptr);
	const VkViewport viewport = {0.f, 0.f, float(m_targets.dimensions.x), float(m_targets.dimensions.y), 0.f, 1.f};
	vkCmdSetViewport(m_commands, 0, 1, &viewport);
	m_applied_scissor = ToRect(m_render_region);
	vkCmdSetScissor(m_commands, 0, 1, &m_applied_scissor);
	m_stencil = StencilState();
	vkCmdSetStencilTestEnable(m_commands, VK_FALSE);
	vkCmdSetStencilOp(m_commands, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
	vkCmdSetStencilReference(m_commands, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
	m_blend_constant = 1.f;
	const float blend_constants[4] = {1.f, 1.f, 1.f, 1.f};
	vkCmdSetBlendConstants(m_commands, blend_constants);

	m_bound_pipeline = Pipeline::Count;
	m_bound_geometry_block = UINT32_MAX;
	// Order this frame after the work of previous frames on the same images.
	m_barrier_pending = true;
	m_scissor = m_render_region;
	m_clip_mask_enabled = false;
	m_clip_mask_set = false;
	m_clip_mask_value = 0;
	SetTransform(nullptr);

	m_layer_count = 1;
	BeginLayerRendering(0, true);
}

void RenderInterface_VKModern::EndFrame()
{
	EndRendering();
	Frame& frame = m_frames[m_frame_counter % FramesInFlight];
	const bool acquired = (m_image_index != UINT32_MAX);

	if (acquired)
	{
		Barrier();
		const VkImage image = m_swapchain.images[m_image_index];
		ImageBarrier(m_commands, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		ResolveOrCopy(m_commands, m_targets.layers[0].image, image, m_samples != VK_SAMPLE_COUNT_1_BIT, m_render_region);
		ImageBarrier(m_commands, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
	}
	Check(vkEndCommandBuffer(m_commands), "vkEndCommandBuffer");

	VkCommandBufferSubmitInfo command_buffers[2] = {};
	uint32_t command_buffer_count = 0;
	if (frame.uploads_recording)
	{
		FullBarrier(frame.uploads);
		Check(vkEndCommandBuffer(frame.uploads), "vkEndCommandBuffer");
		frame.uploads_recording = false;
		command_buffers[command_buffer_count].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
		command_buffers[command_buffer_count++].commandBuffer = frame.uploads;
	}
	command_buffers[command_buffer_count].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
	command_buffers[command_buffer_count++].commandBuffer = m_commands;

	VkSemaphoreSubmitInfo wait = {};
	wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	wait.semaphore = frame.image_acquired;
	wait.stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
	VkSemaphoreSubmitInfo signal = {};
	signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
	signal.semaphore = acquired ? m_swapchain.render_finished[m_image_index] : VK_NULL_HANDLE;
	signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

	VkSubmitInfo2 submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submit.waitSemaphoreInfoCount = acquired ? 1 : 0;
	submit.pWaitSemaphoreInfos = &wait;
	submit.commandBufferInfoCount = command_buffer_count;
	submit.pCommandBufferInfos = command_buffers;
	submit.signalSemaphoreInfoCount = acquired ? 1 : 0;
	submit.pSignalSemaphoreInfos = &signal;
	Check(vkResetFences(m_device, 1, &frame.fence), "vkResetFences");
	std::unique_lock<std::mutex> lock(m_queue_mutex);
	Check(vkQueueSubmit2(m_queue, 1, &submit, frame.fence), "vkQueueSubmit2");

	m_frame_counter++;
	m_frame_prepared = false;
	m_commands = VK_NULL_HANDLE;
	if (!acquired)
		return;

	VkPresentInfoKHR present = {};
	present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present.waitSemaphoreCount = 1;
	present.pWaitSemaphores = &signal.semaphore;
	present.swapchainCount = 1;
	present.pSwapchains = &m_swapchain.swapchain;
	present.pImageIndices = &m_image_index;
	const VkResult result = vkQueuePresentKHR(m_queue, &present);
	lock.unlock();
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		RecreateSwapchain();
	else
		Check(result, "vkQueuePresentKHR");
}

void RenderInterface_VKModern::Barrier()
{
	if (!m_barrier_pending)
		return;
	FullBarrier(m_commands);
	m_barrier_pending = false;
}

void RenderInterface_VKModern::BeginRendering(const Image& target, bool layer, bool clear)
{
	EndRendering();
	Barrier();

	VkRenderingAttachmentInfo color = {};
	color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	color.imageView = target.view;
	color.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	color.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
	color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	VkRenderingAttachmentInfo stencil = {};
	stencil.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	stencil.imageView = m_targets.stencil.view;
	stencil.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	stencil.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	stencil.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	VkRenderingInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	info.renderArea = ToRect(m_render_region);
	info.layerCount = 1;
	info.colorAttachmentCount = 1;
	info.pColorAttachments = &color;
	info.pStencilAttachment = layer ? &stencil : nullptr;
	vkCmdBeginRendering(m_commands, &info);
	m_rendering = true;
}

void RenderInterface_VKModern::EndRendering()
{
	if (!m_rendering)
		return;
	vkCmdEndRendering(m_commands);
	m_rendering = false;
	m_rendering_layer = -1;
	m_barrier_pending = true;
}

void RenderInterface_VKModern::BeginLayerRendering(Rml::LayerHandle layer, bool clear)
{
	BeginRendering(GetLayer(layer), true, clear);
	m_rendering_layer = int(layer);
}

void RenderInterface_VKModern::BindPipeline(Pipeline pipeline)
{
	if (pipeline == m_bound_pipeline)
		return;
	vkCmdBindPipeline(m_commands, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines[size_t(pipeline)]);
	m_bound_pipeline = pipeline;
}

void RenderInterface_VKModern::ApplyStencil(StencilState state)
{
	if (state.test_enable != m_stencil.test_enable)
		vkCmdSetStencilTestEnable(m_commands, state.test_enable);
	if (state.compare_op != m_stencil.compare_op || state.pass_op != m_stencil.pass_op)
		vkCmdSetStencilOp(m_commands, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, state.pass_op, VK_STENCIL_OP_KEEP, state.compare_op);
	if (state.reference != m_stencil.reference)
		vkCmdSetStencilReference(m_commands, VK_STENCIL_FACE_FRONT_AND_BACK, state.reference);
	m_stencil = state;
}

void RenderInterface_VKModern::ApplyScissor(Rml::Rectanglei region)
{
	const VkRect2D rect = ToRect(region);
	if (std::memcmp(&rect, &m_applied_scissor, sizeof(rect)) == 0)
		return;
	vkCmdSetScissor(m_commands, 0, 1, &rect);
	m_applied_scissor = rect;
}

RenderInterface_VKModern::StencilState RenderInterface_VKModern::ClipMaskTest() const
{
	StencilState state;
	state.test_enable = m_clip_mask_enabled;
	if (m_clip_mask_enabled && m_clip_mask_set)
	{
		state.compare_op = VK_COMPARE_OP_EQUAL;
		state.reference = m_clip_mask_value;
	}
	return state;
}

void RenderInterface_VKModern::DrawGeometry(Pipeline pipeline, Rml::CompiledGeometryHandle handle, Rml::Vector2f translation,
	const void* extra_constants, uint32_t extra_size)
{
	const Geometry& geometry = m_geometries[size_t(handle - 1)];
	if (geometry.num_indices == 0)
		return;

	if (m_rendering_layer != m_layer_count - 1)
		BeginLayerRendering(Rml::LayerHandle(m_layer_count - 1), false);
	BindPipeline(pipeline);
	ApplyScissor(m_scissor);

	constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	if (!m_transform_pushed)
	{
		const ShaderMatrix transform = ToShaderMatrix(m_transform);
		vkCmdPushConstants(m_commands, m_pipeline_layout, stages, 0, sizeof(transform), &transform);
		m_transform_pushed = true;
	}
	vkCmdPushConstants(m_commands, m_pipeline_layout, stages, TranslationOffset, sizeof(translation), &translation);
	if (extra_size > 0)
		vkCmdPushConstants(m_commands, m_pipeline_layout, stages, GeometryExtraOffset, extra_size, extra_constants);

	if (geometry.block != m_bound_geometry_block)
	{
		const GeometryBlock& block = m_geometry_blocks[geometry.block];
		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(m_commands, 0, 1, &block.buffer, &vertex_offset);
		vkCmdBindIndexBuffer(m_commands, block.buffer, block.index_offset, VK_INDEX_TYPE_UINT32);
		m_bound_geometry_block = geometry.block;
	}
	vkCmdDrawIndexed(m_commands, geometry.num_indices, 1, geometry.first_index, int32_t(geometry.first_vertex), 0);
}

void RenderInterface_VKModern::DrawQuad(Pipeline pipeline, const Quad& quad, const void* extra_constants, uint32_t extra_size)
{
	BindPipeline(pipeline);
	constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	vkCmdPushConstants(m_commands, m_pipeline_layout, stages, 0, sizeof(quad), &quad);
	if (extra_size > 0)
		vkCmdPushConstants(m_commands, m_pipeline_layout, stages, QuadExtraOffset, extra_size, extra_constants);
	// The quad constants overwrite the transform.
	m_transform_pushed = false;
	vkCmdDraw(m_commands, 4, 1, 0, 0);
}

void RenderInterface_VKModern::RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
	ApplyStencil(ClipMaskTest());
	uint32_t texture_constants[2] = {NoTexture, 0};
	if (texture)
	{
		const Image& image = GetTexture(texture);
		texture_constants[0] = image.texture_index;
		texture_constants[1] = (image.alpha_mode == AlphaMode::Straight ? FlagStraightAlpha : 0);
	}
	DrawGeometry(Pipeline::Geometry, handle, translation, texture_constants, sizeof(texture_constants));
}

void RenderInterface_VKModern::EnableScissorRegion(bool enable)
{
	// Enabling is followed by SetScissorRegion().
	if (!enable)
		m_scissor = m_render_region;
}

void RenderInterface_VKModern::SetScissorRegion(Rml::Rectanglei region)
{
	const Rml::Vector2i p0 = Rml::Math::Clamp(region.p0, m_render_region.p0, m_render_region.p1);
	const Rml::Vector2i p1 = Rml::Math::Clamp(region.p1, m_render_region.p0, m_render_region.p1);
	m_scissor = Rml::Rectanglei::FromCorners(p0, Rml::Math::Max(p0, p1));
}

void RenderInterface_VKModern::EnableClipMask(bool enable)
{
	m_clip_mask_enabled = enable;
}

void RenderInterface_VKModern::RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation)
{
	RMLUI_ASSERT(m_clip_mask_enabled);
	if (m_rendering_layer != m_layer_count - 1)
		BeginLayerRendering(Rml::LayerHandle(m_layer_count - 1), false);

	StencilState write;
	write.test_enable = true;
	write.reference = 1;
	switch (operation)
	{
	case Rml::ClipMaskOperation::Set:
	case Rml::ClipMaskOperation::SetInverse:
	{
		VkClearAttachment clear = {};
		clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
		clear.clearValue.depthStencil.stencil = 0;
		const VkClearRect rect = {ToRect(m_render_region), 0, 1};
		vkCmdClearAttachments(m_commands, 1, &clear, 1, &rect);
		write.pass_op = VK_STENCIL_OP_REPLACE;
		m_clip_mask_value = (operation == Rml::ClipMaskOperation::Set ? 1 : 0);
	}
	break;
	case Rml::ClipMaskOperation::Intersect:
		write.pass_op = VK_STENCIL_OP_INCREMENT_AND_CLAMP;
		m_clip_mask_value += 1;
		break;
	}

	ApplyStencil(write);
	DrawGeometry(Pipeline::ClipMask, geometry, translation, nullptr, 0);
	m_clip_mask_set = true;
}

void RenderInterface_VKModern::SetTransform(const Rml::Matrix4f* transform)
{
	m_transform = transform ? m_projection * (*transform) : m_projection;
	m_transform_pushed = false;
}

Rml::LayerHandle RenderInterface_VKModern::PushLayer()
{
	const Rml::LayerHandle layer = Rml::LayerHandle(m_layer_count++);
	BeginLayerRendering(layer, true);
	return layer;
}

void RenderInterface_VKModern::PopLayer()
{
	m_layer_count--;
}

void RenderInterface_VKModern::ResolveLayerToPostprocess(Rml::LayerHandle layer)
{
	const Image& source = GetLayer(layer);
	const Image& primary = GetPostprocess(Postprocess::Primary);
	EndRendering();
	Barrier();
	ResolveOrCopy(m_commands, source.image, primary.image, m_samples != VK_SAMPLE_COUNT_1_BIT, m_render_region);
	m_barrier_pending = true;
}

void RenderInterface_VKModern::CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode,
	Rml::Span<const Rml::CompiledFilterHandle> filters)
{
	ResolveLayerToPostprocess(source);
	RenderFilters(filters);

	BeginLayerRendering(destination, false);
	ApplyStencil(ClipMaskTest());
	ApplyScissor(m_scissor);
	Quad quad;
	quad.source_texture = GetPostprocess(Postprocess::Primary).texture_index;
	DrawQuad(blend_mode == Rml::BlendMode::Replace ? Pipeline::LayerReplace : Pipeline::LayerBlend, quad, nullptr, 0);
}

void RenderInterface_VKModern::RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filters)
{
	auto BeginPostprocess = [this](Postprocess target) {
		BeginRendering(GetPostprocess(target), false, false);
		ApplyStencil(StencilState());
		ApplyScissor(m_scissor);
	};
	auto SwapPrimarySecondary = [this] {
		std::swap(m_targets.postprocess[size_t(Postprocess::Primary)], m_targets.postprocess[size_t(Postprocess::Secondary)]);
	};

	for (const Rml::CompiledFilterHandle handle : filters)
	{
		const CompiledFilter& filter = *reinterpret_cast<const CompiledFilter*>(handle);
		switch (filter.type)
		{
		case FilterType::Passthrough:
		{
			BeginPostprocess(Postprocess::Secondary);
			if (m_blend_constant != filter.blend_factor)
			{
				const float constants[4] = {filter.blend_factor, filter.blend_factor, filter.blend_factor, filter.blend_factor};
				vkCmdSetBlendConstants(m_commands, constants);
				m_blend_constant = filter.blend_factor;
			}
			Quad quad;
			quad.source_texture = GetPostprocess(Postprocess::Primary).texture_index;
			DrawQuad(Pipeline::Opacity, quad, nullptr, 0);
			SwapPrimarySecondary();
		}
		break;
		case FilterType::Blur:
		{
			RenderBlur(filter.sigma, Postprocess::Primary, Postprocess::Secondary, m_scissor);
		}
		break;
		case FilterType::DropShadow:
		{
			const uint32_t primary = GetPostprocess(Postprocess::Primary).texture_index;
			BeginPostprocess(Postprocess::Secondary);

			DropShadowConstants constants;
			SetTexCoordLimits(constants.tex_coord_min, constants.tex_coord_max, m_scissor, m_targets.dimensions);
			constants.color = ConvertToColorf(filter.color);
			const Rml::Vector2f uv_offset = filter.offset / -Rml::Vector2f(m_targets.dimensions);
			Quad quad;
			quad.SetTexCoord(Rml::Rectanglef::FromCorners(uv_offset, uv_offset + Rml::Vector2f(1.f)));
			quad.source_texture = primary;
			DrawQuad(Pipeline::DropShadow, quad, &constants, sizeof(constants));

			if (filter.sigma >= 0.5f)
				RenderBlur(filter.sigma, Postprocess::Secondary, Postprocess::Tertiary, m_scissor);

			BeginPostprocess(Postprocess::Secondary);
			Quad composite;
			composite.source_texture = primary;
			DrawQuad(Pipeline::Blend, composite, nullptr, 0);
			SwapPrimarySecondary();
		}
		break;
		case FilterType::ColorMatrix:
		{
			BeginPostprocess(Postprocess::Secondary);
			const ShaderMatrix matrix = ToShaderMatrix(filter.color_matrix);
			Quad quad;
			quad.source_texture = GetPostprocess(Postprocess::Primary).texture_index;
			DrawQuad(Pipeline::ColorMatrix, quad, &matrix, sizeof(matrix));
			SwapPrimarySecondary();
		}
		break;
		case FilterType::MaskImage:
		{
			BeginPostprocess(Postprocess::Secondary);
			Quad quad;
			quad.source_texture = GetPostprocess(Postprocess::Primary).texture_index;
			quad.mask_texture = GetPostprocess(Postprocess::BlendMask).texture_index;
			DrawQuad(Pipeline::BlendMask, quad, nullptr, 0);
			SwapPrimarySecondary();
		}
		break;
		case FilterType::Invalid:
		{
			Rml::Log::Message(Rml::Log::LT_WARNING, "Unhandled render filter %d.", int(filter.type));
		}
		break;
		}
	}
}

void RenderInterface_VKModern::RenderBlur(float sigma, Postprocess source_destination, Postprocess temp, Rml::Rectanglei window)
{
	int pass_level = 0;
	SigmaToParameters(sigma, pass_level, sigma);
	if (sigma == 0)
		return;

	const Image& source_image = GetPostprocess(source_destination);
	const Image& temp_image = GetPostprocess(temp);
	const Rml::Vector2i dimensions = m_targets.dimensions;
	ApplyStencil(StencilState());

	// Downscale by iterative half-scaling with bilinear filtering, to reduce aliasing.
	const VkViewport half_viewport = {0.f, 0.f, dimensions.x / 2.f, dimensions.y / 2.f, 0.f, 1.f};
	vkCmdSetViewport(m_commands, 0, 1, &half_viewport);

	// Scale UVs if we have even dimensions, such that texture fetches align perfectly between texels, thereby producing a
	// 50% blend of neighboring texels.
	const Rml::Vector2f uv_scaling = {(dimensions.x % 2 == 1) ? (1.f - 1.f / float(dimensions.x)) : 1.f,
		(dimensions.y % 2 == 1) ? (1.f - 1.f / float(dimensions.y)) : 1.f};

	Rml::Rectanglei scissor = window;
	for (int i = 0; i < pass_level; i++)
	{
		scissor.p0 = (scissor.p0 + Rml::Vector2i(1)) / 2;
		scissor.p1 = Rml::Math::Max(scissor.p1 / 2, scissor.p0);
		const bool from_source = (i % 2 == 0);
		BeginRendering(from_source ? temp_image : source_image, false, false);
		ApplyScissor(scissor);
		Quad quad;
		quad.SetTexCoord(Rml::Rectanglef::FromSize(uv_scaling));
		quad.source_texture = (from_source ? source_image : temp_image).texture_index;
		DrawQuad(Pipeline::Copy, quad, nullptr, 0);
	}

	const VkViewport viewport = {0.f, 0.f, float(dimensions.x), float(dimensions.y), 0.f, 1.f};
	vkCmdSetViewport(m_commands, 0, 1, &viewport);

	// Ensure texture data end up in the temp buffer. Depending on the last downscaling, we might need to move it from the
	// source_destination buffer.
	if (pass_level % 2 == 0)
	{
		BeginRendering(temp_image, false, false);
		ApplyScissor(scissor);
		Quad quad;
		quad.source_texture = source_image.texture_index;
		DrawQuad(Pipeline::Copy, quad, nullptr, 0);
	}

	BlurConstants constants = {};
	SetBlurWeights(constants.weights, sigma);
	SetTexCoordLimits(constants.tex_coord_min, constants.tex_coord_max, scissor, dimensions);

	// Vertical pass.
	BeginRendering(source_image, false, false);
	ApplyScissor(scissor);
	constants.texel_offset = Rml::Vector2f(0.f, 1.f / float(dimensions.y));
	Quad quad;
	quad.source_texture = temp_image.texture_index;
	DrawQuad(Pipeline::Blur, quad, &constants, sizeof(constants));

	// Horizontal pass. Add a 1px transparent border around the blur region by first clearing with a padded scissor. This
	// helps prevent artifacts when upscaling the blur result in the later step.
	BeginRendering(temp_image, false, false);
	const Rml::Rectanglei extended = scissor.Extend(1);
	const Rml::Vector2i clear_p0 = Rml::Math::Clamp(extended.p0, m_render_region.p0, m_render_region.p1);
	const Rml::Vector2i clear_p1 = Rml::Math::Clamp(extended.p1, m_render_region.p0, m_render_region.p1);
	if (clear_p1.x > clear_p0.x && clear_p1.y > clear_p0.y)
	{
		VkClearAttachment clear = {};
		clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		const VkClearRect rect = {ToRect(Rml::Rectanglei::FromCorners(clear_p0, clear_p1)), 0, 1};
		vkCmdClearAttachments(m_commands, 1, &clear, 1, &rect);
	}
	ApplyScissor(scissor);
	constants.texel_offset = Rml::Vector2f(1.f / float(dimensions.x), 0.f);
	quad.source_texture = source_image.texture_index;
	DrawQuad(Pipeline::Blur, quad, &constants, sizeof(constants));

	// Upscale the blurred image to the window.
	BeginRendering(source_image, false, false);
	ApplyScissor(window);
	Quad upscale;
	upscale.SetPosition(PixelsToDevice(window));
	upscale.SetTexCoord(
		Rml::Rectanglef::FromCorners(Rml::Vector2f(scissor.p0) / Rml::Vector2f(dimensions), Rml::Vector2f(scissor.p1) / Rml::Vector2f(dimensions)));
	upscale.source_texture = temp_image.texture_index;
	DrawQuad(Pipeline::Copy, upscale, nullptr, 0);
}

Rml::TextureHandle RenderInterface_VKModern::SaveLayerAsTexture()
{
	const Rml::Rectanglei bounds = m_scissor;
	if (bounds.Width() <= 0 || bounds.Height() <= 0)
		return {};

	ResolveLayerToPostprocess(Rml::LayerHandle(m_layer_count - 1));
	const Image image =
		CreateImage(m_color_format, bounds.Size(), 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

	Barrier();
	VkImageCopy copy = {};
	copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy.srcOffset = {bounds.Left(), bounds.Top(), 0};
	copy.dstSubresource = copy.srcSubresource;
	copy.extent = {uint32_t(bounds.Width()), uint32_t(bounds.Height()), 1};
	vkCmdCopyImage(m_commands, GetPostprocess(Postprocess::Primary).image, VK_IMAGE_LAYOUT_GENERAL, image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
	m_barrier_pending = true;

	return RegisterTexture(image);
}

Rml::CompiledFilterHandle RenderInterface_VKModern::SaveLayerAsMaskImage()
{
	ResolveLayerToPostprocess(Rml::LayerHandle(m_layer_count - 1));

	BeginRendering(GetPostprocess(Postprocess::BlendMask), false, false);
	ApplyStencil(StencilState());
	ApplyScissor(m_scissor);
	Quad quad;
	quad.source_texture = GetPostprocess(Postprocess::Primary).texture_index;
	DrawQuad(Pipeline::Copy, quad, nullptr, 0);

	CompiledFilter filter = {};
	filter.type = FilterType::MaskImage;
	return reinterpret_cast<Rml::CompiledFilterHandle>(new CompiledFilter(std::move(filter)));
}

Rml::CompiledFilterHandle RenderInterface_VKModern::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters)
{
	CompiledFilter filter = {};

	if (name == "opacity")
	{
		filter.type = FilterType::Passthrough;
		filter.blend_factor = Rml::Get(parameters, "value", 1.0f);
	}
	else if (name == "blur")
	{
		filter.type = FilterType::Blur;
		filter.sigma = Rml::Get(parameters, "sigma", 1.0f);
	}
	else if (name == "drop-shadow")
	{
		filter.type = FilterType::DropShadow;
		filter.sigma = Rml::Get(parameters, "sigma", 0.f);
		filter.color = Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
		filter.offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.f));
	}
	else if (name == "brightness")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
	}
	else if (name == "contrast")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float grayness = 0.5f - 0.5f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(grayness, grayness, grayness, 1.f));
	}
	else if (name == "invert")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.f, 1.f);
		const float inverted = 1.f - 2.f * value;
		filter.color_matrix = Rml::Matrix4f::Diag(inverted, inverted, inverted, 1.f);
		filter.color_matrix.SetColumn(3, Rml::Vector4f(value, value, value, 1.f));
	}
	else if (name == "grayscale")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f gray = value * Rml::Vector3f(0.2126f, 0.7152f, 0.0722f);
		// clang-format off
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{gray.x + rev_value, gray.y,             gray.z,             0.f},
			{gray.x,             gray.y + rev_value, gray.z,             0.f},
			{gray.x,             gray.y,             gray.z + rev_value, 0.f},
			{0.f,                0.f,                0.f,                1.f}
		);
		// clang-format on
	}
	else if (name == "sepia")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float rev_value = 1.f - value;
		const Rml::Vector3f r_mix = value * Rml::Vector3f(0.393f, 0.769f, 0.189f);
		const Rml::Vector3f g_mix = value * Rml::Vector3f(0.349f, 0.686f, 0.168f);
		const Rml::Vector3f b_mix = value * Rml::Vector3f(0.272f, 0.534f, 0.131f);
		// clang-format off
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{r_mix.x + rev_value, r_mix.y,             r_mix.z,             0.f},
			{g_mix.x,             g_mix.y + rev_value, g_mix.z,             0.f},
			{b_mix.x,             b_mix.y,             b_mix.z + rev_value, 0.f},
			{0.f,                 0.f,                 0.f,                 1.f}
		);
		// clang-format on
	}
	else if (name == "hue-rotate")
	{
		// Hue-rotation and saturation values based on: https://www.w3.org/TR/filter-effects-1/#attr-valuedef-type-huerotate
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		const float s = Rml::Math::Sin(value);
		const float c = Rml::Math::Cos(value);
		// clang-format off
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{0.213f + 0.787f * c - 0.213f * s,  0.715f - 0.715f * c - 0.715f * s,  0.072f - 0.072f * c + 0.928f * s,  0.f},
			{0.213f - 0.213f * c + 0.143f * s,  0.715f + 0.285f * c + 0.140f * s,  0.072f - 0.072f * c - 0.283f * s,  0.f},
			{0.213f - 0.213f * c - 0.787f * s,  0.715f - 0.715f * c + 0.715f * s,  0.072f + 0.928f * c + 0.072f * s,  0.f},
			{0.f,                               0.f,                               0.f,                               1.f}
		);
		// clang-format on
	}
	else if (name == "saturate")
	{
		filter.type = FilterType::ColorMatrix;
		const float value = Rml::Get(parameters, "value", 1.0f);
		// clang-format off
		filter.color_matrix = Rml::Matrix4f::FromRows(
			{0.213f + 0.787f * value,  0.715f - 0.715f * value,  0.072f - 0.072f * value,  0.f},
			{0.213f - 0.213f * value,  0.715f + 0.285f * value,  0.072f - 0.072f * value,  0.f},
			{0.213f - 0.213f * value,  0.715f - 0.715f * value,  0.072f + 0.928f * value,  0.f},
			{0.f,                      0.f,                      0.f,                      1.f}
		);
		// clang-format on
	}

	if (filter.type != FilterType::Invalid)
		return reinterpret_cast<Rml::CompiledFilterHandle>(new CompiledFilter(std::move(filter)));

	Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported filter type '%s'.", name.c_str());
	return {};
}

void RenderInterface_VKModern::ReleaseFilter(Rml::CompiledFilterHandle filter)
{
	delete reinterpret_cast<CompiledFilter*>(filter);
}

Rml::CompiledShaderHandle RenderInterface_VKModern::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
	auto ApplyColorStopList = [](CompiledShader& shader, const Rml::Dictionary& shader_parameters) {
		auto it = shader_parameters.find("color_stop_list");
		RMLUI_ASSERT(it != shader_parameters.end() && it->second.GetType() == Rml::Variant::COLORSTOPLIST);
		const Rml::ColorStopList& color_stop_list = it->second.GetReference<Rml::ColorStopList>();
		const int num_stops = Rml::Math::Min((int)color_stop_list.size(), MAX_NUM_STOPS);

		shader.stop_positions.resize(num_stops);
		shader.stop_colors.resize(num_stops);
		for (int i = 0; i < num_stops; i++)
		{
			const Rml::ColorStop& stop = color_stop_list[i];
			RMLUI_ASSERT(stop.position.unit == Rml::Unit::NUMBER);
			shader.stop_positions[i] = stop.position.number;
			shader.stop_colors[i] = ConvertToColorf(stop.color);
		}
	};

	CompiledShader shader = {};

	if (name == "linear-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingLinear : ShaderGradientFunction::Linear);
		shader.p = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
		shader.v = Rml::Get(parameters, "p1", Rml::Vector2f(0.f)) - shader.p;
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "radial-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingRadial : ShaderGradientFunction::Radial);
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		shader.v = Rml::Vector2f(1.f) / Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "conic-gradient")
	{
		shader.type = CompiledShaderType::Gradient;
		const bool repeating = Rml::Get(parameters, "repeating", false);
		shader.gradient_function = (repeating ? ShaderGradientFunction::RepeatingConic : ShaderGradientFunction::Conic);
		shader.p = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
		const float angle = Rml::Get(parameters, "angle", 0.f);
		shader.v = {Rml::Math::Cos(angle), Rml::Math::Sin(angle)};
		ApplyColorStopList(shader, parameters);
	}
	else if (name == "shader")
	{
		const Rml::String value = Rml::Get(parameters, "value", Rml::String());
		if (value == "creation")
		{
			shader.type = CompiledShaderType::Creation;
			shader.dimensions = Rml::Get(parameters, "dimensions", Rml::Vector2f(0.f));
		}
	}

	if (shader.type != CompiledShaderType::Invalid)
		return reinterpret_cast<Rml::CompiledShaderHandle>(new CompiledShader(std::move(shader)));

	Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.", name.c_str());
	return {};
}

void RenderInterface_VKModern::RenderShader(Rml::CompiledShaderHandle shader_handle, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
	Rml::TextureHandle /*texture*/)
{
	const CompiledShader& shader = *reinterpret_cast<const CompiledShader*>(shader_handle);
	ApplyStencil(ClipMaskTest());

	switch (shader.type)
	{
	case CompiledShaderType::Gradient:
	{
		const StagingAllocation staging = AllocateStaging(sizeof(GradientData));
		GradientData data = {};
		data.func = int(shader.gradient_function);
		data.num_stops = int(shader.stop_positions.size());
		data.p = shader.p;
		data.v = shader.v;
		std::copy(shader.stop_colors.begin(), shader.stop_colors.end(), data.stop_colors);
		std::copy(shader.stop_positions.begin(), shader.stop_positions.end(), data.stop_positions);
		std::memcpy(staging.mapped, &data, sizeof(data));

		const uint32_t extra[4] = {NoTexture, 0, uint32_t(staging.address), uint32_t(staging.address >> 32)};
		DrawGeometry(Pipeline::Gradient, geometry, translation, extra, sizeof(extra));
	}
	break;
	case CompiledShaderType::Creation:
	{
		struct {
			uint32_t texture_index;
			uint32_t flags;
			CreationConstants creation;
		} extra = {NoTexture, 0, {shader.dimensions, float(Rml::GetSystemInterface()->GetElapsedTime())}};
		DrawGeometry(Pipeline::Creation, geometry, translation, &extra, sizeof(extra));
	}
	break;
	case CompiledShaderType::Invalid:
	{
		Rml::Log::Message(Rml::Log::LT_WARNING, "Unhandled render shader %d.", int(shader.type));
	}
	break;
	}
}

void RenderInterface_VKModern::ReleaseShader(Rml::CompiledShaderHandle shader)
{
	delete reinterpret_cast<CompiledShader*>(shader);
}

bool RenderInterface_VKModern::CaptureScreen(int& width, int& height, int& num_components, Rml::UniquePtr<Rml::byte[]>& data)
{
	if (m_frame_counter == 0 || m_targets.layers.empty())
		return false;

	const Rml::Rectanglei region = m_render_region;
	const VkDeviceSize size = VkDeviceSize(region.Width()) * VkDeviceSize(region.Height()) * 4;

	VkBufferCreateInfo buffer_info = {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = size;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info = {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VmaAllocationInfo mapped;
	Check(vmaCreateBuffer(m_allocator, &buffer_info, &allocation_info, &buffer, &allocation, &mapped), "vmaCreateBuffer");

	// The base layer holds the last frame until the next BeginFrame(). If multisampled, first resolve it into a temporary
	// image.
	const bool multisampled = (m_samples != VK_SAMPLE_COUNT_1_BIT);
	VkImage resolved = VK_NULL_HANDLE;
	VmaAllocation resolved_allocation = VK_NULL_HANDLE;
	if (multisampled)
	{
		VkImageCreateInfo image_info = {};
		image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image_info.imageType = VK_IMAGE_TYPE_2D;
		image_info.format = m_color_format;
		image_info.extent = {uint32_t(region.Width()), uint32_t(region.Height()), 1};
		image_info.mipLevels = 1;
		image_info.arrayLayers = 1;
		image_info.samples = VK_SAMPLE_COUNT_1_BIT;
		image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		VmaAllocationCreateInfo image_allocation = {};
		image_allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		Check(vmaCreateImage(m_allocator, &image_info, &image_allocation, &resolved, &resolved_allocation, nullptr), "vmaCreateImage");
	}
	const VkImage source = multisampled ? resolved : m_targets.layers[0].image;

	VkCommandPoolCreateInfo pool_info = {};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	pool_info.queueFamilyIndex = m_queue_family;
	VkCommandPool pool = VK_NULL_HANDLE;
	Check(vkCreateCommandPool(m_device, &pool_info, nullptr, &pool), "vkCreateCommandPool");
	VkCommandBufferAllocateInfo allocate = {};
	allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate.commandPool = pool;
	allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate.commandBufferCount = 1;
	VkCommandBuffer commands = VK_NULL_HANDLE;
	Check(vkAllocateCommandBuffers(m_device, &allocate, &commands), "vkAllocateCommandBuffers");

	VkCommandBufferBeginInfo begin = {};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	Check(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
	FullBarrier(commands);
	if (multisampled)
	{
		ImageBarrier(commands, resolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE,
			VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
		ResolveOrCopy(commands, m_targets.layers[0].image, resolved, true, region);
		FullBarrier(commands);
	}
	VkBufferImageCopy copy = {};
	copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
	copy.imageOffset = {multisampled ? 0 : region.Left(), multisampled ? 0 : region.Top(), 0};
	copy.imageExtent = {uint32_t(region.Width()), uint32_t(region.Height()), 1};
	vkCmdCopyImageToBuffer(commands, source, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &copy);
	Check(vkEndCommandBuffer(commands), "vkEndCommandBuffer");

	VkCommandBufferSubmitInfo command_info = {};
	command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
	command_info.commandBuffer = commands;
	VkSubmitInfo2 submit = {};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
	submit.commandBufferInfoCount = 1;
	submit.pCommandBufferInfos = &command_info;
	{
		std::lock_guard<std::mutex> lock(m_queue_mutex);
		Check(vkQueueSubmit2(m_queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit2");
		Check(vkQueueWaitIdle(m_queue), "vkQueueWaitIdle");
	}
	vkDestroyCommandPool(m_device, pool, nullptr);
	if (multisampled)
		vmaDestroyImage(m_allocator, resolved, resolved_allocation);

	width = region.Width();
	height = region.Height();
	num_components = 3;
	data.reset(new Rml::byte[size_t(width) * size_t(height) * 3]);
	const bool bgra = (m_color_format == VK_FORMAT_B8G8R8A8_UNORM);
	const Rml::byte* pixels = static_cast<const Rml::byte*>(mapped.pMappedData);
	for (int y = 0; y < height; y++)
	{
		const Rml::byte* source_row = pixels + size_t(height - 1 - y) * size_t(width) * 4;
		Rml::byte* target_row = data.get() + size_t(y) * size_t(width) * 3;
		for (int x = 0; x < width; x++)
		{
			const Rml::byte* pixel = source_row + x * 4;
			target_row[x * 3 + 0] = pixel[bgra ? 2 : 0];
			target_row[x * 3 + 1] = pixel[1];
			target_row[x * 3 + 2] = pixel[bgra ? 0 : 2];
		}
	}

	vmaDestroyBuffer(m_allocator, buffer, allocation);
	return true;
}
