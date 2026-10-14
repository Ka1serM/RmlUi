#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <mutex>

// Maximum MSAA sample count of the layers, the highest supported count up to this value is used.
#ifndef RMLUI_RENDER_BACKEND_FIELD_MSAA_SAMPLE_COUNT
	#define RMLUI_RENDER_BACKEND_FIELD_MSAA_SAMPLE_COUNT 2
#endif

// Present mode of the swapchain, falls back to FIFO if not supported by the surface.
#ifndef RMLUI_RENDER_BACKEND_FIELD_PRESENT_MODE
	#define RMLUI_RENDER_BACKEND_FIELD_PRESENT_MODE VK_PRESENT_MODE_FIFO_KHR
#endif

#if defined _MSC_VER
	#pragma warning(push, 0)
#elif defined __clang__
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wall"
	#pragma clang diagnostic ignored "-Wextra"
	#pragma clang diagnostic ignored "-Wnullability-extension"
	#pragma clang diagnostic ignored "-Wnullability-completeness"
#elif defined __GNUC__
	#pragma GCC system_header
#endif

#include "RmlUi_VKModern/vulkan.h"
// Always include "vulkan.h" first, this comment prevents clang-format from reordering the includes.
#include "RmlUi_Vulkan/vk_mem_alloc.h"

#if defined _MSC_VER
	#pragma warning(pop)
#elif defined __clang__
	#pragma clang diagnostic pop
#endif

/*
 * Vulkan 1.3 renderer backend, supporting the full RmlUi feature set: clip masks, layers, filters, and shaders.
 *
 * Compared to RmlUi_Renderer_VK, it relies on modern Vulkan features to stay small:
 *  - Dynamic rendering, synchronization2, and dynamic state: no render passes, no framebuffers, and few pipelines.
 *  - All images stay in the general layout, only swapchain images are transitioned for presentation.
 *  - Bindless textures: a single descriptor table, indexed through push constants.
 *  - Per-draw data is passed as push constants, gradients are read through a buffer device address.
 *  - Geometry is suballocated from large shared buffers, texture uploads are recorded into the frame submission.
 *  - Layers are multisampled with up to RMLUI_RENDER_BACKEND_FIELD_MSAA_SAMPLE_COUNT samples, and shaded per sample
 *    when supported.
 */
class RenderInterface_VKModern : public Rml::RenderInterface {
public:
	using CreateSurfaceCallback = bool (*)(VkInstance instance, VkSurfaceKHR* out_surface);

	enum class AlphaMode { Premultiplied, Straight };

	// A single mip level of a texture in its stored format, tightly packed.
	struct TextureMip {
		Rml::Vector2i dimensions;
		Rml::Span<const Rml::byte> data;
	};

	RenderInterface_VKModern();
	~RenderInterface_VKModern();

	// The optional device extensions are enabled when the device supports them. The device has one queue in each queue
	// family, for applications that submit their own work.
	bool Initialize(Rml::Vector<const char*> required_extensions, CreateSurfaceCallback create_surface_callback,
		Rml::Vector<const char*> optional_device_extensions = {});
	void Shutdown();

	// The viewport should be updated whenever the window size changes. This recreates the swapchain without waiting for
	// the device. While the window has zero area, such as when minimized, the swapchain is invalid.
	void SetViewport(int width, int height);
	// Returns false when there is no swapchain to render to, in which case BeginFrame() must not be called.
	bool IsSwapchainValid();
	void RecreateSwapchain();

	// Sets up the render state for taking rendering commands from RmlUi.
	void BeginFrame();
	// Submits the frame and presents it to the screen.
	void EndFrame();

	// Captures the last rendered frame as RGB data, with the rows ordered bottom-up.
	bool CaptureScreen(int& width, int& height, int& num_components, Rml::UniquePtr<Rml::byte[]>& data);

	// Creates a texture from data in any format the device can sample, such as block-compressed formats. Mips are ordered
	// from largest to smallest. Returns zero if the format is not supported.
	Rml::TextureHandle CreateTexture(VkFormat format, Rml::Span<const TextureMip> mips, AlphaMode alpha_mode);

	// Creates a texture from an image that the application has created and filled. The renderer takes ownership of the image
	// and its view, and destroys them when the texture is released. The image must be allocated with GetAllocator(), use
	// premultiplied alpha, and be in the general layout. Commands writing to the image must be submitted to GetQueue() before
	// the frame that renders the texture, followed by a barrier making the writes visible to fragment shader reads.
	Rml::TextureHandle RegisterVulkanTexture(VkImage image, VmaAllocation allocation, VkImageView view);

	// Vulkan objects of the renderer, for applications that create their own resources and submit their own commands.
	VkInstance GetInstance() const;
	VkPhysicalDevice GetPhysicalDevice() const;
	VkDevice GetDevice() const;
	VkQueue GetQueue() const;
	uint32_t GetQueueFamily() const;
	VmaAllocator GetAllocator() const;
	const Rml::Vector<const char*>& GetDeviceExtensions() const;
	// The features enabled on the device, with the Vulkan 1.1, 1.2, and 1.3 features chained.
	const VkPhysicalDeviceFeatures2& GetDeviceFeatures() const;
	// Guards every queue of the device. The renderer locks it around its own queue operations, and applications lock it
	// when they use a queue from another thread.
	std::mutex& GetQueueMutex();

	// -- Inherited from Rml::RenderInterface --

	Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
	void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override;
	void ReleaseGeometry(Rml::CompiledGeometryHandle handle) override;

	Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
	Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions) override;
	void ReleaseTexture(Rml::TextureHandle texture_handle) override;

	void EnableScissorRegion(bool enable) override;
	void SetScissorRegion(Rml::Rectanglei region) override;

	void EnableClipMask(bool enable) override;
	void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;

	void SetTransform(const Rml::Matrix4f* transform) override;

	Rml::LayerHandle PushLayer() override;
	void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode,
		Rml::Span<const Rml::CompiledFilterHandle> filters) override;
	void PopLayer() override;

	Rml::TextureHandle SaveLayerAsTexture() override;
	Rml::CompiledFilterHandle SaveLayerAsMaskImage() override;

	Rml::CompiledFilterHandle CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) override;
	void ReleaseFilter(Rml::CompiledFilterHandle filter) override;

	Rml::CompiledShaderHandle CompileShader(const Rml::String& name, const Rml::Dictionary& parameters) override;
	void RenderShader(Rml::CompiledShaderHandle shader, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
		Rml::TextureHandle texture) override;
	void ReleaseShader(Rml::CompiledShaderHandle shader) override;

private:
	// Also used as the requested number of swapchain images.
	static constexpr uint32_t FramesInFlight = 3;
	static constexpr uint32_t NoTexture = 0xFFFFFFFF;

	enum class Pipeline {
		// Layer pipelines, multisampled and with a stencil attachment.
		Geometry,
		Gradient,
		Creation,
		ClipMask,
		LayerBlend,
		LayerReplace,
		// Postprocess pipelines, single-sampled and without a stencil attachment.
		Copy,
		Blend,
		Opacity,
		ColorMatrix,
		BlendMask,
		Blur,
		DropShadow,
		Count
	};

	struct Image {
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;
		// Slot in the bindless texture table, or NoTexture for images that are not sampled.
		uint32_t texture_index = NoTexture;
		AlphaMode alpha_mode = AlphaMode::Premultiplied;
	};

	struct Geometry {
		uint32_t block = 0;
		uint32_t first_vertex = 0;
		uint32_t first_index = 0;
		uint32_t num_indices = 0;
		VmaVirtualAllocation vertices = VK_NULL_HANDLE;
		VmaVirtualAllocation indices = VK_NULL_HANDLE;
	};

	// A host-visible buffer with vertices in the first part and indices after them. Suballocated in units of vertices and
	// indices, so that draws can address their geometry using firstIndex and vertexOffset. Empty blocks are released,
	// except for the first one, leaving a slot without a buffer.
	struct GeometryBlock {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		Rml::byte* mapped = nullptr;
		VkDeviceSize index_offset = 0;
		VmaVirtualBlock vertices = VK_NULL_HANDLE;
		VmaVirtualBlock indices = VK_NULL_HANDLE;
	};

	struct StagingBuffer {
		VkBuffer buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		Rml::byte* mapped = nullptr;
		VkDeviceAddress address = 0;
		VkDeviceSize size = 0;
	};

	struct StagingAllocation {
		VkBuffer buffer;
		VkDeviceSize offset;
		Rml::byte* mapped;
		VkDeviceAddress address;
	};

	struct Frame {
		VkCommandPool command_pool = VK_NULL_HANDLE;
		VkCommandBuffer commands = VK_NULL_HANDLE;
		// Texture uploads and image initialization, submitted before the frame commands.
		VkCommandBuffer uploads = VK_NULL_HANDLE;
		bool uploads_recording = false;
		VkFence fence = VK_NULL_HANDLE;
		VkSemaphore image_acquired = VK_NULL_HANDLE;
		// Upload and per-draw data of this frame. Only the first buffer is kept between frames, if it has the standard size.
		Rml::Vector<StagingBuffer> staging;
		VkDeviceSize staging_used = 0;
	};

	struct Swapchain {
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		VkExtent2D extent = {};
		Rml::Vector<VkImage> images;
		// One per swapchain image, since a presentation may wait on it until the image is acquired again.
		Rml::Vector<VkSemaphore> render_finished;
	};

	struct StencilState {
		bool test_enable = false;
		VkCompareOp compare_op = VK_COMPARE_OP_ALWAYS;
		VkStencilOp pass_op = VK_STENCIL_OP_KEEP;
		uint32_t reference = 0;
	};

	// All render targets share the same dimensions: the swapchain extent rounded up, see RenderTargetGranularity.
	struct RenderTargets {
		Rml::Vector2i dimensions;
		// Multisampled color images of the layer stack, allocated on demand.
		Rml::Vector<Image> layers;
		Image stencil;
		// Single-sampled images, see Postprocess, allocated on demand.
		Rml::Array<Image, 4> postprocess;
	};

	enum class Postprocess { Primary, Secondary, Tertiary, BlendMask };

	// Push constants of a rectangle drawn with shader_vert_quad or shader_vert_blur, covering the whole target by default.
	struct Quad {
		float position[4] = {-1.f, -1.f, 1.f, 1.f};
		float tex_coord[4] = {0.f, 0.f, 1.f, 1.f};
		uint32_t source_texture = NoTexture;
		uint32_t mask_texture = NoTexture;

		void SetPosition(Rml::Rectanglef rect) { Assign(position, rect); }
		void SetTexCoord(Rml::Rectanglef rect) { Assign(tex_coord, rect); }

	private:
		static void Assign(float (&target)[4], Rml::Rectanglef rect)
		{
			target[0] = rect.p0.x;
			target[1] = rect.p0.y;
			target[2] = rect.p1.x;
			target[3] = rect.p1.y;
		}
	};

	bool CreateInstance(Rml::Vector<const char*> required_extensions);
	bool ChoosePhysicalDevice();
	bool CreateDevice(const Rml::Vector<const char*>& optional_extensions);
	bool ChooseFormats();
	void CreateDescriptors();
	void CreatePipelines();
	void CreateFrames();
	void CreateSwapchain(VkExtent2D extent, const VkSurfaceCapabilitiesKHR& capabilities);
	void RetireSwapchain();

	Frame& PrepareFrame();
	void CollectGarbage(bool everything);
	VkCommandBuffer UploadCommands();
	StagingAllocation AllocateStaging(VkDeviceSize size);

	Image CreateImage(VkFormat format, Rml::Vector2i dimensions, uint32_t mip_levels, VkSampleCountFlagBits samples, VkImageUsageFlags usage);
	void DestroyImage(Image& image);
	void DestroyGeometryBlock(GeometryBlock& block);
	void AddToTextureTable(Image& image);
	Rml::TextureHandle RegisterTexture(Image image);
	const Image& GetTexture(Rml::TextureHandle handle) const;
	Rml::TextureHandle UploadTexture(VkFormat format, Rml::Span<const TextureMip> mips, AlphaMode alpha_mode);

	void EnsureRenderTargets();
	const Image& GetLayer(Rml::LayerHandle layer);
	const Image& GetPostprocess(Postprocess postprocess);

	void BeginRendering(const Image& target, bool layer, bool clear);
	void EndRendering();
	void BeginLayerRendering(Rml::LayerHandle layer, bool clear);
	void Barrier();
	void BindPipeline(Pipeline pipeline);
	void ApplyStencil(StencilState state);
	void ApplyScissor(Rml::Rectanglei region);
	StencilState ClipMaskTest() const;
	void DrawGeometry(Pipeline pipeline, Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, const void* extra_constants,
		uint32_t extra_size);
	void DrawQuad(Pipeline pipeline, const Quad& quad, const void* extra_constants, uint32_t extra_size);

	void ResolveLayerToPostprocess(Rml::LayerHandle layer);
	void RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filters);
	void RenderBlur(float sigma, Postprocess source_destination, Postprocess temp, Rml::Rectanglei window);
	Rml::Rectanglef PixelsToDevice(Rml::Rectanglei region) const;

	VkInstance m_instance = VK_NULL_HANDLE;
	VkSurfaceKHR m_surface = VK_NULL_HANDLE;
	VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
	VkDevice m_device = VK_NULL_HANDLE;
	VkQueue m_queue = VK_NULL_HANDLE;
	uint32_t m_queue_family = 0;
	VmaAllocator m_allocator = VK_NULL_HANDLE;
	Rml::Vector<const char*> m_device_extensions;
	VkPhysicalDeviceFeatures2 m_features = {};
	VkPhysicalDeviceVulkan11Features m_features11 = {};
	VkPhysicalDeviceVulkan12Features m_features12 = {};
	VkPhysicalDeviceVulkan13Features m_features13 = {};
	std::mutex m_queue_mutex;

	VkFormat m_color_format = VK_FORMAT_UNDEFINED;
	VkColorSpaceKHR m_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	VkFormat m_stencil_format = VK_FORMAT_UNDEFINED;
	VkSampleCountFlagBits m_samples = VK_SAMPLE_COUNT_1_BIT;
	VkPresentModeKHR m_present_mode = VK_PRESENT_MODE_FIFO_KHR;
	bool m_sample_shading = false;

	VkSampler m_sampler = VK_NULL_HANDLE;
	VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool m_descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet m_descriptor_set = VK_NULL_HANDLE;
	VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
	Rml::Array<VkPipeline, size_t(Pipeline::Count)> m_pipelines = {};

	uint32_t m_texture_capacity = 0;
	uint32_t m_next_texture_index = 0;
	Rml::Vector<uint32_t> m_free_texture_indices;
	// Indexed by bindless slot, a texture handle is its slot plus one.
	Rml::Vector<Image> m_textures;

	Rml::Vector<GeometryBlock> m_geometry_blocks;
	// A geometry handle is its index plus one.
	Rml::Vector<Geometry> m_geometries;
	Rml::Vector<uint32_t> m_free_geometries;

	Rml::Array<Frame, FramesInFlight> m_frames;
	// Number of submitted frames, used to stamp released resources.
	uint64_t m_frame_counter = 0;
	bool m_frame_prepared = false;

	Swapchain m_swapchain;
	VkExtent2D m_requested_extent = {};
	// Swapchain image of the current frame, or UINT32_MAX if none could be acquired.
	uint32_t m_image_index = UINT32_MAX;

	RenderTargets m_targets;
	Rml::Matrix4f m_projection;

	// Resources released while the GPU may still be using them, stamped with the frame counter.
	Rml::Vector<Rml::Pair<uint64_t, Image>> m_garbage_images;
	Rml::Vector<Rml::Pair<uint64_t, Geometry>> m_garbage_geometries;
	Rml::Vector<Rml::Pair<uint64_t, Swapchain>> m_garbage_swapchains;

	// -- Recording state of the current frame --
	VkCommandBuffer m_commands = VK_NULL_HANDLE;
	// The swapchain extent, located at the top-left of the render targets.
	Rml::Rectanglei m_render_region;
	int m_layer_count = 0;
	// The layer currently being rendered to, or -1 for postprocess images or no active rendering.
	int m_rendering_layer = -1;
	bool m_rendering = false;
	bool m_barrier_pending = false;
	Pipeline m_bound_pipeline = Pipeline::Count;
	uint32_t m_bound_geometry_block = UINT32_MAX;
	StencilState m_stencil;
	VkRect2D m_applied_scissor = {};
	float m_blend_constant = -1.f;
	bool m_transform_pushed = false;

	Rml::Matrix4f m_transform;
	// The scissor region in pixels, clamped to the render region. Covers the render region when scissoring is disabled.
	Rml::Rectanglei m_scissor;
	bool m_clip_mask_enabled = false;
	bool m_clip_mask_set = false;
	uint32_t m_clip_mask_value = 0;
};
