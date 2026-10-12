#define BLUR_SIZE 7
#define BLUR_NUM_WEIGHTS ((BLUR_SIZE + 1) / 2)

layout(push_constant) uniform Constants {
	vec4 positionRect;
	vec4 texCoordRect;
	uint sourceTexture;
	layout(offset = 48) vec2 texelOffset;
	vec2 texCoordMin;
	vec2 texCoordMax;
	layout(offset = 80) vec4 weights;
};
