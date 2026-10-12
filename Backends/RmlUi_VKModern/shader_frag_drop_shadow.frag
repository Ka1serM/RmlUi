#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "shader_textures.glsl"

layout(push_constant) uniform Constants {
	layout(offset = 32) uint sourceTexture;
	layout(offset = 48) vec2 texCoordMin;
	vec2 texCoordMax;
	vec4 color;
};

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 finalColor;

void main() {
	vec2 inRegion = step(texCoordMin, fragTexCoord) * step(fragTexCoord, texCoordMax);
	finalColor = SAMPLE(sourceTexture, fragTexCoord).a * inRegion.x * inRegion.y * color;
}
