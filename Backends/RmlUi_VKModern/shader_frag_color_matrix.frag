#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "shader_textures.glsl"

layout(push_constant) uniform Constants {
	layout(offset = 32) uint sourceTexture;
	layout(offset = 48) mat4 colorMatrix;
};

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 finalColor;

void main() {
	vec4 texColor = SAMPLE(sourceTexture, fragTexCoord);
	finalColor = vec4((colorMatrix * texColor).rgb, texColor.a);
}
