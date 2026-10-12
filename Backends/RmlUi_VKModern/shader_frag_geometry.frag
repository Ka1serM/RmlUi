#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "shader_textures.glsl"

#define NO_TEXTURE 0xFFFFFFFFu
#define FLAG_STRAIGHT_ALPHA 1u

layout(push_constant) uniform Constants {
	layout(offset = 72) uint textureIndex;
	uint flags;
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 finalColor;

void main() {
	vec4 color = fragColor;
	if (textureIndex != NO_TEXTURE) {
		vec4 texel = SAMPLE(textureIndex, fragTexCoord);
		if ((flags & FLAG_STRAIGHT_ALPHA) != 0u)
			texel.rgb *= texel.a;
		color *= texel;
	}
	finalColor = color;
}
