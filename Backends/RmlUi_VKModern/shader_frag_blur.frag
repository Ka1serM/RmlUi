#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "shader_textures.glsl"
#include "shader_blur.glsl"

layout(location = 0) in vec2 fragTexCoord[BLUR_SIZE];

layout(location = 0) out vec4 finalColor;

void main() {
	vec4 color = vec4(0.0);
	for (int i = 0; i < BLUR_SIZE; i++) {
		vec2 inRegion = step(texCoordMin, fragTexCoord[i]) * step(fragTexCoord[i], texCoordMax);
		color += SAMPLE(sourceTexture, fragTexCoord[i]) * inRegion.x * inRegion.y * weights[abs(i - BLUR_NUM_WEIGHTS + 1)];
	}
	finalColor = color;
}
