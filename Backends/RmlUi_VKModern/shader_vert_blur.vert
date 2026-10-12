#version 460
#extension GL_GOOGLE_include_directive : require
#include "shader_blur.glsl"

layout(location = 0) out vec2 fragTexCoord[BLUR_SIZE];

void main() {
	const vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
	const vec2 texCoord = mix(texCoordRect.xy, texCoordRect.zw, corner);
	for (int i = 0; i < BLUR_SIZE; i++)
		fragTexCoord[i] = texCoord - float(i - BLUR_NUM_WEIGHTS + 1) * texelOffset;
	gl_Position = vec4(mix(positionRect.xy, positionRect.zw, corner), 0.0, 1.0);
}
