#version 460

// A rectangle drawn as a four-vertex triangle strip, without vertex buffers.
layout(push_constant) uniform Constants {
	vec4 positionRect; // normalized device coordinates: min.xy, max.xy
	vec4 texCoordRect; // min.xy, max.xy
};

layout(location = 0) out vec2 fragTexCoord;

void main() {
	const vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
	fragTexCoord = mix(texCoordRect.xy, texCoordRect.zw, corner);
	gl_Position = vec4(mix(positionRect.xy, positionRect.zw, corner), 0.0, 1.0);
}
