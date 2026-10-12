#version 460
#extension GL_EXT_buffer_reference : require

#define MAX_NUM_STOPS 16
#define LINEAR 0
#define RADIAL 1
#define CONIC 2
#define REPEATING_LINEAR 3
#define REPEATING_RADIAL 4
#define REPEATING_CONIC 5
#define PI 3.14159265

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Gradient {
	int func;
	int numStops;
	vec2 p; // linear: starting point,         radial: center,                        conic: center
	vec2 v; // linear: vector to ending point, radial: 2d curvature (inverse radius), conic: angled unit vector
	vec4 stopColors[MAX_NUM_STOPS];
	float stopPositions[MAX_NUM_STOPS]; // normalized, 0 -> starting point, 1 -> ending point
};

layout(push_constant) uniform Constants {
	layout(offset = 80) Gradient gradient;
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 finalColor;

vec4 lerpStopColors(float t) {
	vec4 color = gradient.stopColors[0];
	for (int i = 1; i < gradient.numStops; i++)
		color = mix(color, gradient.stopColors[i], smoothstep(gradient.stopPositions[i - 1], gradient.stopPositions[i], t));
	return color;
}

void main() {
	const int func = gradient.func;
	const vec2 p = gradient.p;
	const vec2 v = gradient.v;
	float t = 0.0;
	if (func == LINEAR || func == REPEATING_LINEAR) {
		t = dot(v, fragTexCoord - p) / dot(v, v);
	} else if (func == RADIAL || func == REPEATING_RADIAL) {
		t = length(v * (fragTexCoord - p));
	} else if (func == CONIC || func == REPEATING_CONIC) {
		mat2 R = mat2(v.x, -v.y, v.y, v.x);
		vec2 V = R * (fragTexCoord - p);
		t = 0.5 + atan(-V.x, V.y) / (2.0 * PI);
	}
	if (func == REPEATING_LINEAR || func == REPEATING_RADIAL || func == REPEATING_CONIC) {
		float t0 = gradient.stopPositions[0];
		float t1 = gradient.stopPositions[gradient.numStops - 1];
		t = t0 + mod(t - t0, t1 - t0);
	}
	finalColor = fragColor * lerpStopColors(t);
}
