// The bindless texture table, indexed by push constants.
layout(set = 0, binding = 0) uniform texture2D textures[];
layout(set = 0, binding = 1) uniform sampler linearSampler;

#define SAMPLE(index, uv) texture(sampler2D(textures[index], linearSampler), uv)
