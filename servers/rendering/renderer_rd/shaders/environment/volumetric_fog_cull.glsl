#[compute]

#version 450

#VERSION_DEFINES

// Finds, for each column of froxels, how many of its slices can be seen from
// the camera. A froxel behind every pixel of its column in the depth buffer is
// hidden, so volumetric fog doesn't need to compute it.

#define BLOCK_SIZE 8

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in;

layout(push_constant, std430) uniform Params {
	// The third and fourth rows of the inverse projection matrix, which give the
	// view space z and w of a pixel.
	vec4 inv_projection_z;
	vec4 inv_projection_w;

	ivec2 screen_size;
	ivec2 fog_size;

	int fog_depth;
	float fog_inv_length;
	float inv_detail_spread;
	int dilate_radius;
}
params;

#ifdef MODE_REDUCE

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;
layout(r32ui, set = 0, binding = 1) uniform restrict writeonly uimage2D column_slices;

uint get_slices_needed(ivec2 pixel, float depth) {
	if (depth >= 1.0) {
		// The pixel is at the near plane, for instance because a depth occluder
		// wrote to it, so only the first slice can be in front of it.
		return 1u;
	}
	if (depth <= 0.0) {
		// Nothing was drawn here, so the fog is seen all the way to its end.
		return uint(params.fog_depth);
	}
	vec4 ndc = vec4((vec2(pixel) + 0.5) / vec2(params.screen_size) * 2.0 - 1.0, depth, 1.0);
	float view_distance = -dot(params.inv_projection_z, ndc) / dot(params.inv_projection_w, ndc);
	// The same mapping as volumetric_fog_process() in the scene shader.
	float unit_depth = pow(clamp(view_distance * params.fog_inv_length, 0.0, 1.0), params.inv_detail_spread);
	// The slice the pixel is in, and the next one, which linear filtering also
	// reads.
	return uint(min(int(unit_depth * float(params.fog_depth)) + 2, params.fog_depth));
}

// Each column is reduced by COLUMN_INVOCATIONS x COLUMN_INVOCATIONS
// invocations, each taking every COLUMN_INVOCATIONSth pixel. A column covers
// hundreds of pixels at high resolutions, too many to read one after another in
// one invocation.
#define COLUMN_INVOCATIONS 4
#define GROUP_COLUMNS (BLOCK_SIZE / COLUMN_INVOCATIONS)

shared uint invocation_slices[BLOCK_SIZE * BLOCK_SIZE];

void main() {
	ivec2 local = ivec2(gl_LocalInvocationID.xy);
	ivec2 column = ivec2(gl_WorkGroupID.xy) * GROUP_COLUMNS + local / COLUMN_INVOCATIONS;
	ivec2 offset = local % COLUMN_INVOCATIONS;
	bool in_grid = all(lessThan(column, params.fog_size));

	uint slices = 0;
	if (in_grid) {
		ivec2 from = (column * params.screen_size) / params.fog_size;
		ivec2 to = min(((column + 1) * params.screen_size + params.fog_size - 1) / params.fog_size, params.screen_size);

		for (int y = from.y + offset.y; y < to.y && slices < uint(params.fog_depth); y += COLUMN_INVOCATIONS) {
			for (int x = from.x + offset.x; x < to.x; x += COLUMN_INVOCATIONS) {
				ivec2 pixel = ivec2(x, y);
				slices = max(slices, get_slices_needed(pixel, texelFetch(depth_buffer, pixel, 0).r));
			}
		}
	}

	invocation_slices[gl_LocalInvocationIndex] = slices;
	memoryBarrierShared();
	barrier();

	if (in_grid && all(equal(offset, ivec2(0)))) {
		for (int y = 0; y < COLUMN_INVOCATIONS; y++) {
			for (int x = 0; x < COLUMN_INVOCATIONS; x++) {
				slices = max(slices, invocation_slices[(local.y + y) * BLOCK_SIZE + local.x + x]);
			}
		}
		imageStore(column_slices, column, uvec4(slices));
	}
}

#endif

#ifdef MODE_DILATE

layout(r32ui, set = 0, binding = 0) uniform restrict readonly uimage2D column_slices;
layout(r32ui, set = 0, binding = 1) uniform restrict writeonly uimage2D dilated_column_slices;

void main() {
	// Fog is filtered and sampled across neighboring columns, so a froxel is
	// needed if it is visible in any column within the radius.
	ivec2 column = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(column, params.fog_size))) {
		return;
	}

	ivec2 from = max(column - params.dilate_radius, ivec2(0));
	ivec2 to = min(column + params.dilate_radius, params.fog_size - 1);

	uint slices = 0;
	for (int y = from.y; y <= to.y; y++) {
		for (int x = from.x; x <= to.x; x++) {
			slices = max(slices, imageLoad(column_slices, ivec2(x, y)).x);
		}
	}

	imageStore(dilated_column_slices, column, uvec4(slices));
}

#endif
