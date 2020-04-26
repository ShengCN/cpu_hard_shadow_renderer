#include <iostream>
#include "common.h"
#include "mesh.h"
#include "model_loader.h"
#include "ppc.h"

int w = 256, h = 256;

bool load_mesh(const std::string fname, std::shared_ptr<mesh> &m) {
	auto loader = model_loader::create(fname);
	return loader->load_model(fname, m);
}

struct image {
	std::vector<unsigned int> pixels;
	image(int w, int h) {
		pixels.resize(w * h);
	}

	void clear() {
		pixels.clear();
	}

	void set_pixel(vec3 p) {
		//todo
	}
};

struct ground_plane {
	vec3 p, n;
};

void raster_hard_shadow(const ground_plane& plane, std::shared_ptr<mesh>& target, ppc cur_ppc, vec3 light_pos, image& out_img) {
	out_img.clear();

	// iterate over the output image
		// compute the intersection point with the plane
		// compute if it's hit by the light
		// set value for this pixel
}

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(int x, int y, int w = 512, int h = 256) {
	//todo
}

void render_data(const std::string model_file, const std::string output_folder) {
	std::shared_ptr<mesh> render_target;
	glm::vec3 light_position(0.0f);

	// load render target
	if (load_mesh(model_file, render_target)) {
		std::cerr << "Loading success \n";
	}
	else {
		std::cerr << "Loading failed \n";
	}
	
	// normalize and compute ground plane
	render_target->normalize_position_orientation();
	AABB world_aabb = render_target->compute_world_aabb();
	vec3 target_center = render_target->compute_world_center();

	vec3 foot_position = world_aabb.p0;
	ground_plane cur_plane = { foot_position, vec3(0.0f,1.0f,0.0f) };
	
	// set camera position
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(w, h, 65.0f);
	float mesh_length = world_aabb.diag_length();
	if (mesh_length < 0.1f)
		mesh_length = 5.0f;
	vec3 new_pos = target_center + 2.0f * vec3(0.0f, mesh_length * 0.3f, mesh_length);
	cur_ppc->PositionAndOrient(new_pos, target_center, vec3(0.0f, 1.0f, 0.0f));

	// rasterize the 256x256 image to compute hard shadow
	std::string test_output = output_folder + "/" + "test.png";
	image out_img = image(w, h);
	vec3 light_position(0.0f, 100.0f, 100.0f);
	raster_hard_shadow(cur_plane, render_target, *cur_ppc, light_position,out_img);
}

void main() {
	std::string testing_model = "E:/ds/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj";
	
	render_data(testing_model, "output");
	system("pause");
}