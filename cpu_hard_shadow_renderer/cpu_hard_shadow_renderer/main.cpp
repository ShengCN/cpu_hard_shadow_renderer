#include <iostream>
#include <glm/mat3x3.hpp>
#include "omp.h"

#include "common.h"
#include "mesh.h"
#include "model_loader.h"
#include "ppc.h"

int w = 256, h = 256;
using namespace purdue;

bool load_mesh(const std::string fname, std::shared_ptr<mesh> &m) {
	auto loader = model_loader::create(fname);
	return loader->load_model(fname, m);
}

struct image {
	std::vector<unsigned int> pixels;
	int w, h;
	image(int w, int h) :w(w), h(h) {
		pixels.resize(w * h);
	}

	void clear() {
		pixels.resize(w * h);
	}

	void set_pixel(int u, int v, vec3 p) {
		//todo
		size_t ind = (h - 1 - v) * w + u;
		reinterpret_cast<char*>(&(pixels.at(ind)))[0] = (char)(255.9 * p.x);
		reinterpret_cast<char*>(&(pixels.at(ind)))[1] = (char)(255.9 * p.y);
		reinterpret_cast<char*>(&(pixels.at(ind)))[2] = (char)(255.9 * p.z);
		reinterpret_cast<char*>(&(pixels.at(ind)))[3] = (char)(255);
	}

	bool save(const std::string fname) {
		return save_image(fname, pixels.data(), w, h);
	}
};

struct ray {
	vec3 ro, rd;
};

struct plane {
	vec3 p, n;
};

vec3 plane_ppc_intersect(const plane& plane, const ray& cur_ray) {
	float t = glm::dot(plane.p - cur_ray.ro, plane.n) / glm::dot(cur_ray.rd, plane.n);
	return cur_ray.ro + cur_ray.rd * t;
}

bool ray_triangle_intersect(const ray& r, vec3 p0, vec3 p1, vec3 p2) {
	plane tri_plane = { p0, glm::cross(p1 - p0, p2 - p0) };
	vec3 plane_intersect = plane_ppc_intersect(tri_plane, r);
	glm::mat3x3 bary_m(p0 - r.ro, p1 - r.ro, p2 - r.ro); bary_m = bary_m * 10.0f;
	glm::vec3 bary_coord = glm::inverse(bary_m) * (plane_intersect - r.ro) * 10.0f;

	auto inside = [](float t) {
		return t > 0.0f + 1e-3f && t < 1.0f-1e-3f;
	};

	return inside(bary_coord.x) && inside(bary_coord.y) && inside(bary_coord.z);
}

bool ray_aabb_intersect(const ray&r, const AABB& aabb) {
	float tmin = (aabb.p0.x - r.ro.x) / r.rd.x;
	float tmax = (aabb.p1.x - r.ro.x) / r.rd.x;

	if (tmin > tmax) std::swap(tmin, tmax);

	float tymin = (aabb.p0.y - r.ro.y) / r.rd.y;
	float tymax = (aabb.p1.y - r.ro.y) / r.rd.y;

	if (tymin > tymax) std::swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (aabb.p0.z - r.ro.z) / r.rd.z;
	float tzmax = (aabb.p1.z - r.ro.z) / r.rd.z;

	if (tzmin > tzmax) std::swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return true;
}

bool hit_light(vec3 p, std::vector<glm::vec3> &world_verts, AABB& aabb, vec3 light_pos) {
	ray r = { p, light_pos - p };

	if (!ray_aabb_intersect(r, aabb))
		return false;

	bool ret = false;
#pragma omp parallel for
	for (int ti = 0; ti < world_verts.size() / 3; ++ti) {
		vec3 p0 = world_verts[3 * ti + 0];
		vec3 p1 = world_verts[3 * ti + 1];
		vec3 p2 = world_verts[3 * ti + 2];

		if (ray_triangle_intersect(r, p0, p1, p2)) {
#pragma omp critical 
			{ret = true; }
			break;
		}
	}

	return ret;
}

void raster_hard_shadow(const plane& grond_plane, std::shared_ptr<mesh>& target, ppc cur_ppc, vec3 light_pos, image& out_img) {
	out_img.clear();

	auto world_verts = target->compute_world_space_coords();
	AABB aabb = target->compute_world_aabb();

	// iterate over the output image
	for (int j = 0; j < cur_ppc._height; ++j) {
		std::cerr << "finish: " << (float)j / cur_ppc._height * 100.0f << "%" << "\r";
		for (int i = 0; i < cur_ppc._width; ++i) {
			// compute the intersection point with the plane
			ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
			vec3 intersect_pos = plane_ppc_intersect(grond_plane, cur_ray);

			vec3 pixel_value(0.0f);
			// compute if it's hit by the light
			if (hit_light(intersect_pos, world_verts, aabb, light_pos)){
				pixel_value = vec3(1.0f);
			}

			{out_img.set_pixel(i, j, pixel_value); }
		}
	}
}

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(int x, int y, int w = 512, int h = 256) {
	//todo
	float x_fract = (float)x / w, y_fract = (float)y / h;
	deg alpha, beta;
	alpha = x_fract * 360.0f; beta = y_fract * 180.0f - 90.0f;
	return vec3(cos(deg2rad(beta)) * cos(deg2rad(alpha)), sin(deg2rad(beta)), cos(deg2rad(beta)) * sin(deg2rad(alpha)));
}

void render_data(const std::string model_file, const std::string output_folder) {
	std::shared_ptr<mesh> render_target;

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
	plane cur_plane = { foot_position, vec3(0.0f,1.0f,0.0f) };
	
	// set camera position
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(w, h, 65.0f);
	float mesh_length = world_aabb.diag_length();
	if (mesh_length < 0.1f)
		mesh_length = 5.0f;
	vec3 new_pos = target_center + 2.0f * vec3(0.0f, mesh_length * 0.3f, mesh_length);
	cur_ppc->PositionAndOrient(new_pos, target_center, vec3(0.0f, 1.0f, 0.0f));

	// rasterize the 256x256 image to compute hard shadow
	purdue::create_folder(output_folder);
	std::string test_output = output_folder + "/" + "test.png";
	image out_img(w, h);

	glm::vec3 light_position = compute_light_pos(256, 170);
	timer t;
	t.tic();
	raster_hard_shadow(cur_plane, render_target, *cur_ppc, light_position, out_img);
	t.toc();
	t.print_elapsed();

	out_img.save(test_output);
}

void main() {
	std::string testing_model = "E:/ds/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj";
	
	render_data(testing_model, "output");
	
	printf("finished \n");
}