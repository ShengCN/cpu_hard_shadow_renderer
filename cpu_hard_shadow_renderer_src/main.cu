#include <iostream>
#include <glm/mat3x3.hpp>
#include <glm/gtx/transform.hpp>
#include "common.h"
#include "mesh.h"
#include "model_loader.h"
#include "ppc.h"

int w = 256, h = 256;
using namespace purdue;
int i_begin = 256 - (60.0 / 90.0) * 256 * 0.5;

struct pixel_pos {
	float x, y;

	std::string to_string() {
		std::ostringstream oss;
		oss << x << "," << y;
		return oss.str();
	}
};

std::vector<pixel_pos> ibl_map;

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
		size_t ind = (h - 1 - v) * w + u;
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[0] = (unsigned char)(255.0 * p.x);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[1] = (unsigned char)(255.0 * p.y);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[2] = (unsigned char)(255.0 * p.z);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[3] = (unsigned char)(255);
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

__host__ __device__
vec3 plane_ppc_intersect(const plane& plane, const ray& cur_ray) {
	float t = glm::dot(plane.p - cur_ray.ro, plane.n) / glm::dot(cur_ray.rd, plane.n);
	return cur_ray.ro + cur_ray.rd * t;
}

__host__ __device__
bool ray_triangle_intersect(const ray& r, vec3 p0, vec3 p1, vec3 p2) {
	vec3 v0v1 = p1 - p0;
	vec3 v0v2 = p2 - p0;
	vec3 pvec = glm::cross(r.rd, v0v2);
	float det = glm::dot(v0v1, pvec);

	constexpr float kEpsilon = 1e-8;
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < kEpsilon) return false;
	float invDet = 1 / det;
	vec3 tvec = r.ro - p0;
	float u = glm::dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	vec3 qvec = glm::cross(tvec, v0v1);
	float v = glm::dot(r.rd, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;
	return true;

}

__host__ __device__
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

__host__ __device__
bool hit_light(vec3 p, glm::vec3* world_verts, AABB& aabb, vec3 light_pos) {
	ray r = { p, light_pos - p };

	if (!ray_aabb_intersect(r, aabb))
		return false;

	return true;
	/*bool ret = false;
	volatile bool flag = false;
	for (int ti = 0; ti < world_verts.size() / 3; ++ti) {
		if (flag) continue;
		vec3 p0 = world_verts[3 * ti + 0];
		vec3 p1 = world_verts[3 * ti + 1];
		vec3 p2 = world_verts[3 * ti + 2];

		if (ray_triangle_intersect(r, p0, p1, p2)) {
			{flag = true; ret = true; }
		}
	}

	return ret;*/
}

__host__ __device__
void set_pixel(vec3 c, unsigned int& p) {
	reinterpret_cast<unsigned char*>(&p)[0] = (unsigned char)(255.0 * c.x);
	reinterpret_cast<unsigned char*>(&p)[1] = (unsigned char)(255.0 * c.y);
	reinterpret_cast<unsigned char*>(&p)[2] = (unsigned char)(255.0 * c.z);
	reinterpret_cast<unsigned char*>(&p)[3] = (unsigned char)(255);
}

__global__
void render(plane& ground_plane, glm::vec3* world_verts_cuda, int N, ppc& cur_ppc, vec3& light_pos, AABB& aabb, unsigned int* pixel) {
	int idj = blockDim.x * blockIdx.x + threadIdx.x;
	int idi = blockDim.y * blockIdx.y + threadIdx.y;

	int j_stride = gridDim.x * blockDim.x;
	int i_stride = gridDim.y * blockDim.y;

	// iterate over the output image
	for (int j = idj; j < cur_ppc._height; j += j_stride) {
		for (int i = idi; i < cur_ppc._width; i += i_stride) {
			// compute the intersection point with the plane
			ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
			vec3 intersect_pos = plane_ppc_intersect(ground_plane, cur_ray);

			vec3 pixel_value(1.0f);
			// compute if it's hit by the light
			if (hit_light(intersect_pos, world_verts_cuda, aabb, light_pos)) {
				pixel_value = vec3(0.0f);
			}

			set_pixel(pixel_value, pixel[(cur_ppc._height - 1 - j)*cur_ppc._width + i]);
		}

		// std::cerr << "finish: " << (float)j / cur_ppc._height * 100.0f << "%" << "\r";
	}
}

void render_d(plane& ground_plane, glm::vec3* world_verts_cuda, int N, ppc& cur_ppc, vec3& light_pos, AABB& aabb, unsigned int* pixel) {
	// iterate over the output image
	for (int j = 0; j < cur_ppc._height; ++j) {
		for (int i = 0; i < cur_ppc._width; ++i) {
			// compute the intersection point with the plane
			ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
			vec3 intersect_pos = plane_ppc_intersect(ground_plane, cur_ray);

			vec3 pixel_value(1.0f);
			// compute if it's hit by the light
			if (hit_light(intersect_pos, world_verts_cuda, aabb, light_pos)) {
				pixel_value = vec3(0.0f);
			}

			set_pixel(pixel_value, pixel[(cur_ppc._height - 1 - j)*cur_ppc._width + i]);
		}
	}
}

void raster_hard_shadow(plane& grond_plane, std::shared_ptr<mesh>& target, ppc cur_ppc, vec3 light_pos, image& out_img) {
	out_img.clear();

	auto world_verts = target->compute_world_space_coords();
	AABB aabb = target->compute_world_aabb();

	glm::vec3* world_verts_cuda;
	unsigned int* pixels;
	cudaMallocManaged(&world_verts_cuda, world_verts.size() * sizeof(glm::vec3));
	cudaMallocManaged(&pixels, out_img.pixels.size() * sizeof(unsigned int));
	cudaMemcpy(world_verts_cuda, world_verts.data(), world_verts.size() * sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(pixels, out_img.pixels.data(), out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 grid(256, 256);
	dim3 block(256, 256, 256);
	// render <<<grid, block >>> (grond_plane, world_verts_cuda, world_verts.size(), cur_ppc, light_pos, aabb, pixels);
	render_d(grond_plane, world_verts_cuda, world_verts.size(), cur_ppc, light_pos, aabb, pixels);

	cudaDeviceSynchronize();
	cudaMemcpy(pixels, out_img.pixels.data(), out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(world_verts_cuda);
	cudaFree(pixels);
}

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(int x, int y, int w = 512, int h = 256) {
	//todo
	float x_fract = (float)x / w, y_fract = (float)y / h;
	deg alpha, beta;
	alpha = x_fract * 360.0f - 90.0f; beta = y_fract * 180.0f - 90.0f;
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

	vec3 lowest_point = world_aabb.p0;
	float offset = 0.0f - lowest_point.y;
	render_target->m_world = glm::translate(vec3(0.0, offset, 0.0)) * render_target->m_world;

	plane cur_plane = { vec3(0.0f), vec3(0.0f, 1.0f, 0.0f) };

	// set camera position
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(w, h, 65.0f);
	float mesh_length = world_aabb.diag_length();
	if (mesh_length < 0.1f)
		mesh_length = 5.0f;
	vec3 new_pos = target_center + 2.0f * vec3(0.0f, mesh_length * 0.3f, mesh_length);
	cur_ppc->PositionAndOrient(new_pos, lowest_point, vec3(0.0f, 1.0f, 0.0f));

	// rasterize the 256x256 image to compute hard shadow
	purdue::create_folder(output_folder);
	image out_img(w, h);

	int camera_pitch_num = 3;
	int target_rotation_num = 6;

	mat4 old_mat = render_target->m_world;
	ppc  old_ppc = *cur_ppc;

	vec3 render_target_center = render_target->compute_world_center();
	float render_target_size = render_target->compute_world_aabb().diag_length();
	float light_relative_length = 11.0f;
	vec3 ppc_relative = vec3(0.0, 0.0, render_target_size) * 2.0f;
	float delta_target_rot = 360.0 / target_rotation_num;
	float delta_camera_pitch = 30.0 / camera_pitch_num;
	std::vector<std::string> gt_str;

	timer t;
	t.tic();

	int counter = 0;
	int total_counter = target_rotation_num * camera_pitch_num * ibl_map.size();
	for (int trni = 0; trni < target_rotation_num; ++trni) {
		float target_rot = lerp(0.0, 360.0, (float)trni / target_rotation_num);
		// set rotation
		render_target->m_world = glm::rotate(deg2rad(target_rot), glm::vec3(0.0, 1.0, 0.0)) * render_target->m_world;

		for (int cpni = 0; cpni < camera_pitch_num; ++cpni) {
			float camera_pitch = 15.0 + lerp(0.0, 30.0, (float)cpni / camera_pitch_num);
			// set camera rotation
			cur_ppc->PositionAndOrient(glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * ppc_relative + render_target_center,
				render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
				vec3(0.0, 1.0, 0.0));

			for (auto &light_pixel_pos : ibl_map) {
				vec3 light_position = compute_light_pos(light_pixel_pos.x, light_pixel_pos.y) * light_relative_length + render_target_center;

				raster_hard_shadow(cur_plane, render_target, *cur_ppc, light_position, out_img);

				char buff[100];
				snprintf(buff, sizeof(buff), "%07d", counter++);
				std::string cur_prefix = buff;

				std::ostringstream oss;
				oss << cur_prefix << ",";
				oss << light_pixel_pos.to_string() << ",";
				oss << to_string(cur_ppc->_position) << ",";
				oss << target_rot << ",";
				oss << to_string(render_target_center) << ",";
				oss << to_string(light_position) << std::endl;
				gt_str.push_back(oss.str());

				std::string output_fname = output_folder + "/" + cur_prefix + "_shadow.png";
				out_img.save(output_fname);

				std::cerr << "Finish: " << (float)counter / total_counter * 100.0f << "% \r";

				std::string cmd = "start " + output_fname;
				system(cmd.c_str());

				return;
			}
		}

		// set back rotation
		render_target->m_world = old_mat;
	}
	std::ofstream output(output_folder + "/ground_truth.txt");
	if (output.is_open()) {
		for (auto &s : gt_str) {
			output << s;
		}
	}
	t.toc();
	t.print_elapsed();
	std::string total_time = t.to_string();
	output << total_time << std::endl;
	output.close();
}

int main(int argc, char *argv[]) {
	// std::string testing_model = "E:/ds/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj";
	if (argc != 3) {
		std::cerr << "Please check your input! \n";
		std::cerr << "Should be xx model_path out_folder \n";
		return 0;
	}

	for (int i = i_begin; i < 256; ++i) for (int j = 0; j < 512; ++j) {
		ibl_map.push_back({ (float)j, (float)i });
	}

	std::string model_file = argv[1];
	std::string output_folder = argv[2];

	model_file = "E:/ds/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A/notsimulated_combine_male_short_outfits_genesis8_armani_casualoutfit03_Base_Pose_Standing_A.obj";
	output_folder = "./output/";

	render_data(model_file, output_folder);

	printf("finished \n");
	return 0;
}
