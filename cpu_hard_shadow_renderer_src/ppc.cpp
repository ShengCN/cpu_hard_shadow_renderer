#include "ppc.h"
#include <sstream>
#include <glm/common.hpp>
#include <glm/gtx/transform.hpp>

using namespace glm;
using namespace purdue;
float ppc::GetFocal() {
	return static_cast<float>(_width / 2) / tan(_fov / 2.0f);
}

ppc::ppc(int w, int h, float fov, float p_near, float p_far) :
	_width(w),
	_height(h),
	_fov(fov),
	_near(p_near),
	_far(p_far),
	_position(0.0f, 0.35f, 1.3f), 
	_worldUp(0.0f, 1.0f, 0.0f) {
}

ppc::~ppc()
{
}

vec3 ppc::GetRight() const {
	vec3 view = GetViewVec();

	return cross(view, _worldUp);
}

vec3 ppc::GetUp() const {
	return cross(GetRight(), GetViewVec());
}

void ppc::PositionAndOrient(vec3 p, vec3 lookatP, vec3 up)
{
	_position = p;
	_front = glm::normalize(lookatP - p);
	_worldUp = up;
}

glm::mat4 ppc::GetP() const {
	return glm::perspective(deg2rad(_fov), (float)_width / (float)_height, _near, _far);
}

glm::mat4 ppc::GetV() const {
	return glm::lookAt(_position, _position + _front, _worldUp);
	// return glm::lookAt(_position, target, _worldUp);
}

void ppc::Rotate_Axis(glm::vec3 O, glm::vec3 axis, float angled)
{
	glm::vec3 CO = _position - O;
	auto rot_mat = glm::rotate(deg2rad(angled), normalize(axis));
	CO = vec3(rot_mat * vec4(CO, 0.0f));
	_position = CO + O;

}

void ppc::Zoom(float delta)
{
	_fov += delta;
	_fov = glm::clamp(_fov, 10.0f, 150.0f);
}

void ppc::Keyboard(CameraMovement cm, float speed)
{
	switch (cm)
	{
	case CameraMovement::forward:
		_position += speed * _front;
		break;
	case CameraMovement::backward:

		_position -= speed * _front;
		break;
	case CameraMovement::left:

		_position -= speed * GetRight();
		break;
	case CameraMovement::right:

		_position += speed * GetRight();
		break;
	case CameraMovement::up:
		_position += speed * GetUp();
		break;
	case CameraMovement::down:
		_position -= speed * GetUp();
		break;
	default:
		break;
	}
}

void ppc::pan(double deg)
{
	deg = deg / _width * 10.0f;
	//glm::mat4 rot_y = glm::rotate(deg2rad(deg), glm::vec3(0.0f, 1.0f, 0.0f));
	//_front = glm::vec3(rot_y * glm::vec4(_front, 0.0f));
	vec3 right = GetRight();
	_front += right * float(deg);
	_front = glm::normalize(_front);
}

void ppc::tilt(double deg) {
	// #TODO
}

void ppc::pitch(double deg)
{
	deg = deg /_height * 10.0f;
	//glm::mat4 rot_x = glm::rotate(deg2rad(deg), glm::vec3(1.0f, 0.0f, 0.0f));
	//_front = glm::vec3(rot_x * glm::vec4(_front, 0.0f));
	vec3 up = GetUp();
	_front += up * float(deg);
	_front = glm::normalize(_front);
}

void ppc::get_ray(int u, int v, vec3& ro, vec3& rd) const {
	float focal = get_focal();
	vec3 right = glm::normalize(GetRight());
	vec3 up = glm::normalize(GetUp());
	vec3 front = glm::normalize(GetViewVec());
	
	ro = _position;
	int center_u = _width / 2, center_v = _height / 2;

	rd = front * focal + (u - center_u + 0.5f) * right + (v - center_v + 0.5f) * up;
	rd = glm::normalize(rd);
}

float ppc::get_focal() const {
	return 0.5f * _height / std::tan(deg2rad(0.5f * _fov));
}

std::string ppc::to_string() {
	std::ostringstream oss;
	oss << "w: " << _width << " h: " << _height<< std::endl;
	oss << "look: " << purdue::to_string(_position) << std::endl;
	oss << "at: " << purdue::to_string(_position + _front) << std::endl;
	oss << "up: " << purdue::to_string(GetUp()) << std::endl;
	oss << "fov: " << std::to_string(_fov) << std::endl;

	return oss.str();
}
