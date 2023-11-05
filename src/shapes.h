#pragma once
#include <glm/glm.hpp>
#include <vector>

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
};

struct Mesh
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

enum Shape {
	Sphere,
	Plane
};

namespace shapes {

	struct Sphere { constexpr static Shape shape = Shape::Sphere; };
	struct Plane { constexpr static Shape shape = Shape::Plane; };
	
	Mesh createMesh(shapes::Sphere sphere);

}