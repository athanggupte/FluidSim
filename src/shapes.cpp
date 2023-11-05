#include "shapes.h"
#include <glm/gtc/constants.hpp>

Mesh shapes::createMesh(Sphere sphere)
{
	int nlat = 12, nlong = 12;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// top vertex
	vertices.push_back({ {0, 1, 0}, {0, 1, 0} });

	// generate vertices
	for (int i = 0; i < nlat; i++) {
		auto phi = glm::pi<float>() * static_cast<float>(i + 1) / static_cast<float>(nlat);
		for (int j = 0; j < nlong; j++) {
			auto theta = 2.f * glm::pi<float>() * static_cast<float>(j) / static_cast<float>(nlong);
			auto x = glm::sin(phi) * glm::cos(theta);
			auto y = glm::cos(phi);
			auto z = glm::sin(phi) * glm::sin(theta);
			vertices.push_back({ {x, y, z}, {x, y, z}});
		}
	}

	// bottom vertex
	vertices.push_back({ {0, -1, 0}, {0, -1, 0} });

	uint32_t iTop = 0, iBot = vertices.size() - 1;

	// top & bottom triangles
	for (int i = 0; i < nlong; i++) {
		indices.push_back(iTop);
		indices.push_back((i + 1) % nlong + 1);
		indices.push_back(i + 1);

		indices.push_back(iBot);
		indices.push_back(i + nlong * (nlat - 2) + 1);
		indices.push_back((i + 1) % nlong + nlong * (nlat - 2) + 1);
	}

	// generate quads
	for (int j = 0; j < nlat - 2; j++) {
		auto j0 = j * nlong + 1;
		auto j1 = (j + 1) * nlong + 1;
		for (int i = 0; i < nlong; i++) {
			auto i0 = j0 + i;
			auto i1 = j0 + (i + 1) % nlong;
			auto i2 = j1 + (i + 1) % nlong;
			auto i3 = j1 + i;
			indices.push_back(i0);indices.push_back(i2);indices.push_back(i3);
			indices.push_back(i2);indices.push_back(i0);indices.push_back(i1);
		}
	}

	return { .vertices = vertices, .indices = indices };
}
