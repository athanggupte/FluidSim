#pragma once
#include <glm/ext/scalar_constants.hpp>
#include <cmath>

#include "render.h"

// Simulation constants

struct SimulationConfig
{
	size_t NumParticles{};
	uint3  ThreadGridDim{};
	uint3  ThreadBlockDim{};
	size_t NumBlocks{};
	size_t NumThreadsPerBlock{};
	size_t NumParticlesPerWarp{};
	size_t NumWarps{};

	float3 Gravity{};
	float  SpeedOfSound{};
	float  ReferenceDensity{};
	float  TaitExponent{};
	float  ReferencePressure{};
	float  Viscosity{};
	float  CollisionDamping{};

	float  SmoothingRadius{};
	float  ParticleRadius{};
	float  ParticleMass{};

	float3 Region{};
	float3 RegionHalf{};
	float3 InitRegion{};
	float3 InitPosition{};

	float3 CellSize{};
	uint3  CellGridDim{};
	size_t NumCells{};

	size_t RadixSortBlockSize{};

	float PhysicsDeltaTime{};
	float RenderFPS{};
	float RenderDeltaTime{};
};

inline void InitializeSimulationConfig(SimulationConfig& config, cudaDeviceProp const& device_prop)
{
	config.NumParticles        = 27000;
	config.ThreadGridDim       = uint3(3, 3, 3); // gridDim
	config.ThreadBlockDim      = uint3(10, 10, 10); // blockDim
	config.NumBlocks           = static_cast<size_t>(config.ThreadGridDim.x) * config.ThreadGridDim.y;
	config.NumThreadsPerBlock  = static_cast<size_t>(config.ThreadBlockDim.x) * config.ThreadBlockDim.y * config.ThreadBlockDim.z;
	config.NumParticlesPerWarp = config.NumBlocks * config.NumThreadsPerBlock;
	config.NumWarps            = config.NumParticles / config.NumParticlesPerWarp + (config.NumParticles % config.NumParticlesPerWarp ? 1 : 0);

	config.Gravity             = float3(0, -9.81f, 0);
	config.SpeedOfSound        = 330.f;
	config.ReferenceDensity    = 1.f;
	config.TaitExponent        = 7;
	config.ReferencePressure   = 100;// config.ReferenceDensity * config.SpeedOfSound / config.TaitExponent;
	config.Viscosity           = 0.001f;
	config.CollisionDamping    = 0.85f;

	config.SmoothingRadius     = 0.1f;
	config.ParticleRadius      = 0.025f;

	//float h = config.SmoothingRadius;
	//float volume = (1.f/12.f * (h-4) * ((h-5)*h + 10) * h*h*h + 4.f * h) * glm::pi<float>();
	float volume = 4.f / 3.f * glm::pi<float>() * config.SmoothingRadius * config.SmoothingRadius * config.SmoothingRadius;
	// config.ParticleMass        = config.ReferenceDensity * volume * 0.5325f;
	config.ParticleMass        = 0.0025f;
	logDebug("Kernel Volume : %f", volume);
	logDebug("Particle Mass : %f", config.ParticleMass);

	float h_r_factor = 0.125f;
	config.Region              = float3(6.f, 6.f, 6.f);
	config.RegionHalf          = float3(config.Region.x / 2.f, config.Region.y / 2.f, config.Region.z / 2.f);
	config.InitRegion          = float3(h_r_factor * config.ThreadGridDim.x * config.ThreadBlockDim.x,
									    h_r_factor * config.ThreadGridDim.y * config.ThreadBlockDim.y,	
									    h_r_factor * config.ThreadGridDim.z * config.ThreadBlockDim.z);
	config.InitPosition        = float3(-config.RegionHalf.x, config.RegionHalf.y - config.InitRegion.y, -config.InitRegion.z / 2.f);

	config.CellSize            = float3(config.SmoothingRadius);
	config.CellGridDim         = uint3(static_cast<uint32_t>(ceilf(config.Region.x / config.CellSize.x)),
									   static_cast<uint32_t>(ceilf(config.Region.y / config.CellSize.y)),
									   static_cast<uint32_t>(ceilf(config.Region.z / config.CellSize.z)));
	config.NumCells            = static_cast<size_t>(config.CellGridDim.x) * config.CellGridDim.y * config.CellGridDim.z;

	config.RadixSortBlockSize  = 256;

	config.PhysicsDeltaTime    = 0.001f;
	config.RenderFPS           = 30.f;
	config.RenderDeltaTime     = 1.f / config.RenderFPS;
}

inline void InitializeScene(Scene& scene, SimulationConfig const& simulation_config)
{
	scene.cameraPosition    = glm::vec3{ 0, 0, simulation_config.RegionHalf.x * (1 + sqrtf(2)) + simulation_config.RegionHalf.z + 1.f };
	scene.particleSize      = simulation_config.ParticleRadius;
	scene.region            = *(glm::vec3*)&simulation_config.Region;
	scene.displayMode       = DM_Particles;
	scene.densityRange      = 20.0;
	scene.velocityRange     = 10.0;
	scene.pressureRange     = 150.0;
	scene.accelerationRange = 90.0;
}

#ifndef FLSIM_MAIN_CPP
extern SimulationConfig gSimCfg;
#endif

// Switches
#define FLSIM_ENABLE_CUDA() 1
