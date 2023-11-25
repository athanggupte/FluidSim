#pragma once
#include <glm/ext/scalar_constants.hpp>

// Simulation constants

struct SimulationConfig
{
	size_t NumParticles{};
	uint3  GridDim{};
	uint3  BlockDim{};
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

	float PhysicsDeltaTime{};
	float RenderFPS{};
	float RenderDeltaTime{};
};

inline void InitializeSimulationConfig(SimulationConfig& config, cudaDeviceProp const& device_prop)
{
	config.NumParticles        = 27000;
	config.GridDim             = uint3(3, 3, 3); // gridDim
	config.BlockDim            = uint3(10, 10, 10); // blockDim
	config.NumBlocks           = static_cast<size_t>(config.GridDim.x) * config.GridDim.y;
	config.NumThreadsPerBlock  = static_cast<size_t>(config.BlockDim.x) * config.BlockDim.y * config.BlockDim.z;
	config.NumParticlesPerWarp = config.NumBlocks * config.NumThreadsPerBlock;
	config.NumWarps            = config.NumParticles / config.NumParticlesPerWarp + (config.NumParticles % config.NumParticlesPerWarp ? 1 : 0);

	config.Gravity             = float3(0, -9.81f, 0);
	config.SpeedOfSound        = 330.f;
	config.ReferenceDensity    = 10.f;
	config.TaitExponent        = 7;
	config.ReferencePressure   = 100;// config.ReferenceDensity * config.SpeedOfSound / config.TaitExponent;
	config.Viscosity           = 1e-2f;
	config.CollisionDamping    = 0.5f;

	config.SmoothingRadius     = 0.1f;
	config.ParticleRadius      = 0.025f;

	//float h = config.SmoothingRadius;
	//float volume = (1.f/12.f * (h-4) * ((h-5)*h + 10) * h*h*h + 4.f * h) * glm::pi<float>();
	float volume = 4.f / 3.f * glm::pi<float>() * config.SmoothingRadius * config.SmoothingRadius * config.SmoothingRadius;
	// config.ParticleMass        = config.ReferenceDensity * volume * 0.5325f;
	config.ParticleMass        = 5e-3f;
	logDebug("Kernel Volume : %f", volume);
	logDebug("Particle Mass : %f", config.ParticleMass);

	float h_r_factor = 0.075f;
	config.Region              = float3(10.f, 10.f, 10.f);
	config.RegionHalf          = float3(config.Region.x / 2.f, config.Region.y / 2.f, config.Region.z / 2.f);
	config.InitRegion          = float3(h_r_factor * config.GridDim.x * config.BlockDim.x, h_r_factor * config.GridDim.y * config.BlockDim.y, h_r_factor * config.GridDim.z * config.BlockDim.z);
	config.InitPosition        = float3(-config.InitRegion.x / 2.f, config.RegionHalf.y - config.InitRegion.y, -config.InitRegion.z / 2.f);

	config.PhysicsDeltaTime    = 0.001f;
	config.RenderFPS           = 30.f;
	config.RenderDeltaTime     = 1.f / config.RenderFPS;
}

#ifndef FLSIM_MAIN_CPP
extern SimulationConfig gSimCfg;
#endif

// Switches
#define FLSIM_ENABLE_CUDA() 1
