#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <stdexcept>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while (0)

struct alignas(16) Light { float3 pos; float3 color; float radius; };
struct alignas(16) GBufferPixel { float3 pos; float3 normal; float3 albedo; float roughness; };

__host__ __device__ inline float saturatef(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__global__ void tiledDeferredLighting(
    const GBufferPixel* __restrict__ gbuf,
    const Light* __restrict__ lights,
    const int* __restrict__ tileLightOffsets,
    const int* __restrict__ tileLightCounts,
    const int* __restrict__ tileLightList,
    float3* __restrict__ outColor,
    int width, int height, int tileSize)
{
    const int tx = blockIdx.x;
    const int ty = blockIdx.y;
    const int lx = threadIdx.x;
    const int ly = threadIdx.y;

    const int px = tx * tileSize + lx;
    const int py = ty * tileSize + ly;
    if (px >= width || py >= height) return;

    const int pixelIdx = py * width + px;
    const int tileIdx = ty * gridDim.x + tx;

    extern __shared__ int s_lightIndices[];
    const int tileLightCount = tileLightCounts[tileIdx];
    const int offset = tileLightOffsets[tileIdx];

    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < tileLightCount;
         i += blockDim.x * blockDim.y)
    {
        s_lightIndices[i] = tileLightList[offset + i];
    }
    __syncthreads();

    GBufferPixel gp = gbuf[pixelIdx];
    float3 V = normalize(make_float3(0.0f, 0.0f, 0.0f) - gp.pos);
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < tileLightCount; ++i) {
        const Light L = lights[s_lightIndices[i]];
        float3 Ldir = L.pos - gp.pos;
        float dist2 = dot(Ldir, Ldir);
        float invDist = rsqrtf(dist2 + 1e-6f);
        Ldir *= invDist;
        float NdotL = fmaxf(dot(gp.normal, Ldir), 0.0f);
        if (NdotL <= 0.0f) continue;

        float attenuation = saturatef(1.0f - sqrtf(dist2) / L.radius);
        float3 H = normalize(Ldir + V);
        float spec = powf(fmaxf(dot(gp.normal, H), 0.0f), 16.0f * (1.0f - gp.roughness));
        color += (gp.albedo * NdotL + spec) * L.color * attenuation;
    }
    outColor[pixelIdx] = color;
}

void launchTiledLighting(
    const GBufferPixel* d_gbuf, const Light* d_lights,
    const int* d_tileOffsets, const int* d_tileCounts, const int* d_tileList,
    float3* d_out, int width, int height, int tileSize)
{
    dim3 block(tileSize, tileSize);
    dim3 grid((width + tileSize - 1) / tileSize, (height + tileSize - 1) / tileSize);
    size_t sharedBytes = 1024 * sizeof(int);
    tiledDeferredLighting<<<grid, block, sharedBytes>>>(
        d_gbuf, d_lights, d_tileOffsets, d_tileCounts, d_tileList,
        d_out, width, height, tileSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}