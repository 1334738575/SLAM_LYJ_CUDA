#include "CUDACommon.cuh"

namespace CUDA_LYJ
{
	__device__ float dot3(const float3& p1, const float3& p2)
	{
		return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
	}
}