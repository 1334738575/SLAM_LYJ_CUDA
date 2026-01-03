#include "CUDATexture.h"

namespace CUDA_LYJ
{

	__global__ void testTextureCU(float* _output, int _w, int _h, cudaTextureObject_t _texObj)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idx >= _w || idy >= _h)
			return;

		if (false)
		{
			float u = (idx + 0.5f) / _w;
			float v = (idy + 0.5f) / _h;
			float4 pixel = tex2D<float4>(_texObj, u, v);
			_output[idy * _w + idx] = pixel.x + pixel.y + pixel.z + pixel.w;
		}
		else
		{
			float u = idx; // 或 x + 0.5f（若需中心对齐）
			float v = idy;
			uchar4 pixel = tex2D<uchar4>(_texObj, u, v); // 数据类型与通道格式一致
			_output[idy * _w + idx] = pixel.x + pixel.y + pixel.z + pixel.w;
		}
	}
	void testTextureCUDA(float* _output, int _w, int _h, cudaTextureObject_t _texObj)
	{
		dim3 block(16, 16);
		dim3 grid((_w + block.x - 1) / block.x, (_h + block.y - 1) / block.y);
		testTextureCU << <grid, block >> > (_output, _w, _h, _texObj);
	}
}