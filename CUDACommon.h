#ifndef CUDA_LYJCOMMON_H
#define CUDA_LYJCOMMON_H

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace CUDA_LYJ
{
	/// <summary>
	/// colmajor
	/// </summary>
	class Mat34CU
	{
	public:
		Mat34CU()
		{
			cudaMalloc((void **)&dataDev_, 12 * sizeof(float));
		};
		~Mat34CU()
		{
			cudaFree(dataDev_);
		};

		__host__ void upload(float *_data, cudaStream_t _stream = nullptr)
		{
			// cudaMemcpyAsync(dataDev_, _data, 12 * sizeof(float), cudaMemcpyHostToDevice, _stream);
			cudaMemcpy(dataDev_, _data, 12 * sizeof(float), cudaMemcpyHostToDevice);
		}
		__host__ void uploadRotation(float *_data, cudaStream_t _stream = nullptr)
		{
			// cudaMemcpyAsync(dataDev_, _data, 12 * sizeof(float), cudaMemcpyHostToDevice, _stream);
			cudaMemcpy(dataDev_, _data, 9 * sizeof(float), cudaMemcpyHostToDevice);
		}
		__host__ void uploadTrans(float *_data, cudaStream_t _stream = nullptr)
		{
			// cudaMemcpyAsync(dataDev_, _data, 12 * sizeof(float), cudaMemcpyHostToDevice, _stream);
			cudaMemcpy(dataDev_ + 9, _data, 3 * sizeof(float), cudaMemcpyHostToDevice);
		}
		__host__ void download(float *_data, cudaStream_t _stream = nullptr)
		{
			// cudaMemcpyAsync(_data, dataDev_, 12 * sizeof(float), cudaMemcpyDeviceToHost, _stream);
			cudaMemcpy(_data, dataDev_, 12 * sizeof(float), cudaMemcpyDeviceToHost);
		}
		__device__ float3 operator*(const float3 &_p) const
		{
			float3 ret;
			ret.x = dataDev_[0] * _p.x + dataDev_[3] * _p.y + dataDev_[6] * _p.z + dataDev_[9];
			ret.y = dataDev_[1] * _p.x + dataDev_[4] * _p.y + dataDev_[7] * _p.z + dataDev_[10];
			ret.z = dataDev_[2] * _p.x + dataDev_[5] * _p.y + dataDev_[8] * _p.z + dataDev_[11];
			return ret;
		}
		__device__ float3 transform(const float3 &_p) const
		{
			float3 ret;
			ret.x = dataDev_[0] * _p.x + dataDev_[3] * _p.y + dataDev_[6] * _p.z + dataDev_[9];
			ret.y = dataDev_[1] * _p.x + dataDev_[4] * _p.y + dataDev_[7] * _p.z + dataDev_[10];
			ret.z = dataDev_[2] * _p.x + dataDev_[5] * _p.y + dataDev_[8] * _p.z + dataDev_[11];
			return ret;
		}
		__device__ float3 transformNormal(const float3 &_n) const
		{
			float3 ret;
			ret.x = dataDev_[0] * _n.x + dataDev_[3] * _n.y + dataDev_[6] * _n.z;
			ret.y = dataDev_[1] * _n.x + dataDev_[4] * _n.y + dataDev_[7] * _n.z;
			ret.z = dataDev_[2] * _n.x + dataDev_[5] * _n.y + dataDev_[8] * _n.z;
			return ret;
		}

		// private:
		float *dataDev_ = nullptr;
	};

	/// <summary>
	/// fx, fy, cx, cy
	/// </summary>
	class CameraCU
	{
	public:
		CameraCU()
		{
			cudaMalloc((void **)&paramsDev_, 4 * sizeof(float));
			cudaMalloc((void **)&paramsInvDev_, 4 * sizeof(float));
		};
		~CameraCU()
		{
			cudaFree(paramsDev_);
			cudaFree(paramsInvDev_);
		};

		__host__ void upload(int _w, int _h, float *_params, float *_paramsInv, cudaStream_t _stream = nullptr)
		{
			w_ = _w;
			h_ = _h;
			cudaMemcpyAsync(paramsDev_, _params, 4 * sizeof(float), cudaMemcpyHostToDevice, _stream);
			cudaMemcpyAsync(paramsInvDev_, _paramsInv, 4 * sizeof(float), cudaMemcpyHostToDevice, _stream);
		}
		__host__ void download(float *_params, float *_paramsInv, cudaStream_t _stream = nullptr)
		{
			cudaMemcpyAsync(_params, paramsDev_, 4 * sizeof(float), cudaMemcpyDeviceToHost, _stream);
			cudaMemcpyAsync(_paramsInv, paramsInvDev_, 4 * sizeof(float), cudaMemcpyDeviceToHost, _stream);
		}
		__device__ void pointToImage(const float3 &_p3d, float3 &_p2d)
		{
			float invZ = 1.0f / _p3d.z;
			_p2d.x = _p3d.x * paramsDev_[0] * invZ + paramsDev_[2];
			_p2d.y = _p3d.y * paramsDev_[1] * invZ + paramsDev_[3];
			_p2d.z = _p3d.z;
		}
		__device__ void pointToImage(const float3 &_p3d, float2 &_p2d)
		{
			float invZ = 1.0f / _p3d.z;
			_p2d.x = _p3d.x * paramsDev_[0] * invZ + paramsDev_[2];
			_p2d.y = _p3d.y * paramsDev_[1] * invZ + paramsDev_[3];
		}
		__device__ void imageToPoint(const float3 &_p2d, float3 &_p3d)
		{
			// 1/fx, 0, -cx/fx,
			// 0, 1/fy, -cy/fy,
			// 0, 0, 1
			_p3d.x = (_p2d.x * paramsInvDev_[0] + paramsInvDev_[2]) * _p2d.z;
			_p3d.y = (_p2d.y * paramsInvDev_[1] + paramsInvDev_[3]) * _p2d.z;
			_p3d.z = _p3d.z;
		}
		__device__ void imageToPoint(const float2 &_p2d, float3 &_p3d)
		{
			_p3d.x = _p2d.x * paramsInvDev_[0] + paramsInvDev_[2];
			_p3d.y = _p2d.y * paramsInvDev_[1] + paramsInvDev_[3];
			_p3d.z = 1.0f;
		}
		__device__ void imageToPoint(const float &_u, const float &_v, float3 &_p3d)
		{
			_p3d.x = _u * paramsInvDev_[0] + paramsInvDev_[2];
			_p3d.y = _v * paramsInvDev_[1] + paramsInvDev_[3];
			_p3d.z = 1.0f;
		}

		// private:
		float *paramsDev_ = nullptr;
		float *paramsInvDev_ = nullptr;
		int w_ = 0;
		int h_ = 0;
	};

	class BaseCU
	{
	public:
		BaseCU() {};
		~BaseCU() {};

		__device__ void add2(const float2 &p1, const float2 &p2, float2 &p3)
		{
			p3.x = p1.x + p2.x;
			p3.y = p1.y + p2.y;
		}
		__device__ void sub2(const float2 &p1, const float2 &p2, float2 &p3)
		{
			p3.x = p1.x - p2.x;
			p3.y = p1.y - p2.y;
		}
		__device__ float dot2(const float2 &p1, const float2 &p2)
		{
			return p1.x * p2.x + p1.y * p2.y;
		}
		__device__ float crossProduct2(const float2 &p1, const float2 &p2)
		{
			return p1.x * p2.y - p1.y * p2.x;
		}
		__device__ char isP2dInTriangleCU(const float2 &AB, const float2 &BC, const float2 &CA,
										  const float2 &AP, const float2 &BP, const float2 &CP)
		{
			float v1 = crossProduct2(AB, AP);
			float v2 = crossProduct2(BC, BP);
			float v3 = crossProduct2(CA, CP);
			if (v1 >= 0 && v2 >= 0 && v3 >= 0)
				return (char)1;
			if (v1 <= 0 && v2 <= 0 && v3 <= 0)
				return (char)1;
			return (char)0;
		}
		__device__ char isP2dInTriangleCU(const float2 &A, const float2 &B, const float2 &C, const float2 &P)
		{
			float2 AB, BC, CA, AP, BP, CP;
			sub2(B, A, AB);
			sub2(C, B, BC);
			sub2(A, C, CA);
			sub2(P, A, AP);
			sub2(P, B, BP);
			sub2(P, C, CP);
			return isP2dInTriangleCU(AB, BC, CA, AP, BP, CP);
		}

		__device__ void add3(const float3 &p1, const float3 &p2, float3 &p3)
		{
			p3.x = p1.x + p2.x;
			p3.y = p1.y + p2.y;
			p3.z = p1.z + p2.z;
		}
		__device__ void sub3(const float3 &p1, const float3 &p2, float3 &p3)
		{
			p3.x = p1.x - p2.x;
			p3.y = p1.y - p2.y;
			p3.z = p1.z - p2.z;
		}
		__device__ float dot3(const float3 &p1, const float3 &p2)
		{
			return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
		}
		__device__ void crossProduct3(const float3 &p1, const float3 &p2, float3 &p3)
		{
			p3.x = p1.y * p2.z - p1.z * p2.y;
			p3.y = p1.z * p2.x - p1.x * p2.z;
			p3.z = p1.x * p2.y - p1.y * p2.x;
		}
		__device__ void computePlane(const float3 &A, const float3 &B, const float3 &C, float4 &plane)
		{
			float3 AB, AC, n;
			sub3(B, A, AB);
			sub3(C, A, AC);
			crossProduct3(AB, AC, n);
			plane.x = n.x;
			plane.y = n.y;
			plane.z = n.z;
			plane.w = -1 * dot3(n, A);
		}
		__device__ void computePlane(const float3 &A, const float3 &B, const float3 &C, float3 &abc, float &d)
		{
			float3 AB, AC;
			sub3(B, A, AB);
			sub3(C, A, AC);
			crossProduct3(AB, AC, abc);
			d = -1 * dot3(abc, A);
		}

		__device__ void add4(const float4 &p1, const float4 &p2, float4 &p3)
		{
			p3.x = p1.x + p2.x;
			p3.y = p1.y + p2.y;
			p3.z = p1.z + p2.z;
			p3.w = p1.w + p2.w;
		}
		__device__ void sub4(const float4 &p1, const float4 &p2, float4 &p3)
		{
			p3.x = p1.x - p2.x;
			p3.y = p1.y - p2.y;
			p3.z = p1.z - p2.z;
			p3.w = p1.w - p2.w;
		}
		__device__ float dot4(const float4 &p1, const float4 &p2)
		{
			return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z + p1.w * p2.w;
		}
	};
}

#endif // !CUDA_LYJCOMMON_H
