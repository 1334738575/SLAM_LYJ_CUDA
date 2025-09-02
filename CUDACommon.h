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

}

#endif // !CUDA_LYJCOMMON_H
