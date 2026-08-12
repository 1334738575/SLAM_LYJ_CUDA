#ifndef CUDA_LYJCOMMON_H
#define CUDA_LYJCOMMON_H

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <float.h>
#include <cmath>


// export
#ifdef WIN32
#ifdef _MSC_VER
#define CUDA_LYJ_API __declspec(dllexport)
#else
#define CUDA_LYJ_API
#endif
#else
#define CUDA_LYJ_API
#endif

#define CUDAORBMAXW 2000
#define CUDAORBMAXH 2000
#define CUDAORBKPSIZE 8192
#define CUDAORBGRIDSOLU 20
#define CUDAORBEVECELLSIZE 128
#define CUDASIFTKPSIZE CUDAORBKPSIZE
#define CUDASIFTDESCSIZE 128

namespace CUDA_LYJ
{
	__device__ inline void projectCameraPoint(const float* camera, unsigned int cameraModel,
		const float3& point, float& u, float& v)
	{
		const float x = point.x / point.z;
		const float y = point.y / point.z;
		if (cameraModel == 1u)
		{
			const float r = sqrtf(x * x + y * y);
			if (r < 1e-12f)
			{
				u = camera[2];
				v = camera[3];
				return;
			}
			const float theta = atanf(r);
			const float theta2 = theta * theta;
			const float theta4 = theta2 * theta2;
			const float theta6 = theta4 * theta2;
			const float theta8 = theta4 * theta4;
			const float thetaD = theta * (1.0f + camera[4] * theta2 + camera[5] * theta4 +
				camera[6] * theta6 + camera[7] * theta8);
			const float scale = thetaD / r;
			u = camera[0] * x * scale + camera[2];
			v = camera[1] * y * scale + camera[3];
			return;
		}
		u = camera[0] * x + camera[2];
		v = camera[1] * y + camera[3];
	}

	__device__ inline void unprojectCameraPixel(const float* camera, unsigned int cameraModel,
		float u, float v, float3& ray)
	{
		const float xDistorted = (u - camera[2]) / camera[0];
		const float yDistorted = (v - camera[3]) / camera[1];
		if (cameraModel == 1u)
		{
			const float thetaD = sqrtf(xDistorted * xDistorted + yDistorted * yDistorted);
			if (thetaD < 1e-12f)
			{
				ray.x = 0.0f;
				ray.y = 0.0f;
				ray.z = 1.0f;
				return;
			}

			float theta = fminf(thetaD, 1.57079632679f);
			for (int i = 0; i < 8; ++i)
			{
				const float theta2 = theta * theta;
				const float theta4 = theta2 * theta2;
				const float theta6 = theta4 * theta2;
				const float theta8 = theta4 * theta4;
				const float value = theta * (1.0f + camera[4] * theta2 + camera[5] * theta4 +
					camera[6] * theta6 + camera[7] * theta8) - thetaD;
				const float derivative = 1.0f + 3.0f * camera[4] * theta2 + 5.0f * camera[5] * theta4 +
					7.0f * camera[6] * theta6 + 9.0f * camera[7] * theta8;
				if (fabsf(derivative) < 1e-8f)
					break;
				theta = fminf(fmaxf(theta - value / derivative, 0.0f), 1.57079632679f);
			}
			const float scale = tanf(theta) / thetaD;
			ray.x = xDistorted * scale;
			ray.y = yDistorted * scale;
			ray.z = 1.0f;
			return;
		}
		ray.x = xDistorted;
		ray.y = yDistorted;
		ray.z = 1.0f;
	}

	extern __device__ float dot3(const float3& p1, const float3& p2);
		/// <summary>
	/// colmajor
	/// </summary>
	class CUDA_LYJ_API Mat34CU
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
		//__device__ float3 operator*(const float3 &_p) const
		//{
		//	float3 ret;
		//	ret.x = dataDev_[0] * _p.x + dataDev_[3] * _p.y + dataDev_[6] * _p.z + dataDev_[9];
		//	ret.y = dataDev_[1] * _p.x + dataDev_[4] * _p.y + dataDev_[7] * _p.z + dataDev_[10];
		//	ret.z = dataDev_[2] * _p.x + dataDev_[5] * _p.y + dataDev_[8] * _p.z + dataDev_[11];
		//	return ret;
		//}
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
	/// Pinhole: fx, fy, cx, cy. Fisheye additionally stores k1, k2, k3, k4.
	/// </summary>
	class CUDA_LYJ_API CameraCU
	{
	public:
		CameraCU()
		{
			cudaMalloc((void **)&paramsDev_, 8 * sizeof(float));
			cudaMalloc((void **)&paramsInvDev_, 4 * sizeof(float));
		};
		~CameraCU()
		{
			cudaFree(paramsDev_);
			cudaFree(paramsInvDev_);
		};

		__host__ void upload(int _w, int _h, float* _params, cudaStream_t _stream = nullptr)
		{
			upload(_w, _h, _params, 0u, _stream);
		}

		__host__ void upload(int _w, int _h, float* _params, unsigned int _cameraModel,
			cudaStream_t _stream = nullptr)
		{
			w_ = _w;
			h_ = _h;
			cameraModel_ = _cameraModel;
			cudaMemsetAsync(paramsDev_, 0, 8 * sizeof(float), _stream);
			const size_t parameterCount = _cameraModel == 1u ? 8 : 4;
			cudaMemcpyAsync(paramsDev_, _params, parameterCount * sizeof(float), cudaMemcpyHostToDevice, _stream);
			float camInv[4];
			camInv[0] = 1.0f / _params[0];
			camInv[1] = 1.0f / _params[1];
			camInv[2] = -1.0f * _params[2] / _params[0];
			camInv[3] = -1.0f * _params[3] / _params[1];
			cudaMemcpyAsync(paramsInvDev_, camInv, 4 * sizeof(float), cudaMemcpyHostToDevice, _stream);
		}

		__host__ void upload(int _w, int _h, float *_params, float *_paramsInv,
			cudaStream_t _stream = nullptr)
		{
			upload(_w, _h, _params, _paramsInv, 0u, _stream);
		}

		__host__ void upload(int _w, int _h, float *_params, float *_paramsInv,
			unsigned int _cameraModel, cudaStream_t _stream = nullptr)
		{
			w_ = _w;
			h_ = _h;
			cameraModel_ = _cameraModel;
			cudaMemsetAsync(paramsDev_, 0, 8 * sizeof(float), _stream);
			const size_t parameterCount = _cameraModel == 1u ? 8 : 4;
			cudaMemcpyAsync(paramsDev_, _params, parameterCount * sizeof(float), cudaMemcpyHostToDevice, _stream);
			cudaMemcpyAsync(paramsInvDev_, _paramsInv, 4 * sizeof(float), cudaMemcpyHostToDevice, _stream);
		}
		__host__ void download(float *_params, float *_paramsInv, cudaStream_t _stream = nullptr)
		{
			const size_t parameterCount = cameraModel_ == 1u ? 8 : 4;
			cudaMemcpyAsync(_params, paramsDev_, parameterCount * sizeof(float), cudaMemcpyDeviceToHost, _stream);
			cudaMemcpyAsync(_paramsInv, paramsInvDev_, 4 * sizeof(float), cudaMemcpyDeviceToHost, _stream);
		}
		//__device__ void pointToImage(const float3 &_p3d, float3 &_p2d)
		//{
		//	float invZ = 1.0f / _p3d.z;
		//	_p2d.x = _p3d.x * paramsDev_[0] * invZ + paramsDev_[2];
		//	_p2d.y = _p3d.y * paramsDev_[1] * invZ + paramsDev_[3];
		//	_p2d.z = _p3d.z;
		//}
		//__device__ void pointToImage(const float3 &_p3d, float2 &_p2d)
		//{
		//	float invZ = 1.0f / _p3d.z;
		//	_p2d.x = _p3d.x * paramsDev_[0] * invZ + paramsDev_[2];
		//	_p2d.y = _p3d.y * paramsDev_[1] * invZ + paramsDev_[3];
		//}
		//__device__ void imageToPoint(const float3 &_p2d, float3 &_p3d)
		//{
		//	// 1/fx, 0, -cx/fx,
		//	// 0, 1/fy, -cy/fy,
		//	// 0, 0, 1
		//	_p3d.x = (_p2d.x * paramsInvDev_[0] + paramsInvDev_[2]) * _p2d.z;
		//	_p3d.y = (_p2d.y * paramsInvDev_[1] + paramsInvDev_[3]) * _p2d.z;
		//	_p3d.z = _p3d.z;
		//}
		//__device__ void imageToPoint(const float2 &_p2d, float3 &_p3d)
		//{
		//	_p3d.x = _p2d.x * paramsInvDev_[0] + paramsInvDev_[2];
		//	_p3d.y = _p2d.y * paramsInvDev_[1] + paramsInvDev_[3];
		//	_p3d.z = 1.0f;
		//}
		//__device__ void imageToPoint(const float &_u, const float &_v, float3 &_p3d)
		//{
		//	_p3d.x = _u * paramsInvDev_[0] + paramsInvDev_[2];
		//	_p3d.y = _v * paramsInvDev_[1] + paramsInvDev_[3];
		//	_p3d.z = 1.0f;
		//}

		// private:
		float *paramsDev_ = nullptr;
		float *paramsInvDev_ = nullptr;
		unsigned int cameraModel_ = 0;
		int w_ = 0;
		int h_ = 0;
	};


	class CUDA_LYJ_API GridCU
	{
	public:
		GridCU() {
			gridW_ = (CUDAORBMAXW + CUDAORBGRIDSOLU - 1) / CUDAORBGRIDSOLU;
			gridH_ = (CUDAORBMAXH + CUDAORBGRIDSOLU - 1) / CUDAORBGRIDSOLU;
			cudaMalloc((void**)&cellDatasDev_, CUDAORBKPSIZE * sizeof(short));
			cudaMalloc((void**)&cellSzsDev_, (gridW_ * gridH_ + 1) * sizeof(short));
		};
		~GridCU() {
			cudaFree(cellDatasDev_);
			cudaFree(cellSzsDev_);
		};

		__host__ void upload(short* _cellDatas, short* _cellSzs, cudaStream_t _stream=nullptr)
		{
			cudaMemcpyAsync(cellDatasDev_, _cellDatas, CUDAORBKPSIZE * sizeof(short), cudaMemcpyHostToDevice, _stream);
			cudaMemcpyAsync(cellSzsDev_, _cellSzs, (gridW_ * gridH_ + 1) * sizeof(short), cudaMemcpyHostToDevice, _stream);
		}

		__device__ int getIndGrid(const float& _u, const float& _v, int& _cGrid, int& _rGrid)
		{
			_cGrid = int(_u) / CUDAORBGRIDSOLU;
			_rGrid = int(_v) / CUDAORBGRIDSOLU;
			if (_cGrid < 0 || _rGrid < 0 || _cGrid >= gridW_ || _rGrid >= gridH_)
				return 0;
			return 1;
		}
		__device__ void getKpIndInCell(const int& _cGrid, const int& _rGrid, short* _kpIndSt, int& _kpIndSz)
		{
			_kpIndSt = nullptr;
			_kpIndSz = 0;
			if (_cGrid < 0 || _rGrid < 0 || _cGrid >= gridW_ || _rGrid >= gridH_)
				return;
			int ind = _rGrid * gridW_ + _cGrid;
			_kpIndSz = int(cellSzsDev_[ind + 1] - cellSzsDev_[ind]);
			int st = int(cellSzsDev_[ind]);
			_kpIndSt = cellDatasDev_ + st;
		}
		__device__ int isIndInGrid(const int& _cGrid, const int& _rGrid)
		{
			if (_cGrid < 0 || _rGrid < 0 || _cGrid >= gridW_ || _rGrid >= gridH_)
				return 0;
			return 1;
		}


	//private:
		short* cellDatasDev_;
		short* cellSzsDev_;
		int gridW_;
		int gridH_;

	};

}

#endif // !CUDA_LYJCOMMON_H
