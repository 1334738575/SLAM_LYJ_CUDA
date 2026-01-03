#ifndef CUDA_LYJPROJECTOR_H
#define CUDA_LYJPROJECTOR_H

#include "CUDACommon.h"
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "CUDADefines.h"

namespace CUDA_LYJ
{

	class ProjectorCU
	{
	public:
		ProjectorCU() {};
		~ProjectorCU() {};

		void create(const float *Pws, const unsigned int PSize,
					const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
					float *camParams, const int w, const int h)
		{
			PSize_ = PSize;
			fSize_ = fSize;
			w_ = w;
			h_ = h;
			dIdsReset_.assign(w * h, UINT64_MAX);
			std::vector<DepthID2> dIds(w * h);
			for (int i = 0; i < w * h; ++i) {
				dIds[i].depth = FLT_MAX;
				dIds[i].fid = UINT_MAX;
			}
			memcpy(dIdsReset_.data(), dIds.data(), w_ * h_ * sizeof(unsigned long long));

			std::vector<float> camInv(4);
			camInv[0] = 1.0f / camParams[0];
			camInv[1] = 1.0f / camParams[1];
			camInv[2] = -1.0f * camParams[2] / camParams[0];
			camInv[3] = -1.0f * camParams[3] / camParams[1];

			cudaMalloc((void **)&PwsDev_, PSize * 3 * sizeof(float));
			cudaMemcpy(PwsDev_, Pws, PSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&ctrwsDev_, fSize * 3 * sizeof(float));
			cudaMemcpy(ctrwsDev_, centers, fSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&facesDev_, fSize * 3 * sizeof(unsigned int));
			cudaMemcpy(facesDev_, faces, fSize * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&fNormalwsDev_, fSize * 3 * sizeof(float));
			cudaMemcpy(fNormalwsDev_, fNormals, fSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			camDev_.upload(w, h, camParams, camInv.data());
		}

		//void project(float *Tcw,
		//			 float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
		//			 float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1)
		//{
		//	TDev_.upload(Tcw);
		//	cudaMemcpy(dIdsDev_, dIdsReset_.data(), w_ * h_ * sizeof(unsigned long long), cudaMemcpyHostToDevice);

		//	testTransformCUDA(TDev_, PwsDev_, PcsDev_, PSize_);
		//	testTransformCUDA(TDev_, ctrwsDev_, ctrcsDev_, fSize_);
		//	testTransformNormalCUDA(TDev_, fNormalwsDev_, fNormalcsDev_, fSize_);
		//	testCameraCUDA(PcsDev_, pixelsDev_, PSize_, w_, h_, camDev_);
		//	testCameraCUDA(ctrcsDev_, ctrPixelsDev_, fSize_, w_, h_, camDev_);
		//	testDepthAndFidAndCheckCUDA(PcsDev_, pixelsDev_, facesDev_, fNormalcsDev_, PSize_, fSize_, w_, h_, ctrPixelsDev_, minD, maxD, csTh, detDTh, depthDev_, dIdsDev_, isPVisibleDev_, isFVisibleDev_, camDev_);
		//	cudaDeviceSynchronize();

		//	cudaMemcpy(depths, depthDev_, w_ * h_ * sizeof(float), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(dIds_.data(), dIdsDev_, w_ * h_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(allVisiblePIds, isPVisibleDev_, PSize_ * sizeof(char), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(allVisibleFIds, isFVisibleDev_, fSize_ * sizeof(char), cudaMemcpyDeviceToHost);
		//	for (int i = 0; i < w_ * h_; ++i)
		//	{
		//		fIds[i] = dIds_[i].fid;
		//	}
		//}

		void project(ProjectorCache& cache,
			float* Tcw,
			float* depths, unsigned int* fIds, char* allVisiblePIds, char* allVisibleFIds,
			float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1)
		{
			cache.TDev_.upload(Tcw);
			cudaMemcpy(cache.dIdsDev_, dIdsReset_.data(), w_ * h_ * sizeof(unsigned long long), cudaMemcpyHostToDevice);

			testTransformCUDA(cache.TDev_, PwsDev_, cache.PcsDev_, PSize_);
			testTransformCUDA(cache.TDev_, ctrwsDev_, cache.ctrcsDev_, fSize_);
			testTransformNormalCUDA(cache.TDev_, fNormalwsDev_, cache.fNormalcsDev_, fSize_);
			testCameraCUDA(cache.PcsDev_, cache.pixelsDev_, PSize_, w_, h_, camDev_);
			testCameraCUDA(cache.ctrcsDev_, cache.ctrPixelsDev_, fSize_, w_, h_, camDev_);
			testDepthAndFidAndCheckCUDA(cache.PcsDev_, cache.pixelsDev_, facesDev_, cache.fNormalcsDev_, PSize_, fSize_, w_, h_, cache.ctrPixelsDev_, minD, maxD, csTh, detDTh, cache.depthDev_, cache.dIdsDev_, cache.isPVisibleDev_, cache.isFVisibleDev_, camDev_);
			cudaDeviceSynchronize();

			cudaMemcpy(depths, cache.depthDev_, w_ * h_ * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(cache.dIds_.data(), cache.dIdsDev_, w_ * h_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(allVisiblePIds, cache.isPVisibleDev_, PSize_ * sizeof(char), cudaMemcpyDeviceToHost);
			cudaMemcpy(allVisibleFIds, cache.isFVisibleDev_, fSize_ * sizeof(char), cudaMemcpyDeviceToHost);
			for (int i = 0; i < w_ * h_; ++i)
			{
				fIds[i] = cache.dIds_[i].fid;
			}
		}

		void release()
		{
			cudaFree(PwsDev_);
			cudaFree(ctrwsDev_);
			cudaFree(facesDev_);
			cudaFree(fNormalwsDev_);
		}

		void testTransformCUDA(const Mat34CU &_T, float3 *_ps, float3 *_rets, unsigned int _vn);

		void testTransformNormalCUDA(const Mat34CU &_T, float3 *_normals, float3 *_rets, unsigned int _n);

		void testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, const CameraCU &_cam);

		void testDepthAndFidAndCheckCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals,
										 unsigned int _vn, unsigned int _fn, int _w, int _h, float3 *_ctr2ds, float _minD, float _maxD, float _csTh, float _detDTh,
										 float *_depths, unsigned long long *_dIds, char *_isPVisible, char *_isFVisible,
										 const CameraCU &_cam);

		unsigned int PSize_ = 0;
		unsigned int fSize_ = 0;
		int w_ = 0;
		int h_ = 0;
		std::vector<unsigned long long> dIdsReset_;
		CameraCU camDev_;
		float3 *PwsDev_;
		float3 *ctrwsDev_;
		uint3 *facesDev_;
		float3 *fNormalwsDev_;

	};

}

#endif // !CUDA_LYJPROJECTOR_H
