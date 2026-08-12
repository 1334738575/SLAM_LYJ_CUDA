#ifndef CUDA_LYJPROJECTOR_H
#define CUDA_LYJPROJECTOR_H

#include "CUDACommon.cuh"
#include "CUDADefines.h"
#include <stdint.h>

namespace CUDA_LYJ
{

	class ProjectorCU
	{
	public:
		ProjectorCU() {};
		~ProjectorCU() {};

		void create(const float *Pws, const unsigned int PSize,
					const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
					float *camParams, const int w, const int h, CameraModel cameraModel = CameraModel::Pinhole)
		{
			PSize_ = PSize;
			fSize_ = fSize;
			w_ = w;
			h_ = h;
			dIdsReset_.assign(w * h, UINT64_MAX);
			std::vector<DepthID2> dIds(w * h);
			for (int i = 0; i < w * h; ++i)
			{
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
			camDev_.upload(w, h, camParams, camInv.data(), static_cast<unsigned int>(cameraModel));
		}

		void project(ProjectorCache &cache,
					 float *Tcw,
					 float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
					 float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1,
					 std::vector<uint32_t> *faceIds = nullptr, std::vector<uint32_t> *pointIds = nullptr)
		{
			cache.TDev_.upload(Tcw);
			cudaMemcpy(cache.dIdsDev_, dIdsReset_.data(), w_ * h_ * sizeof(unsigned long long), cudaMemcpyHostToDevice);

			unsigned int *faceIdsDev = nullptr;
			unsigned int projectFSize = fSize_;
			unsigned int *pointIdsDev = nullptr;
			char *pointMaskDev = nullptr;
			unsigned int projectPSize = PSize_;
			if (faceIds != nullptr)
			{
				projectFSize = static_cast<unsigned int>(faceIds->size());
				if (projectFSize > 0)
				{
					cudaMalloc((void **)&faceIdsDev, projectFSize * sizeof(unsigned int));
					cudaMemcpy(faceIdsDev, faceIds->data(), projectFSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
				}
			}
			if (pointIds != nullptr)
			{
				projectPSize = static_cast<unsigned int>(pointIds->size());
				cudaMalloc((void **)&pointMaskDev, PSize_ * sizeof(char));
				cudaMemset(pointMaskDev, 0, PSize_ * sizeof(char));
				if (projectPSize > 0)
				{
					std::vector<char> pointMask(PSize_, 0);
					for (uint32_t pointId : *pointIds)
					{
						if (pointId < PSize_)
							pointMask[pointId] = 1;
					}
					cudaMalloc((void **)&pointIdsDev, projectPSize * sizeof(unsigned int));
					cudaMemcpy(pointIdsDev, pointIds->data(), projectPSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
					cudaMemcpy(pointMaskDev, pointMask.data(), PSize_ * sizeof(char), cudaMemcpyHostToDevice);
				}
			}

			testTransformAndCameraCUDA(cache.TDev_, PwsDev_, cache.pixelsDev_, PSize_, camDev_, pointIdsDev, projectPSize, pointIds != nullptr);
			testTransformAndCameraCUDA(cache.TDev_, ctrwsDev_, cache.ctrPixelsDev_, fSize_, camDev_, faceIdsDev, projectFSize, faceIds != nullptr);
			testDepthAndFidAndCheckCUDA(cache.TDev_, cache.pixelsDev_, facesDev_, fNormalwsDev_, PSize_, fSize_, w_, h_, cache.ctrPixelsDev_, minD, maxD, csTh, detDTh, cache.depthDev_, cache.dIdsDev_, cache.isPVisibleDev_, cache.isFVisibleDev_, camDev_, faceIdsDev, projectFSize, faceIds != nullptr, pointIdsDev, projectPSize, pointIds != nullptr, pointMaskDev);
			cudaDeviceSynchronize();
			if (faceIdsDev != nullptr)
				cudaFree(faceIdsDev);
			if (pointIdsDev != nullptr)
				cudaFree(pointIdsDev);
			if (pointMaskDev != nullptr)
				cudaFree(pointMaskDev);

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

		void testTransformAndCameraCUDA(const Mat34CU &_T, float3 *_pws, float3 *_p2ds, unsigned int _vn, const CameraCU &_cam, unsigned int *_ids = nullptr, unsigned int _projectVn = 0, bool _useIds = false);

		void testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, const CameraCU &_cam);

		void testDepthAndFidAndCheckCUDA(const Mat34CU &_T, float3 *_p2ds, uint3 *_faces, float3 *_fNormals,
										 unsigned int _vn, unsigned int _fn, int _w, int _h, float3 *_ctr2ds, float _minD, float _maxD, float _csTh, float _detDTh,
										 float *_depths, unsigned long long *_dIds, char *_isPVisible, char *_isFVisible,
										 const CameraCU &_cam, unsigned int *_faceIds = nullptr, unsigned int _projectFn = 0, bool _useFaceIds = false,
										 unsigned int *_pointIds = nullptr, unsigned int _projectVn = 0, bool _usePointIds = false, char *_pointMask = nullptr);

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
