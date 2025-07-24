#ifndef CUDA_LYJPROJECTOR_H
#define CUDA_LYJPROJECTOR_H

#include "CUDACommon.h"
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace CUDA_LYJ
{
	void testCUDA(int *_as, int *_bs, int *_cs, int _sz);

	void testTextureCUDA(float *_output, int _w, int _h, cudaTextureObject_t _texObj);

	union DepthID2
	{
		struct
		{
			float depth;
			unsigned int fid;
		};
		unsigned long long data;
	};

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
			dIdsReset_.resize(w * h);
			dIds_.resize(w * h);
			for (auto &did : dIds_)
			{
				did.depth = FLT_MAX;
				did.fid = UINT_MAX;
			}
			memcpy(dIdsReset_.data(), dIds_.data(), w * h * sizeof(unsigned long long));
			std::vector<float> camInv(4);
			camInv[0] = 1.0f / camParams[0];
			camInv[1] = 1.0f / camParams[1];
			camInv[2] = -1.0f * camParams[2] / camParams[0];
			camInv[3] = -1.0f * camParams[3] / camParams[1];

			cudaMalloc((void **)&PwsDev_, PSize * 3 * sizeof(float));
			cudaMemcpy(PwsDev_, Pws, PSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&PcsDev_, PSize * 3 * sizeof(float));
			cudaMalloc((void **)&ctrwsDev_, fSize * 3 * sizeof(float));
			cudaMemcpy(ctrwsDev_, centers, fSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&ctrcsDev_, fSize * 3 * sizeof(float));
			cudaMalloc((void **)&facesDev_, fSize * 3 * sizeof(unsigned int));
			cudaMemcpy(facesDev_, faces, fSize * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&fNormalwsDev_, fSize * 3 * sizeof(float));
			cudaMemcpy(fNormalwsDev_, fNormals, fSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&fNormalcsDev_, fSize * 3 * sizeof(float));
			cudaMalloc((void **)&pixelsDev_, PSize * 3 * sizeof(float));
			cudaMalloc((void **)&ctrPixelsDev_, fSize * 3 * sizeof(float));
			camDev_.upload(w, h, camParams, camInv.data());
			cudaMalloc((void **)&depthDev_, w * h * sizeof(float));
			cudaMalloc((void **)&dIdsDev_, w * h * sizeof(unsigned long long));
			cudaMemcpy(dIdsDev_, dIds_.data(), w * h * sizeof(unsigned long long), cudaMemcpyHostToDevice);
			cudaMalloc((void **)&isPVisibleDev_, PSize * sizeof(char));
			cudaMalloc((void **)&isFVisibleDev_, fSize * sizeof(char));
		}

		void project(float *Tcw,
					 float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds)
		{
			TDev_.upload(Tcw);
			cudaMemcpy(dIdsDev_, dIdsReset_.data(), w_ * h_ * sizeof(unsigned long long), cudaMemcpyHostToDevice);

			testTransformCUDA(TDev_, PwsDev_, PcsDev_, PSize_);
			testTransformCUDA(TDev_, ctrwsDev_, ctrcsDev_, fSize_);
			testTransformNormalCUDA(TDev_, fNormalwsDev_, fNormalcsDev_, fSize_);
			testCameraCUDA(PcsDev_, pixelsDev_, PSize_, w_, h_, camDev_);
			testCameraCUDA(ctrcsDev_, ctrPixelsDev_, fSize_, w_, h_, camDev_);
			testDepthAndFidAndCheckCUDA(PcsDev_, pixelsDev_, facesDev_, fNormalcsDev_, PSize_, fSize_, w_, h_, ctrPixelsDev_, depthDev_, dIdsDev_, isPVisibleDev_, isFVisibleDev_, camDev_);
			cudaDeviceSynchronize();

			cudaMemcpy(depths, depthDev_, w_ * h_ * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(dIds_.data(), dIdsDev_, w_ * h_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(allVisiblePIds, isPVisibleDev_, PSize_ * sizeof(char), cudaMemcpyDeviceToHost);
			cudaMemcpy(allVisibleFIds, isFVisibleDev_, fSize_ * sizeof(char), cudaMemcpyDeviceToHost);
			for (int i = 0; i < w_ * h_; ++i)
			{
				fIds[i] = dIds_[i].fid;
			}
		}

		void release()
		{
			cudaFree(PwsDev_);
			cudaFree(PcsDev_);
			cudaFree(ctrwsDev_);
			cudaFree(ctrcsDev_);
			cudaFree(facesDev_);
			cudaFree(fNormalwsDev_);
			cudaFree(fNormalcsDev_);
			cudaFree(pixelsDev_);
			cudaFree(ctrPixelsDev_);
			cudaFree(depthDev_);
			cudaFree(dIdsDev_);
			cudaFree(isPVisibleDev_);
			cudaFree(isFVisibleDev_);
		}

		void testTransformCUDA(const Mat34CU &_T, float3 *_ps, float3 *_rets, unsigned int _vn);

		void testTransformNormalCUDA(const Mat34CU &_T, float3 *_normals, float3 *_rets, unsigned int _n);

		void testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, const CameraCU &_cam);

		void testDepthAndFidCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, unsigned long long *_dIds, const CameraCU &_cam);

		void testDepthAndFidAndCheckCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _vn, unsigned int _fn, int _w, int _h, float3 *_ctr2ds, float *_depths, unsigned long long *_dIds, char *_isPVisible, char *_isFVisible, const CameraCU &_cam);

		unsigned int PSize_ = 0;
		unsigned int fSize_ = 0;
		int w_ = 0;
		int h_ = 0;
		std::vector<DepthID2> dIds_;
		std::vector<unsigned long long> dIdsReset_;
		std::vector<float> Pcs;
		std::vector<float> pixels;
		std::vector<float> ctrcs;
		std::vector<float> ctrPixels;
		std::vector<float> fnormalcs;
		std::vector<char> pVisible;
		std::vector<char> fVisible;

		CameraCU camDev_;
		float3 *PwsDev_;
		float3 *PcsDev_;
		float3 *ctrwsDev_;
		float3 *ctrcsDev_;
		uint3 *facesDev_;
		float3 *fNormalwsDev_;
		float3 *fNormalcsDev_;
		float3 *pixelsDev_;
		float3 *ctrPixelsDev_;

		Mat34CU TDev_;
		float *depthDev_ = nullptr;
		unsigned long long *dIdsDev_;
		char *isPVisibleDev_;
		char *isFVisibleDev_;
	};

}

#endif // !CUDA_LYJPROJECTOR_H
