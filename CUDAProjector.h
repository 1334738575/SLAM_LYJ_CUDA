#ifndef CUDA_LYJPROJECTOR_H
#define CUDA_LYJPROJECTOR_H

#include "CUDACommon.h"

namespace CUDA_LYJ
{
	void testCUDA(int *_as, int *_bs, int *_cs, int _sz);

	void testTextureCUDA(float *_output, int _w, int _h, cudaTextureObject_t _texObj);

	class ProjectorCU
	{
	public:
		ProjectorCU() {};

		void testTransformCUDA(Mat34CU _T, float3 *_ps, float3 *_rets, unsigned int _vn);
		void testTransformCUDA2(float *_T, float3 *_ps, float3 *_rets, unsigned int _vn);

		void testTransformNormalCUDA(Mat34CU _T, float3 *_normals, float3 *_rets, unsigned int _n);

		void testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, CameraCU _cam);

		void testDepthAndFidCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, float *_depths, unsigned int *_fids, CameraCU _cam, BaseCU _base);

		void testDepthAndFidCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, unsigned long long *_dIds, CameraCU _cam, BaseCU _base);

		unsigned int vn_ = 0;
		unsigned int fn_ = 0;
		int w_ = 0;
		int h_ = 0;
		BaseCU baseDev_;
		CameraCU camDev_;
		float3 *vsDev_ = nullptr;
		uint3 *fsDev_ = nullptr;
		float *depthsDev_ = nullptr;
	};

}

#endif // !CUDA_LYJPROJECTOR_H
