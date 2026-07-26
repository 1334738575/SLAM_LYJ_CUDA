#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H

#include "CUDADefines.h"
#include <stdint.h>


namespace CUDA_LYJ
{
	CUDA_LYJ_API void test();
	CUDA_LYJ_API void testTexture(); // texture2d

	typedef void *ProHandle;
	//CUDA_LYJ_API ProHandle initProjector(
	//	const float *Pws, const unsigned int PSize,
	//	const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
	//	float *camParams, const int w, const int h);
	//CUDA_LYJ_API void project(ProHandle handle,
	//						  float *Tcw,
	//						  float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
	//						  float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1);
	//CUDA_LYJ_API void release(ProHandle handle);

	CUDA_LYJ_API ProHandle initProjector(
		const float* Pws, const unsigned int PSize,
		const float* centers, const float* fNormals, const unsigned int* faces, const unsigned int fSize,
		float* camParams, const int w, const int h);
	//CUDA_LYJ_API void project(ProHandle handle,
	//	float* Tcw,
	//	float* depths, unsigned int* fIds, char* allVisiblePIds, char* allVisibleFIds,
	//	float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1);
	CUDA_LYJ_API void project(ProHandle handle, ProjectorCache& cache,
		float* Tcw,
		float* depths, unsigned int* fIds, char* allVisiblePIds, char* allVisibleFIds,
		float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1,
		std::vector<uint32_t>* faceIds = nullptr, std::vector<uint32_t>* pointIds = nullptr);
	CUDA_LYJ_API void release(ProHandle handle);


	typedef void* MatchHanlde;
	CUDA_LYJ_API MatchHanlde initMatcher(
		int _w, int _h, float* _cam
	);
	CUDA_LYJ_API void matchBF(MatchHanlde handle, ORBMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D);
	CUDA_LYJ_API void matchF(MatchHanlde handle, ORBMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D);
	CUDA_LYJ_API void matchPro(MatchHanlde handle, ORBMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D);
	CUDA_LYJ_API void releaseMatcher(MatchHanlde handle);
}

#endif // !CUDA_INCLUDE_H
