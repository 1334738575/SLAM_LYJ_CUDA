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
		int _w = 0, int _h = 0, float* _cam = nullptr
	);
	CUDA_LYJ_API void matchBF(MatchHanlde handle, ORBMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		int _distThDesc = 64, float _nnTh = 0.8f, char _bCheckOri = 0, char _bUse3D = 0, float _squareDistTh3D = 0.0f);
	CUDA_LYJ_API void matchF(MatchHanlde handle, ORBMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D);
	CUDA_LYJ_API void matchPro(MatchHanlde handle, ORBMatcherCache& cache,
		GridCU& _gridCom,
		short* matched2to1, short* matched1to2,
		int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D);
	CUDA_LYJ_API void releaseMatcher(MatchHanlde handle);

	typedef void* SIFTMatchHandle;
	CUDA_LYJ_API SIFTMatchHandle initSIFTMatcher();
	// distMax is acos(cosine similarity), in radians; defaults match SiftGPU.
	CUDA_LYJ_API void matchBF(SIFTMatchHandle handle, SIFTMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		float _distMax = 0.7f, float _ratioMax = 0.8f, char _mutualBestMatch = 1,
		char _bUse3D = 0, float _squareDistTh3D = 0.0f);
	CUDA_LYJ_API void releaseSIFTMatcher(SIFTMatchHandle handle);
}

#endif // !CUDA_INCLUDE_H
