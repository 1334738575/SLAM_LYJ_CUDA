#ifndef CUDA_INCLUDE_H
#define CUDA_INCLUDE_H

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

namespace CUDA_LYJ
{
	CUDA_LYJ_API void testTexture(); // texture2d

	typedef void *ProHandle;
	CUDA_LYJ_API ProHandle initProjector(
		const float *Pws, const unsigned int PSize,
		const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
		float *camParams, const int w, const int h);
	CUDA_LYJ_API void project(ProHandle handle,
							  float *Tcw,
							  float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
							  float minD = 0, float maxD = FLT_MAX, float csTh = 0, float detDTh = 1);
	CUDA_LYJ_API void release(ProHandle handle);
}

#endif // !CUDA_INCLUDE_H
