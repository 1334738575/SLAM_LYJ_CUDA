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
	CUDA_LYJ_API void test1(); // add
	CUDA_LYJ_API void test2(); // project, abort
	CUDA_LYJ_API void test3(); // texture2d

	typedef void *ProHandle;
	CUDA_LYJ_API ProHandle initProjector(
		const float *Pws, const unsigned int PSize,
		const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
		float *camParams, const int w, const int h);
	CUDA_LYJ_API void project(ProHandle handle,
							  float *Tcw,
							  float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds);
	CUDA_LYJ_API void release(ProHandle handle);
}

#endif // !CUDA_INCLUDE_H
