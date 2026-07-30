#include "CUDAInclude.h"
#include "CUDASIFTMatcher.h"

namespace CUDA_LYJ
{
	CUDA_LYJ_API SIFTMatchHandle initSIFTMatcher()
	{
		return static_cast<void*>(new SIFTMatcherCU());
	}

	static void getSIFTMatchResult(SIFTMatcherCache& cache, short* matched2to1, short* matched1to2)
	{
		cudaMemcpy(matched2to1, cache.match2to1Dev_, cache.kpSz1_ * sizeof(short), cudaMemcpyDeviceToHost);
		for (int i = 0; i < cache.kpSz2_; ++i)
			matched1to2[i] = -1;
		for (int i = 0; i < cache.kpSz1_; ++i)
		{
			const int match = static_cast<int>(matched2to1[i]);
			if (match >= 0 && match < cache.kpSz2_ && matched1to2[match] == -1)
				matched1to2[match] = static_cast<short>(i);
		}
	}

	CUDA_LYJ_API void matchBF(SIFTMatchHandle handle, SIFTMatcherCache& cache,
		short* matched2to1, short* matched1to2,
		float distMax, float ratioMax, char mutualBestMatch, char bUse3D, float squareDistTh3D)
	{
		SIFTMatcherCU* matcher = static_cast<SIFTMatcherCU*>(handle);
		matcher->matchBF(cache, distMax, ratioMax, mutualBestMatch, bUse3D, squareDistTh3D);
		cudaDeviceSynchronize();
		getSIFTMatchResult(cache, matched2to1, matched1to2);
	}

	CUDA_LYJ_API void releaseSIFTMatcher(SIFTMatchHandle handle)
	{
		delete static_cast<SIFTMatcherCU*>(handle);
	}
}
