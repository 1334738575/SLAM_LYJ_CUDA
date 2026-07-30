#ifndef CUDA_LYJ_SIFT_MATCHER_H
#define CUDA_LYJ_SIFT_MATCHER_H

#include "CUDADefines.h"

namespace CUDA_LYJ
{
	class SIFTMatcherCU
	{
	public:
		void matchBF(SIFTMatcherCache& _cache, float _distMax, float _ratioMax,
			char _mutualBestMatch, char _bUse3D, float _squareDistTh3D);

	private:
		void matchBFCUDA(int _kp1Sz, int _kp2Sz,
			Mat34CU& _Twc1, Mat34CU& _Twc2,
			const float* _descs1, const float* _descs2,
			const float3* _Pcs1, const float3* _Pcs2,
			float _distMax, float _ratioMax, char _mutualBestMatch,
			char _bUse3D, float _squareDistTh3D,
			short* _match2to1, short* _reverseMatch);
	};
}

#endif
