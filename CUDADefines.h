#ifndef CUDA_LYJ_DEFINES_H
#define CUDA_LYJ_DEFINES_H


#include "CUDACommon.h"
#include <vector>


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
	union DepthID2
	{
		struct
		{
			float depth;
			unsigned int fid;
		};
		unsigned long long data;
	};


	class CUDA_LYJ_API ProjectorCache
	{
	public:
		ProjectorCache() {};
		ProjectorCache(unsigned int _PSize, unsigned int _fSize, int _w, int _h);
		~ProjectorCache();

		unsigned int PSize_ = 0;
		unsigned int fSize_ = 0;
		int w_ = 0;
		int h_ = 0;

		Mat34CU TDev_;
		float3* PcsDev_;
		float3* ctrcsDev_;
		float3* fNormalcsDev_;
		float3* pixelsDev_;
		float3* ctrPixelsDev_;

		float* depthDev_ = nullptr;
		unsigned long long* dIdsDev_;
		char* isPVisibleDev_;
		char* isFVisibleDev_;
		std::vector<DepthID2> dIds_;

		void init(unsigned int _PSize, unsigned int _fSize, int _w, int _h);
	};
}




#endif // !CUDA_LYJ_DEFINES_H
