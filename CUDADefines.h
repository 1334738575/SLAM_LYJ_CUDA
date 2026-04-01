#ifndef CUDA_LYJ_DEFINES_H
#define CUDA_LYJ_DEFINES_H


#include "CUDACommon.h"


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

#define CUDAORBMAXW 2000
#define CUDAORBMAXH 2000
#define CUDAORBKPSIZE 8192
#define CUDAORBGRIDSOLU 20
#define CUDAORBEVECELLSIZE 128

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

	class CUDA_LYJ_API ORBMatcherCache
	{
	public:
		ORBMatcherCache();
		~ORBMatcherCache();

		int wGrid_ = 0;
		int hGrid_ = 0;
		int kpSz1_ = 0;//最大8192
		int kpSz2_ = 0;//最大8192
		//int distThDesc = 50;
		//float nnTh = 0.6;
		//char bCheckOri = 1;
		//char bUse3D = 0;
		//float squareDistTh3D = 1;

		Mat34CU Tcw1Dev_;
		Mat34CU Twc1Dev_;
		Mat34CU Tcw2Dev_;
		Mat34CU Twc2Dev_;
		short* featureGrid2Dev_;//固定w_/20 * h_/20 * 128; 16位够用
		char* eveFeatureGrid2SzDev_;//每个各自的特征数
		float2* kps1Dev_;
		float2* kps2Dev_;
		unsigned int* descs1Dev_;
		unsigned int* descs2Dev_;
		float3* Pcs1Dev_;
		float3* Pcs2Dev_;
		float3* Pws1Dev_;
		char* bPws1Dev_;


		short* match2to1Dev_;

		void init();
		void upload1(int _kpSz, float* _Tcw, float* _Twc, float* _kps, unsigned int* _descs, float* _Pcs, float* _Pws, char* _bPws);
		void upload2(int _kpSz, float* _Tcw, float* _Twc, short* _featureGrid, char* eveFeatureGrid, float* _kps, unsigned int* _descs, float* _Pcs);
	};

}




#endif // !CUDA_LYJ_DEFINES_H
