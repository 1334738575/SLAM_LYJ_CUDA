#ifndef CUDA_LYJ_DEFINES_H
#define CUDA_LYJ_DEFINES_H


#include "CUDACommon.cuh"
#include <stdint.h>



namespace CUDA_LYJ
{
	enum class CameraModel : uint32_t
	{
		Pinhole = 0,
		Fisheye = 1,
	};

	union CUDA_LYJ_API DepthID2
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
		bool hasPcs1_ = false;
		bool hasPcs2_ = false;
		bool hasPws1_ = false;
		bool hasDescs1_ = false;
		bool hasDescs2_ = false;


		short* match2to1Dev_;
		short* reverseMatchDev_;

		void init();
		// 3D inputs may be null when only descriptor matching is required.
		void upload1(int _kpSz, float* _Tcw, float* _Twc, float* _kps, unsigned int* _descs, float* _Pcs = nullptr, float* _Pws = nullptr, char* _bPws = nullptr);
		void upload2(int _kpSz, float* _Tcw, float* _Twc, short* _featureGrid, char* eveFeatureGrid, float* _kps, unsigned int* _descs, float* _Pcs = nullptr);
		void upload1(int _kpSz, const unsigned int* _descs);
		void upload2(int _kpSz, const unsigned int* _descs);
	};

	class CUDA_LYJ_API SIFTMatcherCache
	{
	public:
		SIFTMatcherCache();
		~SIFTMatcherCache();

		int kpSz1_ = 0;
		int kpSz2_ = 0;
		Mat34CU Twc1Dev_;
		Mat34CU Twc2Dev_;
		float* descs1Dev_ = nullptr;
		float* descs2Dev_ = nullptr;
		float3* Pcs1Dev_ = nullptr;
		float3* Pcs2Dev_ = nullptr;
		short* match2to1Dev_ = nullptr;
		short* reverseMatchDev_ = nullptr;
		bool hasDescs1_ = false;
		bool hasDescs2_ = false;
		bool hasPcs1_ = false;
		bool hasPcs2_ = false;

		void init();
		// SiftGPU angular distance requires L2-normalized descriptors.
		void upload1(int _kpSz, float* _Twc, const float* _descs, const float* _Pcs = nullptr,
			bool _normalizeDescriptors = true);
		void upload2(int _kpSz, float* _Twc, const float* _descs, const float* _Pcs = nullptr,
			bool _normalizeDescriptors = true);
		void upload1(int _kpSz, const float* _descs, bool _normalizeDescriptors = true);
		void upload2(int _kpSz, const float* _descs, bool _normalizeDescriptors = true);
	};
}




#endif // !CUDA_LYJ_DEFINES_H
