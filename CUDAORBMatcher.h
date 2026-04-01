#ifndef CUDA_LYJORBMATCHER_H
#define CUDA_LYJORBMATCHER_H


#include "CUDACommon.h"
#include "CUDADefines.h"


namespace CUDA_LYJ
{

	class ORBMatcherCU
	{
	public:
		ORBMatcherCU(int _w, int _h, float* _cam) 
		{
			init(_w, _h, _cam);
		};
		~ORBMatcherCU() {};

		void init(int _w, int _h, float* _cam) 
		{
			w_ = _w;
			h_ = _h;
			camDev_.upload(_w, _h, _cam);
		};

		void matchBF(ORBMatcherCache& _cache, int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D)
		{
			matchBFCUDA(_cache.kpSz1_, _cache.kpSz2_, _cache.Twc1Dev_, _cache.Twc2Dev_, _cache.descs1Dev_, _cache.descs2Dev_, _cache.Pcs1Dev_, _cache.Pcs2Dev_,
				_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
				_cache.match2to1Dev_);
		}
		void matchF(ORBMatcherCache& _cache, int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D)
		{
			matchFCUDA(_cache.kpSz1_, _cache.kpSz2_, _cache.Twc1Dev_, _cache.Twc2Dev_, _cache.Tcw1Dev_, _cache.Tcw2Dev_,
				_cache.wGrid_, _cache.hGrid_, CUDAORBGRIDSOLU, _cache.featureGrid2Dev_, _cache.eveFeatureGrid2SzDev_,
				_cache.kps1Dev_, _cache.kps2Dev_, _cache.descs1Dev_, _cache.descs2Dev_, _cache.Pcs1Dev_, _cache.Pcs2Dev_,
				camDev_,
				_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
				_cache.match2to1Dev_);
		}
		void matchPro(ORBMatcherCache& _cache, int _distThDesc, float _nnTh, char _bCheckOri, char _bUse3D, float _squareDistTh3D)
		{
			matchProCUDA(_cache.kpSz1_, _cache.kpSz2_, _cache.Twc1Dev_, _cache.Twc2Dev_, _cache.Tcw1Dev_, _cache.Tcw2Dev_,
				_cache.wGrid_, _cache.hGrid_, CUDAORBGRIDSOLU, _cache.featureGrid2Dev_, _cache.eveFeatureGrid2SzDev_,
				_cache.descs1Dev_, _cache.descs2Dev_, _cache.Pcs1Dev_, _cache.Pcs2Dev_,
				_cache.Pws1Dev_, _cache.bPws1Dev_, 
				camDev_,
				_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
				_cache.match2to1Dev_);
		}

	private:
		void matchBFCUDA(int _kp1Sz, int _kp2Sz, 
			Mat34CU& _Twc1, Mat34CU& _Twc2,
			unsigned int* _descs1, unsigned int* _descs2,
			float3* _Pcs1, float3* _Pcs2,
			int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
			short* _match2to1);
		void matchFCUDA(int _kp1Sz, int _kp2Sz,
			Mat34CU& _Twc1, Mat34CU& _Twc2,
			Mat34CU& _Tcw1, Mat34CU& _Tcw2,
			int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz,
			float2* _kps1, float2* _kps2,
			unsigned int* _descs1, unsigned int* _descs2,
			float3* _Pcs1, float3* _Pcs2,
			CameraCU& _cam,
			int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
			short* _match2to1);
		void matchProCUDA(int _kp1Sz, int _kp2Sz,
			Mat34CU& _Twc1, Mat34CU& _Twc2,
			Mat34CU& _Tcw1, Mat34CU& _Tcw2,
			int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2,	char* eveFeatureGrid2Sz,
			unsigned int* _descs1, unsigned int* _descs2,
			float3* _Pcs1, float3* _Pcs2,
			float3* _Pws1, char* _bPws1,
			CameraCU& _cam,
			int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
			short* _match2to1);

		int w_ = 0;
		int h_ = 0;
		CameraCU camDev_;

	};




}





#endif // !CUDA_LYJORBMATCHER_H
