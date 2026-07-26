#include "CUDADefines.h"


namespace CUDA_LYJ
{

	ProjectorCache::ProjectorCache(unsigned int _PSize, unsigned int _fSize, int _w, int _h)
	:PSize_(_PSize), fSize_(_fSize), w_(_w), h_(_h)
	{
		init(PSize_, fSize_, w_, h_);
	}
	ProjectorCache::~ProjectorCache()
	{
		cudaFree(pixelsDev_);
		cudaFree(ctrPixelsDev_);
		cudaFree(depthDev_);
		cudaFree(dIdsDev_);
		cudaFree(isPVisibleDev_);
		cudaFree(isFVisibleDev_);
	}
	void ProjectorCache::init(unsigned int _PSize, unsigned int _fSize, int _w, int _h)
	{
		PSize_ = _PSize;
		fSize_ = _fSize;
		w_ = _w;
		h_ = _h;

		dIds_.resize(w_ * h_);
		for (auto& did : dIds_)
		{
			did.depth = FLT_MAX;
			did.fid = UINT_MAX;
		}
		cudaMalloc((void**)&pixelsDev_, PSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&ctrPixelsDev_, fSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&depthDev_, w_ * h_ * sizeof(float));
		cudaMalloc((void**)&dIdsDev_, w_ * h_ * sizeof(unsigned long long));
		cudaMalloc((void**)&isPVisibleDev_, PSize_ * sizeof(char));
		cudaMalloc((void**)&isFVisibleDev_, fSize_ * sizeof(char));
	}

	ORBMatcherCache::ORBMatcherCache()
	{
		init();
	}
	ORBMatcherCache::~ORBMatcherCache()
	{
		cudaFree(featureGrid2Dev_);
		cudaFree(eveFeatureGrid2SzDev_);
		cudaFree(kps1Dev_);
		cudaFree(kps2Dev_);
		cudaFree(descs1Dev_);
		cudaFree(descs2Dev_);
		cudaFree(Pcs1Dev_);
		cudaFree(Pcs2Dev_);
		cudaFree(Pws1Dev_);
		cudaFree(bPws1Dev_);

		cudaFree(match2to1Dev_);
		cudaFree(reverseMatchDev_);
	}
	void ORBMatcherCache::init()
	{
		wGrid_ = (CUDAORBMAXW + CUDAORBGRIDSOLU - 1) / CUDAORBGRIDSOLU;
		hGrid_ = (CUDAORBMAXH + CUDAORBGRIDSOLU - 1) / CUDAORBGRIDSOLU;

		cudaMalloc((void**)&featureGrid2Dev_, wGrid_ * hGrid_ * CUDAORBEVECELLSIZE * sizeof(short));
		cudaMalloc((void**)&eveFeatureGrid2SzDev_, wGrid_ * hGrid_ * sizeof(char));
		cudaMalloc((void**)&kps1Dev_, CUDAORBKPSIZE * 2 * sizeof(float));
		cudaMalloc((void**)&kps2Dev_, CUDAORBKPSIZE * 2 * sizeof(float));
		cudaMalloc((void**)&descs1Dev_, CUDAORBKPSIZE * 8 * sizeof(unsigned int));
		cudaMalloc((void**)&descs2Dev_, CUDAORBKPSIZE * 8 * sizeof(unsigned int));
		cudaMalloc((void**)&Pcs1Dev_, CUDAORBKPSIZE * sizeof(float3));
		cudaMalloc((void**)&Pcs2Dev_, CUDAORBKPSIZE * sizeof(float3));
		cudaMalloc((void**)&Pws1Dev_, CUDAORBKPSIZE * sizeof(float3));
		cudaMalloc((void**)&bPws1Dev_, CUDAORBKPSIZE * sizeof(char));

		cudaMalloc((void**)&match2to1Dev_, CUDAORBKPSIZE * sizeof(short));
		cudaMalloc((void**)&reverseMatchDev_, CUDAORBKPSIZE * sizeof(short));
	}
	void ORBMatcherCache::upload1(int _kpSz, float* _Tcw, float* _Twc, float* _kps, unsigned int* _descs, float* _Pcs, float* _Pws, char* _bPws)
	{
		kpSz1_ = _kpSz < 0 ? 0 : (_kpSz > CUDAORBKPSIZE ? CUDAORBKPSIZE : _kpSz);
		Tcw1Dev_.upload(_Tcw);
		Twc1Dev_.upload(_Twc);
		cudaMemcpy(kps1Dev_, _kps, kpSz1_ * 2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(descs1Dev_, _descs, kpSz1_ * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		hasPcs1_ = _Pcs != nullptr;
		hasPws1_ = _Pws != nullptr && _bPws != nullptr;
		if (_Pcs != nullptr)
			cudaMemcpy(Pcs1Dev_, _Pcs, kpSz1_ * sizeof(float3), cudaMemcpyHostToDevice);
		if (_Pws != nullptr)
			cudaMemcpy(Pws1Dev_, _Pws, kpSz1_ * sizeof(float3), cudaMemcpyHostToDevice);
		if (_bPws != nullptr)
			cudaMemcpy(bPws1Dev_, _bPws, kpSz1_ * sizeof(char), cudaMemcpyHostToDevice);
	}
	void ORBMatcherCache::upload2(int _kpSz, float* _Tcw, float* _Twc, short* _featureGrid, char* eveFeatureGrid, float* _kps, unsigned int* _descs, float* _Pcs)
	{
		kpSz2_ = _kpSz < 0 ? 0 : (_kpSz > CUDAORBKPSIZE ? CUDAORBKPSIZE : _kpSz);
		Tcw2Dev_.upload(_Tcw);
		Twc2Dev_.upload(_Twc);
		cudaMemcpy(featureGrid2Dev_, _featureGrid, wGrid_ * hGrid_ * CUDAORBEVECELLSIZE * sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(eveFeatureGrid2SzDev_, eveFeatureGrid, wGrid_ * hGrid_ * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(kps2Dev_, _kps, kpSz2_ * 2 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(descs2Dev_, _descs, kpSz2_ * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);
		hasPcs2_ = _Pcs != nullptr;
		if (_Pcs != nullptr)
			cudaMemcpy(Pcs2Dev_, _Pcs, kpSz2_ * sizeof(float3), cudaMemcpyHostToDevice);
	}

}
