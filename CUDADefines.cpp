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
		cudaFree(PcsDev_);
		cudaFree(ctrcsDev_);
		cudaFree(fNormalcsDev_);
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
		cudaMalloc((void**)&PcsDev_, PSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&ctrcsDev_, fSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&fNormalcsDev_, fSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&pixelsDev_, PSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&ctrPixelsDev_, fSize_ * 3 * sizeof(float));
		cudaMalloc((void**)&depthDev_, w_ * h_ * sizeof(float));
		cudaMalloc((void**)&dIdsDev_, w_ * h_ * sizeof(unsigned long long));
		cudaMalloc((void**)&isPVisibleDev_, PSize_ * sizeof(char));
		cudaMalloc((void**)&isFVisibleDev_, fSize_ * sizeof(char));
	}

}