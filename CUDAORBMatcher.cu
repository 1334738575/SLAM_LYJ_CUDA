#include "CUDAORBMatcher.h"


namespace CUDA_LYJ
{
	//__device__ float dot3(const float3& p1, const float3& p2)
	//{
	//	return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
	//}	
	__device__ void pointToImageTmp(float* _cam, const float3 &_p3d, float& _u, float& _v)
	{
		float invZ = 1.0f / _p3d.z;
		_u = _p3d.x * _cam[0] * invZ + _cam[2];
		_v = _p3d.y * _cam[1] * invZ + _cam[3];
	}
	__device__ void transformTmp(float* _T, const float3& _p, float3& _ret)
	{
		_ret.x = _T[0] * _p.x + _T[3] * _p.y + _T[6] * _p.z + _T[9];
		_ret.y = _T[1] * _p.x + _T[4] * _p.y + _T[7] * _p.z + _T[10];
		_ret.z = _T[2] * _p.x + _T[5] * _p.y + _T[8] * _p.z + _T[11];
		return;
	}
	__device__ float Point3DSquareDistance(float3& p1, float3& p2)
	{
		return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
	}
	__device__ int DescriptorDistance(unsigned int* a, unsigned int* b)
	{
		int dist = 0;
#pragma unroll 8
		for (int i = 0; i < 8; ++i) {
			unsigned int v = a[i] ^ b[i];
			dist += __popc(v);
		}
		return dist;
	}



	__global__ void matchBFCU(int _kp1Sz, int _kp2Sz, 
		float* _Twc1, float* _Twc2, 
		unsigned int* _descs1, unsigned int* _descs2, 
		float3* _Pcs1, float3* _Pcs2,
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
		short* _match2to1)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;
		_match2to1[idx] = -1;

		unsigned int* desc1 = _descs1 + 8 * idx;
		int bestId = -1;
		int bestDescDist = 255;
		int nextBestId = -1;
		int nextBestDescDist = 255;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		float3* p1 = _Pcs1 + idx;
		transformTmp(_Twc1, p1[0], Pw1);
		if (_bUse3D == 1)
		{
			if (p1[0].z <= 0)
				return;
		}
		float3 Pw2;
		for (int i = 0; i < _kp2Sz; ++i)
		{
			if (_bUse3D == 1)
			{
				float3* p2 = _Pcs2 + i;
				if (p2[0].z <= 0)
					continue;
				transformTmp(_Twc2, p2[0], Pw2);
				squareDistTmp = Point3DSquareDistance(Pw1, Pw2);
				if (squareDistTmp > _squareDistTh3D)
					continue;
			}
			unsigned int* desc2 = _descs2 + 8 * i;
			descDistTmp = DescriptorDistance(desc1, desc2);
			if (descDistTmp > _distThDesc)
				continue;
			if (descDistTmp < bestDescDist)
			{
				nextBestDescDist = bestDescDist;
				nextBestId = bestId;
				bestDescDist = descDistTmp;
				bestId = i;
			}
			else if (descDistTmp < nextBestDescDist)
			{
				nextBestDescDist = descDistTmp;
				nextBestId = i;
			}
		}
		if (bestDescDist > nextBestDescDist * _nnTh)
			return;
		_match2to1[idx] = bestId;
	}
	void ORBMatcherCU::matchBFCUDA(int _kp1Sz, int _kp2Sz, 
		Mat34CU& _Twc1, Mat34CU& _Twc2, 
		unsigned int* _descs1, unsigned int* _descs2, 
		float3* _Pcs1, float3* _Pcs2, 
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D, 
		short* _match2to1)
	{
		int threadNum = 1024;
		int gridSz = (CUDAORBKPSIZE + threadNum - 1) / threadNum;
		dim3 block(threadNum, 1);
		dim3 grid(gridSz, 1);
		matchBFCU << <grid, block >> > (_kp1Sz, _kp2Sz, _Twc1.dataDev_, _Twc2.dataDev_, _descs1, _descs2, _Pcs1, _Pcs2,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D, _match2to1);
	}



	__device__ int getIndInGrid(float* _Tcw, 
		int _wGrid, int _hGrid, int _gridResul, short* _featureGrid, char* eveFeatureGridSz,
		float3& _Pw,
		float* _cam,
		int& _cGrid, int& _rGrid
		)
	{
		float3 Pc;
		transformTmp(_Tcw, _Pw, Pc);
		float u, v;
		pointToImageTmp(_cam, _Pw, u, v);
		_cGrid = int(u) / _gridResul;
		_rGrid = int(v) / _gridResul;
		if (_cGrid < 0 || _rGrid < 0 || _cGrid >= _wGrid || _rGrid >= _hGrid)
			return 0;
		return 1;
	}
	__device__ void getKpIndInCell(int _wGrid, int _hGrid, short* _featureGrid, char* eveFeatureGridSz,
		const int& _cGrid, const int& _rGrid,
		short* _kpIndSt, int& _kpIndSz
	)
	{
		_kpIndSt = _featureGrid + (_rGrid * _wGrid * CUDAORBEVECELLSIZE + _cGrid * CUDAORBEVECELLSIZE);
		_kpIndSz = int(eveFeatureGridSz[_rGrid * _wGrid + _cGrid]);
	}
	__global__ void matchProCU(int _kp1Sz, int _kp2Sz,
		float* _Twc1, float* _Twc2,
		float* _Tcw1, float* _Tcw2,
		int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz,
		unsigned int* _descs1, unsigned int* _descs2,
		float3* _Pcs1, float3* _Pcs2,
		float3* _Pws1,
		char* _bPws1,
		float* _cam,
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
		short* _match2to1)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;
		_match2to1[idx] = -1;

		if (_bPws1[idx] == 0)
			return;
		float3* Pw = _Pws1 + idx;
		int cGrid, int rGrid;
		if (getIndInGrid(_Tcw2, _wGrid2, _hGrid2, _gridResul, _featureGrid2, eveFeatureGrid2Sz, Pw[0], _cam, cGrid, rGrid) == 0)
			return;

		unsigned int* desc1 = _descs1 + 8 * idx;
		int bestId = -1;
		int bestDescDist = 255;
		int nextBestId = -1;
		int nextBestDescDist = 255;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		float3* p1 = _Pcs1 + idx;
		transformTmp(_Twc1, p1[0], Pw1);
		if (_bUse3D == 1)
		{
			if (p1[0].z <= 0)
				return;
		}
		float3 Pw2;
		short* kpIndSt;
		int kpIndSz;
		for (int i = rGrid-1; i <= rGrid + 1; ++i)
		{
			for (int j = cGrid-1; j <= cGrid+1; ++j)
			{
				if (i < 0 || j < 0 || i >= _wGrid2 || j >= _hGrid2)
					 continue;
				getKpIndInCell(_wGrid2, _hGrid2, _featureGrid2, eveFeatureGrid2Sz, j, i, kpIndSt, kpIndSz);
				if (kpIndSz <= 0)
					continue;
				for (int k = 0; k < kpIndSz; ++k)
				{
					int kpInd = int(kpIndSt[k]);
					if (_bUse3D == 1)
					{
						float3* p2 = _Pcs2 + kpInd;
						if (p2[0].z <= 0)
							continue;
						transformTmp(_Twc2, p2[0], Pw2);
						squareDistTmp = Point3DSquareDistance(Pw1, Pw2);
						if (squareDistTmp > _squareDistTh3D)
							continue;
					}
					unsigned int* desc2 = _descs2 + 8 * kpInd;
					descDistTmp = DescriptorDistance(desc1, desc2);
					if (descDistTmp > _distThDesc)
						continue;
					if (descDistTmp < bestDescDist)
					{
						nextBestDescDist = bestDescDist;
						nextBestId = bestId;
						bestDescDist = descDistTmp;
						bestId = kpInd;
					}
					else if (descDistTmp < nextBestDescDist)
					{
						nextBestDescDist = descDistTmp;
						nextBestId = kpInd;
					}
				}
			}
		}
		if (bestDescDist > nextBestDescDist * _nnTh)
			return;
		_match2to1[idx] = bestId;
	}
	void ORBMatcherCU::matchProCUDA(int _kp1Sz, int _kp2Sz, 
		Mat34CU& _Twc1, Mat34CU& _Twc2, 
		Mat34CU& _Tcw1, Mat34CU& _Tcw2, 
		int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz,
		unsigned int* _descs1, unsigned int* _descs2, 
		float3* _Pcs1, float3* _Pcs2, 
		float3* _Pws1, char* _bPws1, 
		CameraCU& _cam, 
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D, 
		short* _match2to1)
	{
		int threadNum = 1024;
		int gridSz = (CUDAORBKPSIZE + threadNum - 1) / threadNum;
		dim3 block(threadNum, 1);
		dim3 grid(gridSz, 1);
		matchProCU << <grid, block >> > (_kp1Sz, _kp2Sz,
			_Twc1.dataDev_, _Twc2.dataDev_,
			_Tcw1.dataDev_, _Tcw2.dataDev_,
			_wGrid2, _hGrid2, _gridResul, _featureGrid2, eveFeatureGrid2Sz,
			_descs1, _descs2,
			_Pcs1, _Pcs2,
			_Pws1, _bPws1,
			_cam.paramsDev_,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
			_match2to1
			);
	}



}