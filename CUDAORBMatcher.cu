#include "CUDAORBMatcher.h"


namespace CUDA_LYJ
{
	//__device__ float dot3(const float3& p1, const float3& p2)
	//{
	//	return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
	//}	
	__device__ void pointToImageTmp(float* _cam, unsigned int _cameraModel,
		const float3 &_p3d, float& _u, float& _v)
	{
		projectCameraPoint(_cam, _cameraModel, _p3d, _u, _v);
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
		int bestDescDist = 257;
		int nextBestDescDist = 257;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		if (_bUse3D == 1)
		{
			float3* p1 = _Pcs1 + idx;
			if (p1[0].z <= 0)
				return;
			transformTmp(_Twc1, p1[0], Pw1);
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
			if (descDistTmp < bestDescDist)
			{
				nextBestDescDist = bestDescDist;
				bestDescDist = descDistTmp;
				bestId = i;
			}
			else if (descDistTmp < nextBestDescDist)
			{
				nextBestDescDist = descDistTmp;
			}
		}
		if (bestId < 0 || bestDescDist > _distThDesc || nextBestDescDist > 256 ||
			bestDescDist >= nextBestDescDist * _nnTh)
			return;
		_match2to1[idx] = static_cast<short>(bestId);
	}

	__global__ void keepMutualMatchesCU(int _kp1Sz, int _kp2Sz,
		short* _match2to1, const short* _reverseMatch)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;

		int match = static_cast<int>(_match2to1[idx]);
		if (match < 0 || match >= _kp2Sz || static_cast<int>(_reverseMatch[match]) != idx)
			_match2to1[idx] = -1;
	}

	void ORBMatcherCU::matchBFCUDA(int _kp1Sz, int _kp2Sz, 
		Mat34CU& _Twc1, Mat34CU& _Twc2, 
		unsigned int* _descs1, unsigned int* _descs2, 
		float3* _Pcs1, float3* _Pcs2, 
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D, 
		short* _match2to1, short* _reverseMatch)
	{
		if (_kp1Sz > 0)
			cudaMemset(_match2to1, 0xff, _kp1Sz * sizeof(short));
		if (_kp2Sz > 0)
			cudaMemset(_reverseMatch, 0xff, _kp2Sz * sizeof(short));
		if (_kp1Sz < 2 || _kp2Sz < 2 || _distThDesc < 0 || _distThDesc > 256 ||
			_nnTh != _nnTh || _nnTh <= 0.0f || _nnTh >= 1.0f)
			return;

		constexpr int threadNum = 256;
		dim3 block(threadNum, 1);
		dim3 forwardGrid((_kp1Sz + threadNum - 1) / threadNum, 1);
		dim3 reverseGrid((_kp2Sz + threadNum - 1) / threadNum, 1);
		matchBFCU << <forwardGrid, block >> > (_kp1Sz, _kp2Sz, _Twc1.dataDev_, _Twc2.dataDev_, _descs1, _descs2, _Pcs1, _Pcs2,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D, _match2to1);
		matchBFCU << <reverseGrid, block >> > (_kp2Sz, _kp1Sz, _Twc2.dataDev_, _Twc1.dataDev_, _descs2, _descs1, _Pcs2, _Pcs1,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D, _reverseMatch);
		keepMutualMatchesCU << <forwardGrid, block >> > (_kp1Sz, _kp2Sz, _match2to1, _reverseMatch);
	}


	__device__ int getIndInGrid(float* _Tcw,
		int _wGrid, int _hGrid, int _gridResul, short* _featureGrid, char* eveFeatureGridSz,
		float3& _Pw,
		float* _cam, unsigned int _cameraModel,
		int& _cGrid, int& _rGrid
	)
	{
		float3 Pc;
		transformTmp(_Tcw, _Pw, Pc);
		float u, v;
		if (Pc.z <= 0.0f)
			return 0;
		pointToImageTmp(_cam, _cameraModel, Pc, u, v);
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
	__device__ void getF(float* _Tcw2, float* _Twc1, float* _cam, float* _F21)
	{
		// ==========================
		// 直接读取位姿：无任何局部数组！
		// ==========================
		// Twc1: 相机1->世界
		const float R1_00 = _Twc1[0], R1_01 = _Twc1[3], R1_02 = _Twc1[6];
		const float R1_10 = _Twc1[1], R1_11 = _Twc1[4], R1_12 = _Twc1[7];
		const float R1_20 = _Twc1[2], R1_21 = _Twc1[5], R1_22 = _Twc1[8];
		const float t1_0 = _Twc1[9], t1_1 = _Twc1[10], t1_2 = _Twc1[11];

		// Tcw2: 世界->相机2
		const float R2_00 = _Tcw2[0], R2_01 = _Tcw2[3], R2_02 = _Tcw2[6];
		const float R2_10 = _Tcw2[1], R2_11 = _Tcw2[4], R2_12 = _Tcw2[7];
		const float R2_20 = _Tcw2[2], R2_21 = _Tcw2[5], R2_22 = _Tcw2[8];
		const float t2_0 = _Tcw2[9], t2_1 = _Tcw2[10], t2_2 = _Tcw2[11];

		// ==========================
		// 相对位姿 R = R2*R1
		// ==========================
		const float R00 = R2_00 * R1_00 + R2_01 * R1_10 + R2_02 * R1_20;
		const float R01 = R2_00 * R1_01 + R2_01 * R1_11 + R2_02 * R1_21;
		const float R02 = R2_00 * R1_02 + R2_01 * R1_12 + R2_02 * R1_22;
		const float R10 = R2_10 * R1_00 + R2_11 * R1_10 + R2_12 * R1_20;
		const float R11 = R2_10 * R1_01 + R2_11 * R1_11 + R2_12 * R1_21;
		const float R12 = R2_10 * R1_02 + R2_11 * R1_12 + R2_12 * R1_22;
		const float R20 = R2_20 * R1_00 + R2_21 * R1_10 + R2_22 * R1_20;
		const float R21 = R2_20 * R1_01 + R2_21 * R1_11 + R2_22 * R1_21;
		const float R22 = R2_20 * R1_02 + R2_21 * R1_12 + R2_22 * R1_22;

		// ==========================
		// 相对平移 t = R2*t1 + t2
		// ==========================
		const float t0 = R2_00 * t1_0 + R2_01 * t1_1 + R2_02 * t1_2 + t2_0;
		const float t1 = R2_10 * t1_0 + R2_11 * t1_1 + R2_12 * t1_2 + t2_1;
		const float t2 = R2_20 * t1_0 + R2_21 * t1_1 + R2_22 * t1_2 + t2_2;

		// ==========================
		// 本质矩阵 E = R * [t]x
		// ==========================
		const float E00 = R01 * t2 - R02 * t1;
		const float E01 = R02 * t0 - R00 * t2;
		const float E02 = R00 * t1 - R01 * t0;
		const float E10 = R11 * t2 - R12 * t1;
		const float E11 = R12 * t0 - R10 * t2;
		const float E12 = R10 * t1 - R11 * t0;
		const float E20 = R21 * t2 - R22 * t1;
		const float E21 = R22 * t0 - R20 * t2;
		const float E22 = R20 * t1 - R21 * t0;

		// ==========================
		// 直接计算 F21，零变量！
		// 不定义 fx, fy, cx, cy，不占任何内存！
		// ==========================
		_F21[0] = E00 * (1.0f / _cam[0]);
		_F21[1] = E10 * (1.0f / _cam[0]);
		_F21[2] = E20 * (1.0f / _cam[0]);

		_F21[3] = E01 * (1.0f / _cam[1]);
		_F21[4] = E11 * (1.0f / _cam[1]);
		_F21[5] = E21 * (1.0f / _cam[1]);

		_F21[6] = E00 * (-_cam[2] / _cam[0]) + E01 * (-_cam[3] / _cam[1]) + E02;
		_F21[7] = E10 * (-_cam[2] / _cam[0]) + E11 * (-_cam[3] / _cam[1]) + E12;
		_F21[8] = E20 * (-_cam[2] / _cam[0]) + E21 * (-_cam[3] / _cam[1]) + E22;
	}
	__device__ void getGridInd(float* _line, 
		int _wGrid2, int _hGrid2, 
		char* _gridInd, int& _gridIndSz)
	{
		_gridIndSz = 0;

		// ======================
		// 直线 L2 归一化（必须！）
		// ======================
		float a = _line[0];
		float b = _line[1];
		float c = _line[2];

		float norm = sqrtf(a * a + b * b + 1e-7f); // 防除0
		a /= norm;
		b /= norm;
		c /= norm;

		// ======================
		// 求直线与网格边界的两个交点
		// ======================
		float x0, y0, x1, y1;
		int n = 0;

		// 左 x=0
		if (fabsf(b) > 1e-9f) {
			float y = -c / b;
			if (y >= 0 && y < _hGrid2) { x0 = 0; y0 = y; n++; }
		}
		// 右 x=w-1
		if (n < 2 && fabsf(b) > 1e-9f) {
			float y = -(a * (_wGrid2 - 1) + c) / b;
			if (y >= 0 && y < _hGrid2) {
				if (n) { x1 = _wGrid2 - 1; y1 = y; }
				else { x0 = _wGrid2 - 1; y0 = y; n = 1; }
			}
		}
		// 下 y=0
		if (n < 2 && fabsf(a) > 1e-9f) {
			float x = -c / a;
			if (x >= 0 && x < _wGrid2) {
				if (n) { x1 = x; y1 = 0; }
				else { x0 = x; y0 = 0; n = 1; }
			}
		}
		// 上 y=h-1
		if (n < 2 && fabsf(a) > 1e-9f) {
			float x = -(b * (_hGrid2 - 1) + c) / a;
			if (x >= 0 && x < _wGrid2) {
				if (n) { x1 = x; y1 = _hGrid2 - 1; }
				else { x0 = x; y0 = _hGrid2 - 1; n = 1; }
			}
		}

		if (n < 2) return;

		// ======================
		// Bresenham 绘制（输出 行row，列col）
		// ======================
		int cx = int(x0);
		int cy = int(y0);
		int x2 = int(x1);
		int y2 = int(y1);

		int dx = abs(x2 - cx);
		int dy = -abs(y2 - cy);
		int sx = (cx < x2) ? 1 : -1;
		int sy = (cy < y2) ? 1 : -1;
		int err = dx + dy;

		while (true) {
			if (cx >= 0 && cx < _wGrid2 && cy >= 0 && cy < _hGrid2) {
				_gridInd[_gridIndSz++] = cy; // 行
				_gridInd[_gridIndSz++] = cx; // 列
			}
			if (cx == x2 && cy == y2) break;

			int e2 = err << 1;
			if (e2 >= dy) { err += dy; cx += sx; }
			if (e2 <= dx) { err += dx; cy += sy; }
		}
	}
	__global__ void matchFCU(int _kp1Sz, int _kp2Sz,
		float* _Twc1, float* _Twc2,
		float* _Tcw1, float* _Tcw2,
		int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz,
		float2* _kps1, float2* _kps2,
		unsigned int* _descs1, unsigned int* _descs2,
		float3* _Pcs1, float3* _Pcs2,
		float* _cam, unsigned int _cameraModel, int _imageWidth, int _imageHeight,
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
		short* _match2to1)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;
		_match2to1[idx] = -1;

		float3* p1 = _Pcs1 + idx;
		if (_bUse3D == 1)
		{
			if (p1[0].z <= 0)
				return;
		}

		//
		char rcGrids[CUDAORBMAXW / CUDAORBGRIDSOLU * 2 * 3];
		int gridIndSz = 0;
		if (_cameraModel == 0u)
		{
			float2* kp1 = _kps1 + idx;
			float F21[9];
			getF(_Tcw2, _Twc1, _cam, F21);
			float line[3];
			for (int i = 0; i < 3; ++i)
				line[i] = F21[i] * kp1[0].x + F21[3 + i] * kp1[0].y + F21[6 + i];
			getGridInd(line, _wGrid2, _hGrid2, rcGrids, gridIndSz);
		}

		unsigned int* desc1 = _descs1 + 8 * idx;
		int bestId = -1;
		int bestDescDist = 255;
		int nextBestId = -1;
		int nextBestDescDist = 255;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		transformTmp(_Twc1, p1[0], Pw1);
		float3 Pw2;
		short* kpIndSt;
		int kpIndSz;
		const int activeWidth = min(_wGrid2, (_imageWidth + _gridResul - 1) / _gridResul);
		const int activeHeight = min(_hGrid2, (_imageHeight + _gridResul - 1) / _gridResul);
		if (_cameraModel == 1u && (activeWidth <= 0 || activeHeight <= 0))
			return;
		const int cellCount = _cameraModel == 1u ? activeWidth * activeHeight : gridIndSz;
		for (int ii = 0; ii < cellCount; ++ii)
		{
			int i = _cameraModel == 1u ? ii / activeWidth : int(rcGrids[2 * ii]);
			int j = _cameraModel == 1u ? ii % activeWidth : int(rcGrids[2 * ii + 1]);
			if (i < 0 || j < 0 || i >= _hGrid2 || j >= _wGrid2)
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
		if (bestDescDist > nextBestDescDist * _nnTh)
			return;
		_match2to1[idx] = bestId;
	}
	void ORBMatcherCU::matchFCUDA(int _kp1Sz, int _kp2Sz, 
		Mat34CU& _Twc1, Mat34CU& _Twc2, 
		Mat34CU& _Tcw1, Mat34CU& _Tcw2, 
		int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz, 
		float2* _kps1, float2* _kps2,
		unsigned int* _descs1, unsigned int* _descs2, 
		float3* _Pcs1, float3* _Pcs2,
		CameraCU& _cam, 
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D, 
		short* _match2to1)
	{
		int threadNum = 1024;
		int gridSz = (CUDAORBKPSIZE + threadNum - 1) / threadNum;
		dim3 block(threadNum, 1);
		dim3 grid(gridSz, 1);
		matchFCU << <grid, block >> > (_kp1Sz, _kp2Sz,
			_Twc1.dataDev_, _Twc2.dataDev_,
			_Tcw1.dataDev_, _Tcw2.dataDev_,
			_wGrid2, _hGrid2, _gridResul, _featureGrid2, eveFeatureGrid2Sz,
			_kps1, _kps2,
			_descs1, _descs2,
			_Pcs1, _Pcs2,
			_cam.paramsDev_, _cam.cameraModel_, _cam.w_, _cam.h_,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
			_match2to1
			);
	}



	__global__ void matchProCU(int _kp1Sz, int _kp2Sz,
		float* _Twc1, float* _Twc2,
		float* _Tcw1, float* _Tcw2,
		int _wGrid2, int _hGrid2, int _gridResul, short* _featureGrid2, char* eveFeatureGrid2Sz,
		unsigned int* _descs1, unsigned int* _descs2,
		float3* _Pcs1, float3* _Pcs2,
		float3* _Pws1,
		char* _bPws1,
		float* _cam, unsigned int _cameraModel,
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
		short* _match2to1)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;
		_match2to1[idx] = -1;

		float3* p1 = _Pcs1 + idx;
		if (_bUse3D == 1)
		{
			if (p1[0].z <= 0)
				return;
		}

		if (_bPws1[idx] == 0)
			return;
		float3* Pw = _Pws1 + idx;
		int cGrid;
		int rGrid;
		if (getIndInGrid(_Tcw2, _wGrid2, _hGrid2, _gridResul, _featureGrid2, eveFeatureGrid2Sz,
			Pw[0], _cam, _cameraModel, cGrid, rGrid) == 0)
			return;

		unsigned int* desc1 = _descs1 + 8 * idx;
		int bestId = -1;
		int bestDescDist = 255;
		int nextBestId = -1;
		int nextBestDescDist = 255;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		transformTmp(_Twc1, p1[0], Pw1);
		float3 Pw2;
		short* kpIndSt;
		int kpIndSz;
		for (int i = rGrid-1; i <= rGrid + 1; ++i)
		{
			for (int j = cGrid-1; j <= cGrid+1; ++j)
			{
				if (i < 0 || j < 0 || i >= _hGrid2 || j >= _wGrid2)
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
			_cam.paramsDev_, _cam.cameraModel_,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
			_match2to1
			);
	}

	__device__ int getIndInGridCom(float* _Tcw,
		GridCU& _gridCom,
		float3& _Pw,
		float* _cam, unsigned int _cameraModel,
		int& _cGrid, int& _rGrid
	)
	{
		float3 Pc;
		transformTmp(_Tcw, _Pw, Pc);
		float u, v;
		if (Pc.z <= 0.0f)
			return 0;
		pointToImageTmp(_cam, _cameraModel, Pc, u, v);
		return _gridCom.getIndGrid(u, v, _cGrid, _rGrid);
	}
	__global__ void matchProCUCom(int _kp1Sz, int _kp2Sz,
		float* _Twc1, float* _Twc2,
		float* _Tcw1, float* _Tcw2,
		GridCU& _gridCom,
		unsigned int* _descs1, unsigned int* _descs2,
		float3* _Pcs1, float3* _Pcs2,
		float3* _Pws1,
		char* _bPws1,
		float* _cam, unsigned int _cameraModel,
		int _distThDesc, float _nnTh, char _bUse3D, float _squareDistTh3D,
		short* _match2to1)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= _kp1Sz)
			return;
		_match2to1[idx] = -1;

		float3* p1 = _Pcs1 + idx;
		if (_bUse3D == 1)
		{
			if (p1[0].z <= 0)
				return;
		}

		if (_bPws1[idx] == 0)
			return;
		float3* Pw = _Pws1 + idx;
		int cGrid;
		int rGrid;
		if (getIndInGridCom(_Tcw2, _gridCom, Pw[0], _cam, _cameraModel, cGrid, rGrid) == 0)
			return;

		unsigned int* desc1 = _descs1 + 8 * idx;
		int bestId = -1;
		int bestDescDist = 255;
		int nextBestId = -1;
		int nextBestDescDist = 255;
		int descDistTmp = 0;
		float squareDistTmp = 0;
		float3 Pw1;
		transformTmp(_Twc1, p1[0], Pw1);
		float3 Pw2;
		short* kpIndSt;
		int kpIndSz;
		for (int i = rGrid - 1; i <= rGrid + 1; ++i)
		{
			for (int j = cGrid - 1; j <= cGrid + 1; ++j)
			{
				if (_gridCom.isIndInGrid(j, i) == 0)
					continue;
				_gridCom.getKpIndInCell(j, i, kpIndSt, kpIndSz);
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
	void ORBMatcherCU::matchProCUDACom(int _kp1Sz, int _kp2Sz,
		Mat34CU& _Twc1, Mat34CU& _Twc2,
		Mat34CU& _Tcw1, Mat34CU& _Tcw2,
		GridCU& _gridCom,
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
		matchProCUCom << <grid, block >> > (_kp1Sz, _kp2Sz,
			_Twc1.dataDev_, _Twc2.dataDev_,
			_Tcw1.dataDev_, _Tcw2.dataDev_,
			_gridCom,
			_descs1, _descs2,
			_Pcs1, _Pcs2,
			_Pws1, _bPws1,
			_cam.paramsDev_, _cam.cameraModel_,
			_distThDesc, _nnTh, _bUse3D, _squareDistTh3D,
			_match2to1
			);
	}
}
