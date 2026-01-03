#include "CUDAProjector.h"
#include "CUDACommon.h"
#include <cstdio>

namespace CUDA_LYJ
{
	__device__ float dot3(const float3 &p1, const float3 &p2)
	{
		return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
	}
	__device__ float crossProduct2(const float2 &p1, const float2 &p2)
	{
		return p1.x * p2.y - p1.y * p2.x;
	}
	__device__ char isP2dInTriangleCU(const float2 &AB, const float2 &BC, const float2 &CA,
									  const float2 &AP, const float2 &BP, const float2 &CP)
	{
		float v1 = crossProduct2(AB, AP);
		float v2 = crossProduct2(BC, BP);
		float v3 = crossProduct2(CA, CP);
		if (v1 >= 0 && v2 >= 0 && v3 >= 0)
			return (char)1;
		if (v1 <= 0 && v2 <= 0 && v3 <= 0)
			return (char)1;
		return (char)0;
	}
	__device__ void imageToPoint(float *_camInv, const float &_u, const float &_v, float3 &_p3d)
	{
		_p3d.x = _u * _camInv[0] + _camInv[2];
		_p3d.y = _v * _camInv[1] + _camInv[3];
		_p3d.z = 1.0f;
	}

	__device__ float3 transform(float *_T, const float3 &_p)
	{
		float3 ret;
		ret.x = _T[0] * _p.x + _T[3] * _p.y + _T[6] * _p.z + _T[9];
		ret.y = _T[1] * _p.x + _T[4] * _p.y + _T[7] * _p.z + _T[10];
		ret.z = _T[2] * _p.x + _T[5] * _p.y + _T[8] * _p.z + _T[11];
		return ret;
	}
	__device__ void transform(float *_T, const float3 &_p, float3 &_ret)
	{
		_ret.x = _T[0] * _p.x + _T[3] * _p.y + _T[6] * _p.z + _T[9];
		_ret.y = _T[1] * _p.x + _T[4] * _p.y + _T[7] * _p.z + _T[10];
		_ret.z = _T[2] * _p.x + _T[5] * _p.y + _T[8] * _p.z + _T[11];
		return;
	}

	__device__ float3 transformNormal(float *_T, const float3 &_n)
	{
		float3 ret;
		ret.x = _T[0] * _n.x + _T[3] * _n.y + _T[6] * _n.z;
		ret.y = _T[1] * _n.x + _T[4] * _n.y + _T[7] * _n.z;
		ret.z = _T[2] * _n.x + _T[5] * _n.y + _T[8] * _n.z;
		return ret;
	}
	__device__ void transformNormal(float *_T, const float3 &_n, float3 &_ret)
	{
		_ret.x = _T[0] * _n.x + _T[3] * _n.y + _T[6] * _n.z;
		_ret.y = _T[1] * _n.x + _T[4] * _n.y + _T[7] * _n.z;
		_ret.z = _T[2] * _n.x + _T[5] * _n.y + _T[8] * _n.z;
		return;
	}

	__device__ void pointToImage(float *_cam, const float3 &_p3d, float3 &_p2d)
	{
		float invZ = 1.0f / _p3d.z;
		_p2d.x = _p3d.x * _cam[0] * invZ + _cam[2];
		_p2d.y = _p3d.y * _cam[1] * invZ + _cam[3];
		_p2d.z = _p3d.z;
	}


	__global__ void testTransformCU(float *_T, float3 *_ps, float3 *_rets, unsigned int _vn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		for (unsigned int vi = idx; vi < idx + _step; ++vi)
		{
			if (vi >= _vn)
				break;
			// _rets[vi] = _T * _ps[vi];
			transform(_T, _ps[vi], _rets[vi]);
		}
	}
	void ProjectorCU::testTransformCUDA(const Mat34CU &_T, float3 *_ps, float3 *_rets, unsigned int _vn)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_vn + 1024 * 1024 - 1) / (1024 * 1024);
		testTransformCU<<<grid, block>>>(_T.dataDev_, _ps, _rets, _vn, step);
	}

	__global__ void testTransformNormalCU(float *_T, float3 *_normals, float3 *_rets, unsigned int _n, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _n)
			return;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _n)
				break;
			// _rets[i] = _T.transformNormal(_normals[i]);
			transformNormal(_T, _normals[i], _rets[i]);
		}
	}
	void ProjectorCU::testTransformNormalCUDA(const Mat34CU &_T, float3 *_normals, float3 *_rets, unsigned int _n)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_n + 1024 * 1024 - 1) / (1024 * 1024);
		testTransformNormalCU<<<grid, block>>>(_T.dataDev_, _normals, _rets, _n, step);
	}

	__global__ void testCameraCU(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, float *_cam, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		for (unsigned int vi = idx; vi < idx + _step; ++vi)
		{
			if (vi >= _vn)
				break;
			if (_p3ds[vi].z == 0)
			{
				_p2ds[vi].z = 0;
				continue;
			}
			pointToImage(_cam, _p3ds[vi], _p2ds[vi]);
			// _p2ds[vi].z = 0;
			// printf("%f %f %f\n", _p3ds[vi].x, _p3ds[vi].y, _p3ds[vi].z);
			// printf("%f %f %f %f\n", _cam.paramsDev_[0], _cam.paramsDev_[1], _cam.paramsDev_[2], _cam.paramsDev_[3]);
			// printf("%f %f %f %f\n", _cam.paramsInvDev_[0], _cam.paramsInvDev_[1], _cam.paramsInvDev_[2], _cam.paramsInvDev_[3]);
			// printf("%f %f %f\n", _p2ds[vi].x, _p2ds[vi].y, _p2ds[vi].z);
			// printf("\n");
		}
	}
	void ProjectorCU::testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, const CameraCU &_cam)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_vn + 1024 * 1024 - 1) / (1024 * 1024);
		testCameraCU<<<grid, block>>>(_p3ds, _p2ds, _vn, _w, _h, _cam.paramsDev_, step);
	}

	////union 64 bit
	union DepthID
	{
		struct
		{
			float depth;
			unsigned int fid;
		};
		unsigned long long data;
	};
	__device__ void atomicUpdateDepthID(unsigned long long *addr, float new_depth, unsigned int new_fid)
	{
		union DepthID expected, desired;
		desired.depth = new_depth;
		desired.fid = new_fid;

		unsigned long long *addr_uint64 = (unsigned long long *)addr;
		unsigned long long expected_uint64 = *addr_uint64;
		do
		{
			expected.data = expected_uint64;
			if (desired.depth >= expected.depth)
				break;
			expected_uint64 = atomicCAS(addr_uint64, expected_uint64, desired.data);
		} while (expected_uint64 != expected.data);
	}
	__device__ bool checkDepth(const float &_minD, const float &_maxD, const float &_d)
	{
		if (_d > _minD && _d < _maxD)
			return true;
		return false;
	}
	__global__ void testDepthAndFidCU(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, char *_isFVisible, unsigned int _fn, int _w, int _h, float _minD, float _maxD, float _csTh, unsigned long long *_dIds, float *_camInv, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		// if (idx == 0)
		// return;
		if (idx >= _fn)
			return;
		int maxu, minu, maxv, minv;
		int bord = 1;
		float d;
		float2 AB, BC, CA, AP, BP, CP;
		int loc;
		float3 p;
		float depth;
		for (unsigned int fi = idx; fi < idx + _step; ++fi)
		{
			if (fi >= _fn)
				break;
			if (_fNormals[fi].z >= _csTh)
				continue;
			if (!checkDepth(_minD, _maxD, _p2ds[_faces[fi].x].z) && !checkDepth(_minD, _maxD, _p2ds[_faces[fi].y].z) && !checkDepth(_minD, _maxD, _p2ds[_faces[fi].z].z))
				continue;
			if (_p2ds[_faces[fi].x].z <= 0 || _p2ds[_faces[fi].y].z <= 0 || _p2ds[_faces[fi].z].z <= 0)
				continue;
			maxu = _p2ds[_faces[fi].x].x;
			minu = _p2ds[_faces[fi].x].x;
			maxv = _p2ds[_faces[fi].x].y;
			minv = _p2ds[_faces[fi].x].y;
			if (maxu < _p2ds[_faces[fi].y].x)
				maxu = _p2ds[_faces[fi].y].x;
			else if (minu > _p2ds[_faces[fi].y].x)
				minu = _p2ds[_faces[fi].y].x;
			if (maxv < _p2ds[_faces[fi].y].y)
				maxv = _p2ds[_faces[fi].y].y;
			else if (minv > _p2ds[_faces[fi].y].y)
				minv = _p2ds[_faces[fi].y].y;
			if (maxu < _p2ds[_faces[fi].z].x)
				maxu = _p2ds[_faces[fi].z].x;
			else if (minu > _p2ds[_faces[fi].z].x)
				minu = _p2ds[_faces[fi].z].x;
			if (maxv < _p2ds[_faces[fi].z].y)
				maxv = _p2ds[_faces[fi].z].y;
			else if (minv > _p2ds[_faces[fi].z].y)
				minv = _p2ds[_faces[fi].z].y;
			// printf("%f %f\n", _p2ds[_faces[fi].x].x, _p2ds[_faces[fi].x].y);
			// printf("%f %f\n", _p2ds[_faces[fi].y].x, _p2ds[_faces[fi].y].y);
			// printf("%f %f\n", _p2ds[_faces[fi].z].x, _p2ds[_faces[fi].z].y);

			if (minu >= _w || maxu < 0 || minv >= _h || maxv < 0)
				continue;

			_isFVisible[fi] = 1;
			d = -1 * dot3(_fNormals[fi], _p3ds[_faces[fi].x]); // TODO
			// printf("%f %f %f %f\n", _fNormals[fi].x, _fNormals[fi].y, _fNormals[fi].z, d);

			AB.x = _p2ds[_faces[fi].y].x - _p2ds[_faces[fi].x].x;
			AB.y = _p2ds[_faces[fi].y].y - _p2ds[_faces[fi].x].y;
			BC.x = _p2ds[_faces[fi].z].x - _p2ds[_faces[fi].y].x;
			BC.y = _p2ds[_faces[fi].z].y - _p2ds[_faces[fi].y].y;
			CA.x = _p2ds[_faces[fi].x].x - _p2ds[_faces[fi].z].x;
			CA.y = _p2ds[_faces[fi].x].y - _p2ds[_faces[fi].z].y;

			minu = minu - bord > 0 ? minu - bord : 0;
			maxu = maxu + bord < _w ? maxu + bord : _w - 1;
			minv = minv - bord > 0 ? minv - bord : 0;
			maxv = maxv + bord < _h ? maxv + bord : _h - 1;
			// printf("%u %d %d %d %d\n", fi, minu, maxu, minv, maxv);

			for (int v = minv; v <= maxv; ++v)
			{
				for (int u = minu; u <= maxu; ++u)
				{
					if (u >= _w || u < 0 || v >= _h || v < 0)
						continue;
					loc = v * _w + u;

					AP.x = u - _p2ds[_faces[fi].x].x;
					AP.y = v - _p2ds[_faces[fi].x].y;
					BP.x = u - _p2ds[_faces[fi].y].x;
					BP.y = v - _p2ds[_faces[fi].y].y;
					CP.x = u - _p2ds[_faces[fi].z].x;
					CP.y = v - _p2ds[_faces[fi].z].y;
					if (isP2dInTriangleCU(AB, BC, CA, AP, BP, CP) == (char)0) // TODO
						continue;

					imageToPoint(_camInv, u, v, p); // TODO
					if (dot3(_fNormals[fi], p) == 0)
						continue;
					depth = -1 * d / dot3(_fNormals[fi], p);
					if (!checkDepth(_minD, _maxD, depth))
						continue;
					atomicUpdateDepthID(_dIds + loc, depth, fi);
				}
			}
		}
	}

	__global__ void dIds2Depth(unsigned long long *_dIds, float *_depth, unsigned int _w, unsigned int _h, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _w * _h)
			return;
		DepthID didTmp;
		for (int ind = idx; ind < idx + _step; ++ind)
		{
			if (ind >= _w * _h)
				return;
			didTmp.data = _dIds[ind];
			_depth[ind] = didTmp.depth;
		}
	}
	__global__ void testPoint2UVZ(float3 *_p2ds, char *_isVisible, float *_depth, unsigned int _w, unsigned int _h, float _detDTh, unsigned int _vn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		int u, v;
		float ddd;
		for (int ind = idx; ind < idx + _step; ++ind)
		{
			if (ind >= _vn)
				continue;
			u = (int)_p2ds[ind].x;
			v = (int)_p2ds[ind].y;
			if (u >= _w || u < 0 || v >= _h || v < 0)
				continue;
			float &z = _p2ds[ind].z;
			if (z <= 0)
				continue;
			ddd = _depth[v * _w + u];
			if (ddd != FLT_MAX && z <= (ddd + _detDTh) && z >= (ddd - _detDTh))
			{
				_isVisible[ind] = 1;
				continue;
			}
			for (int i = v - 1; i <= v + 1; ++i)
			{
				if (i < 0 || i >= _h)
					continue;
				for (int j = u - 1; j <= u + 1; ++j)
				{
					if (j < 0 || j >= _w)
						continue;
					float &zPre = _depth[i * _w + j];
					if (zPre == FLT_MAX)
						continue;
					if (ddd > zPre)
						ddd = zPre;
				}
			}
			if (ddd == FLT_MAX)
				continue;
			if (z > (ddd + _detDTh) || z < (ddd - _detDTh))
				continue;
			_isVisible[ind] = 1;
		}

		// for (int ind = idx; ind < idx + _step; ++ind)
		// {
		// 	if (ind >= _vn)
		// 		return;
		// 	u = (int)_p2ds[ind].x;
		// 	v = (int)_p2ds[ind].y;
		// 	if (u >= _w || u < 0 || v >= _h || v < 0)
		// 		continue;
		// 	float &z = _p2ds[ind].z;
		// 	if (z <= 0)
		// 		continue;
		// 	float &zPre = _depth[v * _w + u];
		// 	if (zPre == FLT_MAX) // TODO
		// 		continue;
		// 	if (z > (zPre + _detDTh))
		// 		continue;
		// 	_isVisible[ind] = 1;
		// }
	}

	__global__ void testCenter2UVZ(char *_isPVisible, float3 *_ctr2ds, uint3 *_faces, char *_isVisible, float *_depth, unsigned int _w, unsigned int _h, unsigned int _fn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _fn)
			return;
		//int u, v;
		for (int ind = idx; ind < idx + _step; ++ind)
		{
			if (ind >= _fn)
				return;
			if (_isVisible[ind] == 0)
				continue;
			// u = (int)_ctr2ds[ind].x;
			// v = (int)_ctr2ds[ind].y;
			// if (u >= _w || u < 0 || v >= _h || v < 0)
			// 	continue;
			// float &z = _ctr2ds[ind].z;
			// if (z <= 0)
			// 	continue;
			// float &zPre = _depth[v * _w + u];
			// if (zPre == FLT_MAX)
			// 	continue;
			// if (z > (zPre + 1))
			// 	continue;
			uint3 &face = _faces[ind];
			if (_isPVisible[face.x] == 1 || _isPVisible[face.y] == 1 || _isPVisible[face.z] == 1)
				_isVisible[ind] = 1;
			else
				_isVisible[ind] = 0;
		}
	}

	void ProjectorCU::testDepthAndFidAndCheckCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals,
												  unsigned int _vn, unsigned int _fn, int _w, int _h, float3 *_ctr2ds, float _minD, float _maxD, float _csTh, float _detDTh,
												  float *_depths, unsigned long long *_dIds, char *_isPVisible, char *_isFVisible,
												  const CameraCU &_cam)
	{
		int threadNum = 1024;
		dim3 block(threadNum, 1);
		dim3 grid(threadNum, 1);
		unsigned int stepF = (_fn + threadNum * threadNum - 1) / (threadNum * threadNum);
		cudaMemset(_isFVisible, 0, _fn * sizeof(char));
		testDepthAndFidCU<<<grid, block>>>(_p3ds, _p2ds, _faces, _fNormals, _isFVisible, _fn, _w, _h, _minD, _maxD, _csTh, _dIds, _cam.paramsInvDev_, stepF);

		unsigned int stepI = (_w * _h + threadNum * threadNum - 1) / (threadNum * threadNum);
		dIds2Depth<<<grid, block>>>(_dIds, _depths, _w, _h, stepI);

		unsigned int stepV = (_vn + threadNum * threadNum - 1) / (threadNum * threadNum);
		cudaMemset(_isPVisible, 0, _vn * sizeof(char));
		testPoint2UVZ<<<grid, block>>>(_p2ds, _isPVisible, _depths, _w, _h, _detDTh, _vn, stepV);

		testCenter2UVZ<<<grid, block>>>(_isPVisible, _ctr2ds, _faces, _isFVisible, _depths, _w, _h, _fn, stepF);
	}
}