#include "CUDAProjector.h"
#include "CUDACommon.cuh"
#include <cstdio>

namespace CUDA_LYJ
{

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
	__device__ float transformNormalZ(float *_T, const float3 &_n)
	{
		return _T[2] * _n.x + _T[5] * _n.y + _T[8] * _n.z;
	}

	__device__ void pointToImage(float *_cam, const float3 &_p3d, float3 &_p2d)
	{
		float invZ = 1.0f / _p3d.z;
		_p2d.x = _p3d.x * _cam[0] * invZ + _cam[2];
		_p2d.y = _p3d.y * _cam[1] * invZ + _cam[3];
		_p2d.z = _p3d.z;
	}

	__device__ void transformAndProjectPoint(float *_T, float3 *_pws, float3 *_p2ds, unsigned int _ind, float *_cam)
	{
		float3 pc = transform(_T, _pws[_ind]);
		if (pc.z == 0)
		{
			_p2ds[_ind].z = 0;
			return;
		}
		pointToImage(_cam, pc, _p2ds[_ind]);
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

	__global__ void testTransformAndCameraCU(float *_T, float3 *_pws, float3 *_p2ds, unsigned int _vn, float *_cam, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		for (unsigned int vi = idx; vi < idx + _step; ++vi)
		{
			if (vi >= _vn)
				break;
			transformAndProjectPoint(_T, _pws, _p2ds, vi, _cam);
		}
	}
	__global__ void testSelectedTransformAndCameraCU(float *_T, float3 *_pws, float3 *_p2ds, unsigned int _vn, float *_cam, unsigned int *_ids, unsigned int _projectVn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _projectVn)
			return;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _projectVn)
				break;
			unsigned int vi = _ids[i];
			if (vi >= _vn)
				continue;
			transformAndProjectPoint(_T, _pws, _p2ds, vi, _cam);
		}
	}
	void ProjectorCU::testTransformAndCameraCUDA(const Mat34CU &_T, float3 *_pws, float3 *_p2ds, unsigned int _vn, const CameraCU &_cam, unsigned int *_ids, unsigned int _projectVn, bool _useIds)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		if (!_useIds)
		{
			_projectVn = _vn;
			_ids = nullptr;
		}
		unsigned int step = (_projectVn + 1024 * 1024 - 1) / (1024 * 1024);
		if (_useIds)
			testSelectedTransformAndCameraCU<<<grid, block>>>(_T.dataDev_, _pws, _p2ds, _vn, _cam.paramsDev_, _ids, _projectVn, step);
		else
			testTransformAndCameraCU<<<grid, block>>>(_T.dataDev_, _pws, _p2ds, _vn, _cam.paramsDev_, step);
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
	__global__ void testDepthAndFidCU(float *_T, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, char *_isFVisible, unsigned int _fn, int _w, int _h, float _minD, float _maxD, unsigned long long *_dIds, float *_camInv, unsigned int *_faceIds, unsigned int _projectFn, char *_pointMask, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		// if (idx == 0)
		// return;
		if (idx >= _projectFn)
			return;
		int maxu, minu, maxv, minv;
		int bord = 1;
		float d;
		float2 AB, BC, CA, AP, BP, CP;
		int loc;
		float3 p;
		float3 fNormal;
		float normalDotRay;
		float depth;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _projectFn)
				break;
			unsigned int fi = _faceIds == nullptr ? i : _faceIds[i];
			if (fi >= _fn)
				continue;
			if (_pointMask != nullptr && (_pointMask[_faces[fi].x] == 0 || _pointMask[_faces[fi].y] == 0 || _pointMask[_faces[fi].z] == 0))
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
			fNormal = transformNormal(_T, _fNormals[fi]);
			imageToPoint(_camInv, _p2ds[_faces[fi].x].x, _p2ds[_faces[fi].x].y, p);
			p.x *= _p2ds[_faces[fi].x].z;
			p.y *= _p2ds[_faces[fi].x].z;
			p.z = _p2ds[_faces[fi].x].z;
			d = -1 * dot3(fNormal, p); // TODO
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
					normalDotRay = dot3(fNormal, p);
					if (normalDotRay == 0)
						continue;
					depth = -1 * d / normalDotRay;
					if (!checkDepth(_minD, _maxD, depth))
						continue;
					atomicUpdateDepthID(_dIds + loc, depth, fi);
				}
			}
		}
	}

	__global__ void dIds2Depth(float *_T, unsigned long long *_dIds, float *_depth, float3 *_fNormals, unsigned int _fn, float _csTh, unsigned int _w, unsigned int _h, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _w * _h)
			return;
		DepthID didTmp;
		DepthID reset;
		reset.depth = FLT_MAX;
		reset.fid = UINT_MAX;
		for (int ind = idx; ind < idx + _step; ++ind)
		{
			if (ind >= _w * _h)
				return;
			didTmp.data = _dIds[ind];
			bool invalidNormal = didTmp.fid != UINT_MAX && didTmp.fid >= _fn;
			if (didTmp.fid != UINT_MAX && !invalidNormal)
				invalidNormal = transformNormalZ(_T, _fNormals[didTmp.fid]) >= _csTh;
			if (invalidNormal)
			{
				didTmp = reset;
				_dIds[ind] = reset.data;
			}
			_depth[ind] = didTmp.depth;
		}
	}

	__device__ char checkPoint2UVZ(float3 *_p2ds, unsigned int _ind, float *_depth, unsigned int _w, unsigned int _h, float _detDTh)
	{
		int u = (int)_p2ds[_ind].x;
		int v = (int)_p2ds[_ind].y;
		if (u >= _w || u < 0 || v >= _h || v < 0)
			return 0;
		float &z = _p2ds[_ind].z;
		if (z <= 0)
			return 0;
		float ddd = _depth[v * _w + u];
		if (ddd != FLT_MAX && z <= (ddd + _detDTh) && z >= (ddd - _detDTh))
			return 1;
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
			return 0;
		if (z > (ddd + _detDTh) || z < (ddd - _detDTh))
			return 0;
		return 1;
	}

	__global__ void testSelectedPoint2UVZ(float3 *_p2ds, uint3 *_faces, unsigned int *_faceIds, char *_isVisible, float *_depth, unsigned int _w, unsigned int _h, float _detDTh, unsigned int _fn, unsigned int _projectFn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _projectFn)
			return;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _projectFn)
				break;
			unsigned int fi = _faceIds == nullptr ? i : _faceIds[i];
			if (fi >= _fn)
				continue;
			uint3 &face = _faces[fi];
			if (_isVisible[face.x] == 0 && checkPoint2UVZ(_p2ds, face.x, _depth, _w, _h, _detDTh))
				_isVisible[face.x] = 1;
			if (_isVisible[face.y] == 0 && checkPoint2UVZ(_p2ds, face.y, _depth, _w, _h, _detDTh))
				_isVisible[face.y] = 1;
			if (_isVisible[face.z] == 0 && checkPoint2UVZ(_p2ds, face.z, _depth, _w, _h, _detDTh))
				_isVisible[face.z] = 1;
		}
	}

	__global__ void testSelectedPointIds2UVZ(float3 *_p2ds, unsigned int *_pointIds, char *_isVisible, float *_depth, unsigned int _w, unsigned int _h, float _detDTh, unsigned int _vn, unsigned int _projectVn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _projectVn)
			return;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _projectVn)
				break;
			unsigned int ind = _pointIds[i];
			if (ind >= _vn)
				continue;
			if (_isVisible[ind] == 0 && checkPoint2UVZ(_p2ds, ind, _depth, _w, _h, _detDTh))
				_isVisible[ind] = 1;
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

	__global__ void testCenter2UVZ(char *_isPVisible, float3 *_ctr2ds, uint3 *_faces, char *_isVisible, float *_depth, unsigned int _w, unsigned int _h, unsigned int _fn, unsigned int *_faceIds, unsigned int _projectFn, char *_pointMask, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _projectFn)
			return;
		//int u, v;
		for (int i = idx; i < idx + _step; ++i)
		{
			if (i >= _projectFn)
				return;
			unsigned int ind = _faceIds == nullptr ? i : _faceIds[i];
			if (ind >= _fn)
				continue;
			uint3 &face = _faces[ind];
			if (_pointMask != nullptr && (_pointMask[face.x] == 0 || _pointMask[face.y] == 0 || _pointMask[face.z] == 0))
				continue;
			if (_isVisible[ind] == 0 && _isPVisible[face.x] == 0 && _isPVisible[face.y] == 0 && _isPVisible[face.z] == 0)
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
			if (_isPVisible[face.x] == 1 || _isPVisible[face.y] == 1 || _isPVisible[face.z] == 1)
				_isVisible[ind] = 1;
			else
				_isVisible[ind] = 0;
		}
	}

	void ProjectorCU::testDepthAndFidAndCheckCUDA(const Mat34CU &_T, float3 *_p2ds, uint3 *_faces, float3 *_fNormals,
												  unsigned int _vn, unsigned int _fn, int _w, int _h, float3 *_ctr2ds, float _minD, float _maxD, float _csTh, float _detDTh,
												  float *_depths, unsigned long long *_dIds, char *_isPVisible, char *_isFVisible,
												  const CameraCU &_cam, unsigned int *_faceIds, unsigned int _projectFn, bool _useFaceIds,
												  unsigned int *_pointIds, unsigned int _projectVn, bool _usePointIds, char *_pointMask)
	{
		int threadNum = 1024;
		dim3 block(threadNum, 1);
		dim3 grid(threadNum, 1);
		if (!_useFaceIds)
		{
			_faceIds = nullptr;
			_projectFn = _fn;
		}
		if (!_usePointIds)
		{
			_pointIds = nullptr;
			_projectVn = _vn;
			_pointMask = nullptr;
		}
		unsigned int stepF = (_projectFn + threadNum * threadNum - 1) / (threadNum * threadNum);
		cudaMemset(_isFVisible, 0, _fn * sizeof(char));
		testDepthAndFidCU<<<grid, block>>>(_T.dataDev_, _p2ds, _faces, _fNormals, _isFVisible, _fn, _w, _h, _minD, _maxD, _dIds, _cam.paramsInvDev_, _faceIds, _projectFn, _pointMask, stepF);

		unsigned int stepI = (_w * _h + threadNum * threadNum - 1) / (threadNum * threadNum);
		dIds2Depth<<<grid, block>>>(_T.dataDev_, _dIds, _depths, _fNormals, _fn, _csTh, _w, _h, stepI);

		unsigned int stepV = (_vn + threadNum * threadNum - 1) / (threadNum * threadNum);
		unsigned int stepSelectedV = (_projectVn + threadNum * threadNum - 1) / (threadNum * threadNum);
		cudaMemset(_isPVisible, 0, _vn * sizeof(char));
		if (_usePointIds)
			testSelectedPointIds2UVZ<<<grid, block>>>(_p2ds, _pointIds, _isPVisible, _depths, _w, _h, _detDTh, _vn, _projectVn, stepSelectedV);
		else if (_useFaceIds)
			testSelectedPoint2UVZ<<<grid, block>>>(_p2ds, _faces, _faceIds, _isPVisible, _depths, _w, _h, _detDTh, _fn, _projectFn, stepF);
		else
			testPoint2UVZ<<<grid, block>>>(_p2ds, _isPVisible, _depths, _w, _h, _detDTh, _vn, stepV);

		testCenter2UVZ<<<grid, block>>>(_isPVisible, _ctr2ds, _faces, _isFVisible, _depths, _w, _h, _fn, _faceIds, _projectFn, _pointMask, stepF);
	}
}
