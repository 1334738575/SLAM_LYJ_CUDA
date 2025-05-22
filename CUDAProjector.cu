#include "CUDAProjector.h"
#include "CUDACommon.h"
#include <cstdio>

namespace SLAM_LYJ_CUDA
{
	__global__ void testCU(int *_as, int *_bs, int *_cs, int _sz)
	{
		unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
		// int idy = threadIdx.y + blockDim.y * blockIdx.y;
		unsigned int id = idx;
		if (id >= _sz)
			return;
		_cs[id] = _as[id] + _bs[id];
	}
	void testCUDA(int *_as, int *_bs, int *_cs, int _sz)
	{
		dim3 block(128, 1);
		unsigned int gz = (_sz + 127) / 128;
		dim3 grid(gz, 1);
		testCU<<<grid, block>>>(_as, _bs, _cs, _sz);
	}

	__global__ void testTextureCU(float *_output, int _w, int _h, cudaTextureObject_t _texObj)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idx >= _w || idy >= _h)
			return;

		if (false)
		{
			float u = (idx + 0.5f) / _w;
			float v = (idy + 0.5f) / _h;
			float4 pixel = tex2D<float4>(_texObj, u, v);
			_output[idy * _w + idx] = pixel.x + pixel.y + pixel.z + pixel.w;
		}
		else
		{
			float u = idx; // 或 x + 0.5f（若需中心对齐）
			float v = idy;
			uchar4 pixel = tex2D<uchar4>(_texObj, u, v); // 数据类型与通道格式一致
			_output[idy * _w + idx] = pixel.x + pixel.y + pixel.z + pixel.w;
		}
	}
	void testTextureCUDA(float *_output, int _w, int _h, cudaTextureObject_t _texObj)
	{
		dim3 block(16, 16);
		dim3 grid((_w + block.x - 1) / block.x, (_h + block.y - 1) / block.y);
		testTextureCU<<<grid, block>>>(_output, _w, _h, _texObj);
	}

	__global__ void testTransformCU(Mat34CU _T, float3 *_ps, float3 *_rets, unsigned int _vn, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		for (unsigned int vi = idx; vi < idx + _step; ++vi)
		{
			if (vi >= _vn)
				break;
			_rets[vi] = _T * _ps[vi];
		}
	}
	void ProjectorCU::testTransformCUDA(Mat34CU _T, float3 *_ps, float3 *_rets, unsigned int _vn)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_vn + 1024 * 1024 - 1) / (1024 * 1024);
		testTransformCU<<<grid, block>>>(_T, _ps, _rets, _vn, step);
	}

	__global__ void testTransformNormalCU(Mat34CU _T, float3 *_normals, float3 *_rets, unsigned int _n, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _n)
			return;
		for (unsigned int i = idx; i < idx + _step; ++i)
		{
			if (i >= _n)
				break;
			_rets[i] = _T.transformNormal(_normals[i]);
		}
	}
	void ProjectorCU::testTransformNormalCUDA(Mat34CU _T, float3 *_normals, float3 *_rets, unsigned int _n)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_n + 1024 * 1024 - 1) / (1024 * 1024);
		testTransformNormalCU<<<grid, block>>>(_T, _normals, _rets, _n, step);
	}

	__global__ void testCameraCU(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, CameraCU _cam, unsigned int _step)
	{
		unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) * _step;
		if (idx >= _vn)
			return;
		for (unsigned int vi = idx; vi < idx + _step; ++vi)
		{
			if (vi >= _vn)
				break;
			// if (_p3ds[vi].z <= 0) {
			// _p2ds[vi].z = 0;
			// continue;
			// }
			_cam.pointToImage(_p3ds[vi], _p2ds[vi]);
			// if (_p2ds[vi].x < 0 || _p2ds[vi].x >= _w || _p2ds[vi].y < 0 || _p2ds[vi].y >= _h)
			// _p2ds[vi].z = 0;
			// printf("%f %f %f\n", _p3ds[vi].x, _p3ds[vi].y, _p3ds[vi].z);
			// printf("%f %f %f %f\n", _cam.paramsDev_[0], _cam.paramsDev_[1], _cam.paramsDev_[2], _cam.paramsDev_[3]);
			// printf("%f %f %f %f\n", _cam.paramsInvDev_[0], _cam.paramsInvDev_[1], _cam.paramsInvDev_[2], _cam.paramsInvDev_[3]);
			// printf("%f %f %f\n", _p2ds[vi].x, _p2ds[vi].y, _p2ds[vi].z);
			// printf("\n");
		}
	}
	void ProjectorCU::testCameraCUDA(float3 *_p3ds, float3 *_p2ds, unsigned int _vn, int _w, int _h, CameraCU _cam)
	{
		dim3 block(1024, 1);
		dim3 grid(1024, 1);
		unsigned int step = (_vn + 1024 * 1024 - 1) / (1024 * 1024);
		testCameraCU<<<grid, block>>>(_p3ds, _p2ds, _vn, _w, _h, _cam, step);
	}

	__global__ void testDepthAndFidCU(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, float *_depths, unsigned int *_fids, CameraCU _cam, BaseCU _base, unsigned int _step)
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
		int old, assume;
		float *pOld = (float *)(&old);
		float depthLast;
		int *add;
		int *pv;
		unsigned int oldFid;
		unsigned int *fidPtr;
		unsigned int tmp;
		for (unsigned int fi = idx; fi < idx + _step; ++fi)
		{
			if (fi >= _fn)
				break;
			if (_p2ds[_faces[fi].x].z == 0 && _p2ds[_faces[fi].y].z == 0 && _p2ds[_faces[fi].z].z == 0)
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
			else if (minv < _p2ds[_faces[fi].y].y)
				minv = _p2ds[_faces[fi].y].y;
			if (maxu < _p2ds[_faces[fi].z].x)
				maxu = _p2ds[_faces[fi].z].x;
			else if (minu > _p2ds[_faces[fi].z].x)
				minu = _p2ds[_faces[fi].z].x;
			if (maxv < _p2ds[_faces[fi].z].y)
				maxv = _p2ds[_faces[fi].z].y;
			else if (minv < _p2ds[_faces[fi].z].y)
				minv = _p2ds[_faces[fi].z].y;
			// printf("%f %f\n", _p2ds[_faces[fi].x].x, _p2ds[_faces[fi].x].y);
			// printf("%f %f\n", _p2ds[_faces[fi].y].x, _p2ds[_faces[fi].y].y);
			// printf("%f %f\n", _p2ds[_faces[fi].z].x, _p2ds[_faces[fi].z].y);

			d = -1 * _base.dot3(_fNormals[fi], _p3ds[_faces[fi].x]);
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
					loc = v * _w + u;

					AP.x = u - _p2ds[_faces[fi].x].x;
					AP.y = v - _p2ds[_faces[fi].x].y;
					BP.x = u - _p2ds[_faces[fi].y].x;
					BP.y = v - _p2ds[_faces[fi].y].y;
					CP.x = u - _p2ds[_faces[fi].z].x;
					CP.y = v - _p2ds[_faces[fi].z].y;
					if (_base.isP2dInTriangleCU(AB, BC, CA, AP, BP, CP) == (char)0)
						continue;

					_cam.imageToPoint(u, v, p);
					depth = -1 * d / _base.dot3(_fNormals[fi], p);
					depthLast = _depths[loc];
					if (depth >= depthLast)
						continue;

					//_depths[loc] = depth;
					//_fids[loc] = fi;

					add = (int *)_depths + loc;
					old = *(int *)(&depthLast);
					pv = (int *)&depth;
					for (int k = 0; k < 9999; ++k)
					{
						assume = old;
						old = atomicCAS(add, assume, *pv);
						if (old == assume)
						{
							oldFid = _fids[loc];
							fidPtr = _fids + loc;
							tmp = oldFid;
							for (int kk = 0; kk < 9999; ++kk)
							{
								if (depth > _depths[loc])
									break;
								oldFid = tmp;
								tmp = atomicCAS(fidPtr, oldFid, fi);
								if (tmp == oldFid)
									break;
							}

							//_fids[loc] = fi;

							break;
						}
						if (depth > *pOld)
							break;
					}
				}
			}
		}
	}
	void ProjectorCU::testDepthAndFidCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, float *_depths, unsigned int *_fids, CameraCU _cam, BaseCU _base)
	{
		int threadNum = 50;
		// cudaMemset(_depths, 0xff, _w * _h * sizeof(float)); //init outside
		cudaMemset(_fids, 0xff, _w * _h * sizeof(unsigned int));
		dim3 block(threadNum, 1);
		dim3 grid(threadNum, 1);
		unsigned int step = (_fn + threadNum * threadNum - 1) / (threadNum * threadNum);
		testDepthAndFidCU<<<grid, block>>>(_p3ds, _p2ds, _faces, _fNormals, _fn, _w, _h, _depths, _fids, _cam, _base, step);
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
	__global__ void testDepthAndFidCU(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, unsigned long long *_dIds, CameraCU _cam, BaseCU _base, unsigned int _step)
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
			if (_p2ds[_faces[fi].x].z <= 0 && _p2ds[_faces[fi].y].z <= 0 && _p2ds[_faces[fi].z].z <= 0)
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

			d = -1 * _base.dot3(_fNormals[fi], _p3ds[_faces[fi].x]);
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
					if (_base.isP2dInTriangleCU(AB, BC, CA, AP, BP, CP) == (char)0)
						continue;

					_cam.imageToPoint(u, v, p);
					depth = -1 * d / _base.dot3(_fNormals[fi], p);
					atomicUpdateDepthID(_dIds + loc, depth, fi);
				}
			}
		}
	}
	void ProjectorCU::testDepthAndFidCUDA(float3 *_p3ds, float3 *_p2ds, uint3 *_faces, float3 *_fNormals, unsigned int _fn, int _w, int _h, unsigned long long *_dIds, CameraCU _cam, BaseCU _base)
	{
		int threadNum = 50;
		dim3 block(threadNum, 1);
		dim3 grid(threadNum, 1);
		unsigned int step = (_fn + threadNum * threadNum - 1) / (threadNum * threadNum);
		testDepthAndFidCU<<<grid, block>>>(_p3ds, _p2ds, _faces, _fNormals, _fn, _w, _h, _dIds, _cam, _base, step);
	}

}