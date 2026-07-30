#include "CUDASIFTMatcher.h"

namespace CUDA_LYJ
{
	namespace
	{
		__device__ float3 transformSIFT(const float* T, const float3& p)
		{
			return make_float3(
				T[0] * p.x + T[3] * p.y + T[6] * p.z + T[9],
				T[1] * p.x + T[4] * p.y + T[7] * p.z + T[10],
				T[2] * p.x + T[5] * p.y + T[8] * p.z + T[11]);
		}

		__device__ float squareDistanceSIFT(const float3& a, const float3& b)
		{
			const float x = a.x - b.x;
			const float y = a.y - b.y;
			const float z = a.z - b.z;
			return x * x + y * y + z * z;
		}

		__device__ float descriptorDotSIFT(const float* a, const float* b)
		{
			float dot = 0.0f;
#pragma unroll 8
			for (int i = 0; i < CUDASIFTDESCSIZE; ++i)
				dot += a[i] * b[i];
			return dot;
		}

		__global__ void matchSIFTBFCU(int kp1Sz, int kp2Sz,
			const float* Twc1, const float* Twc2,
			const float* descs1, const float* descs2,
			const float3* Pcs1, const float3* Pcs2,
			float distMax, float ratioMax, char use3D, float squareDistTh3D,
			short* match2to1)
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= static_cast<unsigned int>(kp1Sz))
				return;

			match2to1[idx] = -1;
			float3 Pw1{};
			if (use3D == 1)
			{
				if (Pcs1[idx].z <= 0.0f)
					return;
				Pw1 = transformSIFT(Twc1, Pcs1[idx]);
			}

			const float* desc1 = descs1 + idx * CUDASIFTDESCSIZE;
			float bestDot = 0.0f;
			float nextBestDot = 0.0f;
			int bestId = -1;
			for (int i = 0; i < kp2Sz; ++i)
			{
				if (use3D == 1)
				{
					if (Pcs2[i].z <= 0.0f)
						continue;
					const float3 Pw2 = transformSIFT(Twc2, Pcs2[i]);
					if (squareDistanceSIFT(Pw1, Pw2) > squareDistTh3D)
						continue;
				}

				const float dot = descriptorDotSIFT(desc1, descs2 + i * CUDASIFTDESCSIZE);
				if (dot > bestDot)
				{
					nextBestDot = bestDot;
					bestDot = dot;
					bestId = i;
				}
				else if (dot > nextBestDot)
				{
					nextBestDot = dot;
				}
			}

			if (bestId < 0)
				return;
			const float bestDist = acosf(fminf(fmaxf(bestDot, -1.0f), 1.0f));
			const float nextBestDist = acosf(fminf(fmaxf(nextBestDot, -1.0f), 1.0f));
			if (bestDist < distMax && bestDist < nextBestDist * ratioMax)
				match2to1[idx] = static_cast<short>(bestId);
		}

		__global__ void keepMutualSIFTMatchesCU(int kp1Sz, int kp2Sz,
			short* match2to1, const short* reverseMatch)
		{
			const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= static_cast<unsigned int>(kp1Sz))
				return;
			const int match = static_cast<int>(match2to1[idx]);
			if (match < 0 || match >= kp2Sz || static_cast<int>(reverseMatch[match]) != static_cast<int>(idx))
				match2to1[idx] = -1;
		}
	}

	void SIFTMatcherCU::matchBF(SIFTMatcherCache& cache, float distMax, float ratioMax,
		char mutualBestMatch, char bUse3D, float squareDistTh3D)
	{
		if (!cache.hasDescs1_ || !cache.hasDescs2_ ||
			(bUse3D == 1 && (!cache.hasPcs1_ || !cache.hasPcs2_)))
		{
			cudaMemset(cache.match2to1Dev_, 0xff, cache.kpSz1_ * sizeof(short));
			cudaMemset(cache.reverseMatchDev_, 0xff, cache.kpSz2_ * sizeof(short));
			return;
		}
		matchBFCUDA(cache.kpSz1_, cache.kpSz2_, cache.Twc1Dev_, cache.Twc2Dev_,
			cache.descs1Dev_, cache.descs2Dev_, cache.Pcs1Dev_, cache.Pcs2Dev_,
			distMax, ratioMax, mutualBestMatch, bUse3D, squareDistTh3D,
			cache.match2to1Dev_, cache.reverseMatchDev_);
	}

	void SIFTMatcherCU::matchBFCUDA(int kp1Sz, int kp2Sz,
		Mat34CU& Twc1, Mat34CU& Twc2,
		const float* descs1, const float* descs2,
		const float3* Pcs1, const float3* Pcs2,
		float distMax, float ratioMax, char mutualBestMatch,
		char bUse3D, float squareDistTh3D,
		short* match2to1, short* reverseMatch)
	{
		if (kp1Sz > 0)
			cudaMemset(match2to1, 0xff, kp1Sz * sizeof(short));
		if (kp2Sz > 0)
			cudaMemset(reverseMatch, 0xff, kp2Sz * sizeof(short));
		if (kp1Sz <= 0 || kp2Sz <= 0 || distMax != distMax || distMax <= 0.0f ||
			ratioMax != ratioMax || ratioMax <= 0.0f ||
			(bUse3D == 1 && (squareDistTh3D != squareDistTh3D || squareDistTh3D < 0.0f)))
			return;

		constexpr int threadNum = 256;
		const dim3 block(threadNum, 1);
		const dim3 forwardGrid((kp1Sz + threadNum - 1) / threadNum, 1);
		matchSIFTBFCU<<<forwardGrid, block>>>(kp1Sz, kp2Sz, Twc1.dataDev_, Twc2.dataDev_,
			descs1, descs2, Pcs1, Pcs2, distMax, ratioMax, bUse3D, squareDistTh3D, match2to1);

		if (mutualBestMatch != 0)
		{
			const dim3 reverseGrid((kp2Sz + threadNum - 1) / threadNum, 1);
			matchSIFTBFCU<<<reverseGrid, block>>>(kp2Sz, kp1Sz, Twc2.dataDev_, Twc1.dataDev_,
				descs2, descs1, Pcs2, Pcs1, distMax, ratioMax, bUse3D, squareDistTh3D, reverseMatch);
			keepMutualSIFTMatchesCU<<<forwardGrid, block>>>(kp1Sz, kp2Sz, match2to1, reverseMatch);
		}
	}
}
