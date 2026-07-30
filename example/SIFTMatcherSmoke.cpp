#include <CUDAInclude.h>

#include <array>
#include <iostream>

int main()
{
	std::array<float, 3 * 128> descriptors1{};
	std::array<float, 3 * 128> descriptors2{};
	descriptors1[0] = 512.0f;
	descriptors1[128 + 1] = 512.0f;
	descriptors1[256 + 2] = 512.0f;
	descriptors2[0] = 1.0f;
	descriptors2[128 + 2] = 1.0f;
	descriptors2[256 + 1] = 1.0f;

	CUDA_LYJ::SIFTMatcherCache cache;
	cache.upload1(3, descriptors1.data());
	cache.upload2(3, descriptors2.data(), false);
	CUDA_LYJ::SIFTMatchHandle matcher = CUDA_LYJ::initSIFTMatcher();

	short forward[3] = { -1, -1, -1 };
	short reverse[3] = { -1, -1, -1 };
	CUDA_LYJ::matchBF(matcher, cache, forward, reverse);

	const short expectedForward[3] = { 0, 2, 1 };
	const short expectedReverse[3] = { 0, 2, 1 };
	for (int i = 0; i < 3; ++i)
	{
		if (forward[i] != expectedForward[i] || reverse[i] != expectedReverse[i])
		{
			std::cerr << "unexpected SIFT BF match at " << i << std::endl;
			CUDA_LYJ::releaseSIFTMatcher(matcher);
			return 1;
		}
	}

	CUDA_LYJ::releaseSIFTMatcher(matcher);
	std::cout << "descriptor-only CUDA SIFT matcher test passed" << std::endl;
	return 0;
}
