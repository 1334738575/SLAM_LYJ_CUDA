#include <CUDAInclude.h>

#include <algorithm>
#include <array>
#include <iostream>

int main()
{
	std::array<unsigned int, 24> descriptors1{};
	std::array<unsigned int, 24> descriptors2{};
	std::fill(descriptors1.begin() + 8, descriptors1.begin() + 16, 0xffffffffu);
	std::fill(descriptors1.begin() + 16, descriptors1.end(), 0xaaaaaaaau);
	std::fill(descriptors2.begin(), descriptors2.begin() + 8, 0xffffffffu);
	std::fill(descriptors2.begin() + 16, descriptors2.end(), 0xaaaaaaaau);

	CUDA_LYJ::ORBMatcherCache cache;
	cache.upload1(3, descriptors1.data());
	cache.upload2(3, descriptors2.data());
	CUDA_LYJ::MatchHanlde matcher = CUDA_LYJ::initMatcher();

	short forward[3] = { -1, -1, -1 };
	short reverse[3] = { -1, -1, -1 };
	CUDA_LYJ::matchBF(matcher, cache, forward, reverse);

	const short expected[3] = { 1, 0, 2 };
	for (int i = 0; i < 3; ++i)
	{
		if (forward[i] != expected[i] || reverse[i] != expected[i])
		{
			std::cerr << "unexpected ORB BF match at " << i << std::endl;
			CUDA_LYJ::releaseMatcher(matcher);
			return 1;
		}
	}

	CUDA_LYJ::releaseMatcher(matcher);
	std::cout << "descriptor-only CUDA ORB matcher test passed" << std::endl;
	return 0;
}
