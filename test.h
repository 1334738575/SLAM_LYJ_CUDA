#ifndef TEST_LYJ_H
#define TEST_LYJ_H

// export
#ifdef WIN32
#ifdef _MSC_VER
#define CUDA_LYJ_API __declspec(dllexport)
#else
#define CUDA_LYJ_API
#endif
#else
#define CUDA_LYJ_API
#endif



namespace SLAM_LYJ_CUDA
{
	CUDA_LYJ_API void test1(); //add
	CUDA_LYJ_API void test2(); //project
	CUDA_LYJ_API void test3(); //texture2d
}



#endif // !TEST_LYJ_H
