#include <CUDAInclude.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

void testTexture()
{
    cv::Mat m = cv::imread("D:/SLAM_LYJ/other/000.jpg");
    cv::imshow("imgIn", m);
    cv::waitKey();
}

int main(int argc, char *argv[])
{
    testTexture();
    //CUDA_LYJ::testTexture();


    return 0;
}