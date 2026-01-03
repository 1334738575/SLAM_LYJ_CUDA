#include <iostream>
#include "CUDAProjector.h"
#include <vector>
#include <fstream>
#include <random>
#include "CUDAInclude.h"
#include "CUDATexture.h"

namespace CUDA_LYJ
{
    CUDA_LYJ_API void testTexture()
    {
        // base
        int w = 160;
        int h = 120;
        std::vector<unsigned char> img(w * h * 4, 0);
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                int loc = (i * w + j) * 4;
                for (int k = 0; k < 4; ++k)
                {
                    img[loc + k] = 255;
                }
            }
        }
        std::vector<float> rets(w * h, 0);
        float *retsDev;

        // malloc
        // 纹理对象
        // texture<uchar4, cudaTextureType2D> texRef;
        cudaTextureObject_t texObj;
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        // 初始化资源描述符
        memset(&resDesc, 0, sizeof(resDesc));
        /*内存本质：Linear绑定普通线性全局内存，Array绑定专为 2D/3D 优化的 CUDA 数组，后者无法直接指针访问；
        性能与功能：Array支持 2D/3D 线性插值，纹理访问效率更高；Linear仅支持 1D 插值，但灵活性更好（可指针访问）；
        场景选择：2D/3D 图像处理优先用Array，一维数据或需混合访问优先用Linear。*/
        resDesc.resType = cudaResourceTypeArray;
        // 初始化纹理描述符
        memset(&texDesc, 0, sizeof(texDesc));
        /*cudaAddressModeClamp：超出边界时取边界值（默认）；
        cudaAddressModeWrap：循环重复（仅对归一化坐标有效）；
        cudaAddressModeMirror：镜像重复（仅对归一化坐标有效）；
        cudaAddressModeBorder：超出边界返回 0。*/
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        if (false)
        {
            /*cudaFilterModePoint：最近邻插值（默认）；
            cudaFilterModeLinear：仅支持浮点型数据（float/double）+ 归一化坐标（normalizedCoords=1）；
            2D/3D 线性插值仅对cudaResourceTypeArray生效。*/
            texDesc.filterMode = cudaFilterModeLinear; // 归一化
            texDesc.readMode = cudaReadModeNormalizedFloat;//数据归一化
            texDesc.normalizedCoords = true;//坐标归一化
        }
        else
        {
            texDesc.filterMode = cudaFilterModePoint;   // 点采样（整数坐标推荐）
            texDesc.readMode = cudaReadModeElementType; // 原始数据类型读取
            texDesc.normalizedCoords = false;           // 禁用归一化坐标
        }
        // cuda数组
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, w, h);
        cudaMemcpy2DToArray(cuArray, 0, 0, img.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
        // 数组绑定到纹理对象
        resDesc.res.array.array = cuArray;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        // 输出
        cudaMalloc((void **)&retsDev, w * h * sizeof(float));

        // process
        testTextureCUDA(retsDev, w, h, texObj);

        // output
        cudaMemcpy(rets.data(), retsDev, w * h * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < w * h; ++i)
        {
            // if (rets[i] != 4)
            //     printf("111\n");
        }

        // free
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);

        return;
    }

    CUDA_LYJ_API ProHandle initProjector(
        const float *Pws, const unsigned int PSize,
        const float *centers, const float *fNormals, const unsigned int *faces, const unsigned int fSize,
        float *camParams, const int w, const int h)
    {
        ProjectorCU *pro = new ProjectorCU();
        pro->create(Pws, PSize, centers, fNormals, faces, fSize, camParams, w, h);
        return (void *)pro;
    }
    //CUDA_LYJ_API void project(ProHandle handle,
    //                          float *Tcw,
    //                          float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
    //                          float minD, float maxD, float csTh, float detDTh)
    //{
    //    ProjectorCU *pro = (ProjectorCU *)handle;
    //    pro->project(Tcw, depths, fIds, allVisiblePIds, allVisibleFIds, minD, maxD, csTh, detDTh);
    //}
    CUDA_LYJ_API void project(ProHandle handle, ProjectorCache& cache,
                              float *Tcw,
                              float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
                              float minD, float maxD, float csTh, float detDTh)
    {
        ProjectorCU *pro = (ProjectorCU *)handle;
        pro->project(cache, Tcw, depths, fIds, allVisiblePIds, allVisibleFIds, minD, maxD, csTh, detDTh);
    }
    CUDA_LYJ_API void release(ProHandle handle)
    {
        ProjectorCU *pro = (ProjectorCU *)handle;
        pro->release();
        delete pro;
        return;
    }
}