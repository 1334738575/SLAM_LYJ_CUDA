#include <iostream>
#include "CUDAProjector.h"
#include <vector>
#include <fstream>
#include <random>
#include "CUDAInclude.h"

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
        // �������
        // texture<uchar4, cudaTextureType2D> texRef;
        cudaTextureObject_t texObj;
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        // ��ʼ����Դ������
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        // ��ʼ������������
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        if (false)
        {
            texDesc.filterMode = cudaFilterModeLinear; // ��һ��
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = true;
        }
        else
        {
            texDesc.filterMode = cudaFilterModePoint;   // ����������������Ƽ���
            texDesc.readMode = cudaReadModeElementType; // ԭʼ�������Ͷ�ȡ
            texDesc.normalizedCoords = false;           // ���ù�һ������
        }
        // cuda����
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, w, h);
        cudaMemcpy2DToArray(cuArray, 0, 0, img.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
        // ����󶨵��������
        resDesc.res.array.array = cuArray;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        // ���
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
    CUDA_LYJ_API void project(ProHandle handle,
                              float *Tcw,
                              float *depths, unsigned int *fIds, char *allVisiblePIds, char *allVisibleFIds,
                              float minD, float maxD, float csTh, float detDTh)
    {
        ProjectorCU *pro = (ProjectorCU *)handle;
        pro->project(Tcw, depths, fIds, allVisiblePIds, allVisibleFIds, minD, maxD, csTh, detDTh);
    }
    CUDA_LYJ_API void release(ProHandle handle)
    {
        ProjectorCU *pro = (ProjectorCU *)handle;
        pro->release();
        delete pro;
        return;
    }
}