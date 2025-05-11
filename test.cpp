#include <iostream>
#include "CUDAProjector.h"
#include <vector>
#include <fstream>
#include <random>
#include "test.h"

namespace SLAM_LYJ_CUDA {
    CUDA_LYJ_API void test1()
    {
        int sz = 100;
        int szBit = sz * sizeof(int);
        std::vector<int> as(100, 0);
        std::vector<int> bs(100, 1);
        std::vector<int> cs(100, 0);
        int* asDev;
        int* bsDev;
        int* csDev;
        cudaMalloc((void**)&asDev, szBit);
        cudaMalloc((void**)&bsDev, szBit);
        cudaMalloc((void**)&csDev, szBit);
        cudaMemcpy(asDev, as.data(), szBit, cudaMemcpyHostToDevice);
        cudaMemcpy(bsDev, bs.data(), szBit, cudaMemcpyHostToDevice);

        SLAM_LYJ_CUDA::testCUDA(asDev, bsDev, csDev, sz);

        std::cout << "before:" << std::endl;
        for (int i = 0; i < sz; ++i)
            std::cout << cs[i] << std::endl;

        cudaMemcpy(cs.data(), csDev, szBit, cudaMemcpyDeviceToHost);
        cudaFree(asDev);
        cudaFree(bsDev);
        cudaFree(csDev);

        std::cout << "after:" << std::endl;
        for (int i = 0; i < sz; ++i)
            std::cout << cs[i] << std::endl;
    }

    union DepthID2
    {
        struct
        {
            float depth;
            unsigned int fid;
        };
        uint64_t data;
    };
    CUDA_LYJ_API void test2()
    {
        // generate data
        int fn = 1000;
        int vn = fn * 3;
        std::vector<float> Pws(vn * 3);
        std::vector<float> Pcs(vn * 3);
        std::vector<unsigned int> faces(fn * 3);
        std::vector<float> fNormalws(fn * 3);
        std::vector<float> fNormalcs(fn * 3);
        int h = 120;
        int w = 160;
        std::vector<float> cam(4);
        std::vector<float> camInv(4);
        cam[0] = 80;
        cam[1] = 60;
        cam[2] = 80;
        cam[3] = 60;
        camInv[0] = 1.0f / cam[0];
        camInv[1] = 1.0f / cam[1];
        camInv[2] = -1 * cam[2] / cam[0];
        camInv[3] = -1 * cam[3] / cam[1];
        std::vector<float> pixels(vn * 3);
        std::vector<float> depths(w * h, FLT_MAX);
        std::vector<unsigned int> fids(w * h, UINT_MAX);
        std::vector<int> uvs1 = { 3, 3, 3, 100, 100, 3 };
        std::vector<int> uvs2 = { 4, 4, 4, 97, 97, 4 };
        for (int i = 0; i < fn; ++i)
        {
            int ind = 3 * i;
            faces[ind] = ind;
            faces[ind + 1] = ind + 1;
            faces[ind + 2] = ind + 2;
            int ind2 = 3 * ind;
            int d = (i + 1) * 2;
            std::vector<int> uvs = uvs2;
            if (i == 0)
                uvs = uvs1;
            Pws[ind2] = (camInv[0] * uvs[0] + camInv[2]) * d;
            Pws[ind2 + 1] = (camInv[1] * uvs[1] + camInv[3]) * d;
            Pws[ind2 + 2] = d;
            Pws[ind2 + 3] = (camInv[0] * uvs[2] + camInv[2]) * d;
            Pws[ind2 + 4] = (camInv[1] * uvs[3] + camInv[3]) * d;
            Pws[ind2 + 5] = d;
            Pws[ind2 + 6] = (camInv[0] * uvs[4] + camInv[2]) * d;
            Pws[ind2 + 7] = (camInv[1] * uvs[5] + camInv[3]) * d;
            Pws[ind2 + 8] = d;
        }
        auto funcCrossProduct3 = [](const float* p1, const float* p2, float* p3)
            {
                p3[0] = p1[1] * p2[2] - p1[2] * p2[1];
                p3[1] = p1[2] * p2[0] - p1[0] * p2[2];
                p3[2] = p1[0] * p2[1] - p1[1] * p2[0];
                float n = std::sqrt(p3[0] * p3[0] + p3[1] * p3[1] + p3[2] * p3[2]);
                p3[0] /= n;
                p3[1] /= n;
                p3[2] /= n;
                // printf("%f %f %f\n", p3[0], p3[1], p3[2]);
            };
        std::vector<float> AB(3);
        std::vector<float> AC(3);
        for (int i = 0; i < fn; ++i)
        {
            float* A = Pws.data() + 3 * faces[3 * i];
            float* B = Pws.data() + 3 * faces[3 * i + 1];
            float* C = Pws.data() + 3 * faces[3 * i + 2];
            AB[0] = B[0] - A[0];
            AB[1] = B[1] - A[1];
            AB[2] = B[2] - A[2];
            AC[0] = C[0] - A[0];
            AC[1] = C[1] - A[1];
            AC[2] = C[2] - A[2];
            funcCrossProduct3(AB.data(), AC.data(), fNormalws.data() + 3 * i);
        }
        std::vector<float> Tcw(12, 0);
        Tcw[0] = 1;
        Tcw[4] = 1;
        Tcw[8] = 1;
        // Tcw[9] = 1;
        // Tcw[10] = 5;
        // Tcw[11] = 10;
        std::ofstream fP("Pws.txt");
        std::ofstream fcam("cam.txt");
        for (int i = 0; i < vn; ++i)
        {
            fP << Pws[3 * i] << " " << Pws[3 * i + 1] << " " << Pws[3 * i + 2] << std::endl;
        }
        int maxd = 2000;
        fcam << 0 << " " << 0 << " " << 0 << std::endl;
        fcam << (camInv[0] * 0 + camInv[2]) * maxd << " " << (camInv[1] * 0 + camInv[3]) * maxd << " " << maxd << std::endl;
        fcam << (camInv[0] * w + camInv[2]) * maxd << " " << (camInv[1] * 0 + camInv[3]) * maxd << " " << maxd << std::endl;
        fcam << (camInv[0] * 0 + camInv[2]) * maxd << " " << (camInv[1] * h + camInv[3]) * maxd << " " << maxd << std::endl;
        fcam << (camInv[0] * w + camInv[2]) * maxd << " " << (camInv[1] * h + camInv[3]) * maxd << " " << maxd << std::endl;
        fP.close();
        fcam.close();
        std::vector<uint64_t> dids(w * h);
        std::vector<DepthID2> did2s(w * h);
        DepthID2 tmp;
        tmp.depth = FLT_MAX;
        tmp.fid = UINT_MAX;
        for (auto& did : did2s)
            did = tmp;
        memcpy(dids.data(), did2s.data(), w * h * sizeof(uint64_t));

        // upload
        float3* PwsDev;
        cudaMalloc((void**)&PwsDev, vn * 3 * sizeof(float));
        cudaMemcpy(PwsDev, Pws.data(), vn * 3 * sizeof(float), cudaMemcpyHostToDevice);
        float3* PcsDev;
        cudaMalloc((void**)&PcsDev, vn * 3 * sizeof(float));
        SLAM_LYJ_CUDA::Mat34CU TDev;
        TDev.upload(Tcw.data());
        uint3* facesDev;
        cudaMalloc((void**)&facesDev, fn * sizeof(uint3));
        cudaMemcpy(facesDev, faces.data(), fn * sizeof(uint3), cudaMemcpyHostToDevice);
        float3* fNormalwsDev;
        cudaMalloc((void**)&fNormalwsDev, fn * sizeof(float3));
        cudaMemcpy(fNormalwsDev, fNormalws.data(), fn * sizeof(float3), cudaMemcpyHostToDevice);
        float3* fNormalcsDev;
        cudaMalloc((void**)&fNormalcsDev, fn * sizeof(float3));
        float3* pixelsDev;
        cudaMalloc((void**)&pixelsDev, vn * sizeof(float3));
        SLAM_LYJ_CUDA::CameraCU camDev;
        camDev.upload(w, h, cam.data(), camInv.data());
        float* depthsDev;
        cudaMalloc((void**)&depthsDev, w * h * sizeof(float));
        cudaMemcpy(depthsDev, depths.data(), w * h * sizeof(float), cudaMemcpyHostToDevice);
        unsigned int* fidsDev;
        cudaMalloc((void**)&fidsDev, w * h * sizeof(unsigned int));
        cudaMemcpy(fidsDev, fids.data(), w * h * sizeof(unsigned int), cudaMemcpyHostToDevice);
        uint64_t* didsDev;
        cudaMalloc((void**)&didsDev, w * h * sizeof(uint64_t));
        cudaMemcpy(didsDev, dids.data(), w * h * sizeof(uint64_t), cudaMemcpyHostToDevice);
        SLAM_LYJ_CUDA::BaseCU baseDev;
        cudaDeviceSynchronize();

        // run
        SLAM_LYJ_CUDA::ProjectorCU projector;
        projector.testTransformCUDA(TDev, PwsDev, PcsDev, vn);
        projector.testTransformNormalCUDA(TDev, fNormalwsDev, fNormalcsDev, vn);
        projector.testCameraCUDA(PcsDev, pixelsDev, vn, w, h, camDev);
        projector.testDepthAndFidCUDA(PcsDev, pixelsDev, facesDev, fNormalcsDev, fn, w, h, depthsDev, fidsDev, camDev, baseDev);
        projector.testDepthAndFidCUDA(PcsDev, pixelsDev, facesDev, fNormalcsDev, fn, w, h, didsDev, camDev, baseDev);
        cudaDeviceSynchronize();

        // download
        cudaMemcpy(Pcs.data(), PcsDev, vn * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(pixels.data(), pixelsDev, vn * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(depths.data(), depthsDev, w * h * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(fids.data(), fidsDev, w * h * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(dids.data(), didsDev, w * h * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free
        cudaFree(PwsDev);
        cudaFree(PcsDev);
        cudaFree(facesDev);
        cudaFree(fNormalwsDev);
        cudaFree(fNormalcsDev);
        cudaFree(pixelsDev);
        cudaFree(depthsDev);
        cudaFree(fidsDev);
        cudaFree(didsDev);

        // output
        std::ofstream f1("rawPs.txt");
        std::ofstream f2("transPs.txt");
        std::ofstream f3("finalPs.txt");
        std::ofstream f4("finalPixels.txt");
        std::ofstream f5("fullPixels.txt");
        std::ofstream f6("error.txt");
        for (int i = 0; i < vn; ++i)
        {
            f1 << Pws[3 * i] << " " << Pws[3 * i + 1] << " " << Pws[3 * i + 2] << std::endl;
            f2 << Pcs[3 * i] << " " << Pcs[3 * i + 1] << " " << Pcs[3 * i + 2] << std::endl;
        }
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                f5 << i << " " << j << " " << 1 << std::endl;
                int loc = i * w + j;
                float d = depths[loc];
                if (fids[loc] != 0 && fids[loc] != UINT_MAX)
                {
                    float x = (j * camInv[0] + camInv[2]) * d;
                    float y = (i * camInv[1] + camInv[3]) * d;
                    float z = d;
                    f6 << x << " " << y << " " << z << std::endl;
                    f6 << i << " " << j << " " << 1 << std::endl;
                    printf("err3!\n");
                }
                if (d == FLT_MAX)
                    continue;
                float x = (j * camInv[0] + camInv[2]) * d;
                float y = (i * camInv[1] + camInv[3]) * d;
                float z = d;
                f3 << x << " " << y << " " << z << std::endl;
                // std::cout << x << " " << y << " " << z << std::endl;
                f4 << i << " " << j << " " << 1 << std::endl;
            }
        }
        f1.close();
        f2.close();
        f3.close();
        f4.close();
        f5.close();
        f6.close();
        memcpy(did2s.data(), dids.data(), w * h * sizeof(uint64_t));
        for (int i = 0; i < w * h; ++i)
        {
            if (did2s[i].depth == FLT_MAX)
                continue;
            if (did2s[i].depth != 2.0f && did2s[i].fid != 0)
                printf("error!\n");
        }

        return;
    }
    CUDA_LYJ_API void test3()
    {
        //base
        int w = 160;
        int h = 120;
        std::vector<unsigned char> img(w * h * 4, 0);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int loc = (i * w + j) * 4;
                for (int k = 0; k < 4; ++k) {
                    img[loc + k] = 255;
                }
            }
        }
        std::vector<float> rets(w * h, 0);
        float* retsDev;
        
        //malloc
        //纹理对象
        //texture<uchar4, cudaTextureType2D> texRef;
        cudaTextureObject_t texObj;
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        //初始化资源描述符
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        //初始化纹理描述符
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        if (false) {
            texDesc.filterMode = cudaFilterModeLinear; //归一化
            texDesc.readMode = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords = true;
        }
        else {
            texDesc.filterMode = cudaFilterModePoint;      // 点采样（整数坐标推荐）
            texDesc.readMode = cudaReadModeElementType;    // 原始数据类型读取
            texDesc.normalizedCoords = false;              // 禁用归一化坐标
        }
        //cuda数组
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, w, h);
        cudaMemcpy2DToArray(cuArray, 0, 0, img.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
        //数组绑定到纹理对象
        resDesc.res.array.array = cuArray;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        //输出
        cudaMalloc((void**)&retsDev, w * h * sizeof(float));

        //process
        testTextureCUDA(retsDev, w, h, texObj);

        //output
        cudaMemcpy(rets.data(), retsDev, w * h * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < w * h; ++i) {
            //if (rets[i] != 4)
            //    printf("111\n");
        }
        
        //free
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);

        return;
    }
}