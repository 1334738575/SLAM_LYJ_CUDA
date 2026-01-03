#ifndef CUDA_TEXTURE_H
#define CUDA_TEXTURE_H

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace CUDA_LYJ
{
	void testTextureCUDA(float* _output, int _w, int _h, cudaTextureObject_t _texObj);


	class TextureCU
	{
	public:
		TextureCU();
		~TextureCU();

	private:

	};
}


#endif // !CUDA_TEXTURE_H
