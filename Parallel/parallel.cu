#include "wb.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define MASK_SIZE 5
#define sigma 0.9
#define MASK_RADIUS MASK_SIZE/ 2
#define TILE_WIDTH 16
#define SIZE        (TILE_WIDTH + MASK_SIZE - 1)
#define PI 3.141592653589793238  

__constant__ float M[MASK_SIZE * MASK_SIZE];

__global__ void convolution2D (float * I,float * P,
        int channels, int width, int height)
{
    __shared__ float N_ds[SIZE][SIZE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

      for(int k=0;k<channels;k++){
        int dest  = ty * TILE_WIDTH + tx;
        int destX = dest % SIZE;
        int destY = dest / SIZE;
        int srcY  = by * TILE_WIDTH + destY - MASK_RADIUS;
        int srcX  = bx * TILE_WIDTH + destX - MASK_RADIUS;
        int src   = (srcY * width + srcX) * channels + k;

        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = I[src];
        else
            N_ds[destY][destX] = 0.0;

        __syncthreads();

        dest  = ty * TILE_WIDTH + tx + TILE_WIDTH * TILE_WIDTH;
        destY = dest / SIZE;
        destX = dest % SIZE;
        srcY  = by * TILE_WIDTH + destY - MASK_RADIUS;
        srcX  = bx * TILE_WIDTH + destX - MASK_RADIUS;
        src   = (srcY * width + srcX) * channels + k;

        if (destY < SIZE) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0.0;
        }
        __syncthreads();

        float accum = 0;
        for (int y = 0; y < MASK_SIZE; ++y)
            for (int x = 0; x < MASK_SIZE; ++x)
                accum += N_ds[ty + y][tx + x] * M[y * MASK_SIZE + x];

        int y = by * TILE_WIDTH + ty;
        int x = bx * TILE_WIDTH + tx;
        if (y < height && x < width)
            P[(y * width + x) * channels +k ] = min(max(accum, 0.0), 1.0);

        __syncthreads();
   }
}

int main (int argc, char * argv[ ])
{
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float hostMaskData[MASK_SIZE*MASK_SIZE];
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
    clock_t begin = clock();

    clock_t begin_imread = clock();
    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);


    inputImage = wbImport(inputImageFile);
   
    printf("Image Dimension: %40d X %d \n",wbImage_getWidth(inputImage),wbImage_getHeight(inputImage));
   
    //IplImage *img = cvLoadImage("input0.ppm",CV_LOAD_IMAGE_GRAYSCALE);   
    printf("Image Loading time: %40.6lf secs\n",(double)(clock()-begin_imread)/(double)(CLOCKS_PER_SEC));

    maskRows = MASK_SIZE;
    maskColumns = MASK_SIZE; 
    
    float mask[MASK_SIZE][MASK_SIZE];
    float x,y;
    clock_t begin_gauss = clock();
    for(int i=0;i<MASK_SIZE;i++){
	for(int j=0;j<MASK_SIZE;j++){
		x = i - (maskRows/2);
		y = j - (maskColumns/2);
		mask[i][j] = -1.0 * (2 * sigma * sigma - (x * x + y * y)) /(2.0 * PI * sigma * sigma * sigma * sigma) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));				
		hostMaskData[i*MASK_SIZE+j] = mask[i][j];
		}
	}    
     clock_t end_gauss = clock();
     printf("Log Filter execution time: %40.6lf secs\n",(double)(end_gauss-begin_gauss)/(double)(CLOCKS_PER_SEC));

      /*for(int i=0;i<MASK_SIZE;i++){
	
	for(int j=0;j<MASK_SIZE;j++){
		printf("%.1f ",hostMaskData[i*MASK_SIZE+j]);
	}
	cout<<endl;
	}
    */
    //////////////////////////////

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

	
   // Mat A = Mat(imageHeight, imageWidth, CV_32FC3 ,wbImage_getData(inputImage));
    
   // A.convertTo(A, CV_8UC3, 255.0);   
   // imwrite("Wind.jpg",A);
    
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);//(float *)img->imageData;
    hostOutputImageData = wbImage_getData(outputImage);

    clock_t begin_gpu_comp = clock();
    
    clock_t begin_gpu_malloc = clock();
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    
    printf("GPU memory allocation time: %40.6lf secs\n",(double)(clock()-begin_gpu_malloc)/(double)(CLOCKS_PER_SEC));

    clock_t begin_copy_htod = clock();
    cudaMemcpyToSymbol(M, hostMaskData, sizeof(int) * MASK_SIZE * MASK_SIZE);//
    cudaMemcpy(deviceInputImageData, hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData,maskRows * maskColumns * sizeof(float),cudaMemcpyHostToDevice);

    printf("Copy Time HOST to Device: %40.6lf secs\n",(double)(clock()-begin_copy_htod)/(double)(CLOCKS_PER_SEC));

    
    cudaEvent_t start,stop;
    float tot;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    clock_t begin_comp = clock();
    dim3 dimGrid(ceil((float) imageWidth / TILE_WIDTH),ceil((float) imageHeight / TILE_WIDTH));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution2D<<<dimGrid, dimBlock>>>(deviceInputImageData, /*deviceMaskData,*/deviceOutputImageData, imageChannels, imageWidth, imageHeight);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tot,start,stop);

    printf("Computation time on GPU: %40.6lf secs\n",(double)(clock()-begin_comp)/(double)(CLOCKS_PER_SEC));

    clock_t begin_copy_dtoh = clock();
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),cudaMemcpyDeviceToHost);
    printf("Copy time Device to HOST: %40.6lf secs\n",(double)(clock()-begin_copy_dtoh)/(double)(CLOCKS_PER_SEC));
    
    printf("Total time: %40.6lf secs\n",(double)(clock()-begin_gpu_comp)/(double)(CLOCKS_PER_SEC));

    Mat B = Mat(imageHeight, imageWidth, CV_32FC3, wbImage_getData(outputImage));
    B.convertTo(B, CV_8UC3, 255.0);
    imwrite("OUTPUT.jpg",B);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    
    cvWaitKey(0);
    return 0;
}
