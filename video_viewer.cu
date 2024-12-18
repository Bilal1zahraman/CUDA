#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaFont.h>
#include <cuda_runtime.h>
#include <iostream>
#include <climits> // For INT_MAX and INT_MIN

using namespace std;

__global__ void motionDetectionKernel(const uchar3* current, const uchar3* previous, uchar3* output, int* bbox, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        uchar3 curPixel = current[idx];
        uchar3 prevPixel = previous[idx];

        int diff = abs(curPixel.x - prevPixel.x) +
                   abs(curPixel.y - prevPixel.y) +
                   abs(curPixel.z - prevPixel.z);

        if (diff > 50) {
            output[idx] = curPixel;
            atomicMin(&bbox[0], x);
            atomicMax(&bbox[1], x);
            atomicMin(&bbox[2], y);
            atomicMax(&bbox[3], y);
        } else {
            output[idx] = curPixel;
        }
    }
}

float calculateSpeed(int displacementPixels, float realWorldWidth, int frameWidth, float elapsedTime) {
    if (elapsedTime <= 0.0f) return 0.0f;
    float speedMps = (displacementPixels * realWorldWidth) / (frameWidth * elapsedTime);
    return speedMps * 3.6;
}

int main(int argc, char** argv) {
    videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput* output = videoOutput::Create(argc, argv, ARG_POSITION(1));

    if (!input || !output) {
        cerr << "Failed to initialize video input/output." << endl;
        return -1;
    }

    int inputWidth = input->GetWidth();
    int inputHeight = input->GetHeight();
    size_t frameSize = inputWidth * inputHeight * sizeof(uchar3);

    uchar3* imgCurrent = NULL;
    uchar3* imgPrevious = NULL;
    uchar3* imgOutput = NULL;
    cudaMalloc(&imgPrevious, frameSize);
    cudaMalloc(&imgOutput, frameSize);

    int bbox[4] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
    int* d_bbox;
    cudaMalloc(&d_bbox, 4 * sizeof(int));

    cudaFont* font = cudaFont::Create();
    if (!font) {
        cerr << "Failed to load font for overlay!" << endl;
        return -1;
    }

    const float realWorldWidth = 5.0;
    const float frameTime = 1.0 / 30;
    const int numFrames = 10;
    int positions[numFrames] = {0};
    int frameIndex = 0;

    int lastDisplacement = 0;
    float lastSpeed = 0.0f;
    static int smoothedDisplacement = 0;

    while (true) {
        int status = 0;
        if (!input->Capture(&imgCurrent, 1000, &status)) {
            if (status == videoSource::TIMEOUT) continue;
            break;
        }

        if (!imgPrevious) {
            cudaMemcpy(imgPrevious, imgCurrent, frameSize, cudaMemcpyDeviceToDevice);
            continue;
        }

        bbox[0] = INT_MAX; bbox[1] = INT_MIN;
        bbox[2] = INT_MAX; bbox[3] = INT_MIN;
        cudaMemcpy(d_bbox, bbox, 4 * sizeof(int), cudaMemcpyHostToDevice);

        dim3 blockDim(16, 16);
        dim3 gridDim((inputWidth + blockDim.x - 1) / blockDim.x, (inputHeight + blockDim.y - 1) / blockDim.y);
        motionDetectionKernel<<<gridDim, blockDim>>>(imgCurrent, imgPrevious, imgOutput, d_bbox, inputWidth, inputHeight);

        cudaMemcpy(bbox, d_bbox, 4 * sizeof(int), cudaMemcpyDeviceToHost);

        int currentX = (bbox[0] + bbox[1]) / 2;
        positions[frameIndex] = currentX;
        frameIndex = (frameIndex + 1) % numFrames;

        int firstPosition = positions[frameIndex];
        int lastPosition = positions[(frameIndex + 1) % numFrames];
        int displacementPixels = abs(lastPosition - firstPosition);

        if (displacementPixels > 500) displacementPixels = 500;

        smoothedDisplacement = (smoothedDisplacement * 0.8) + (displacementPixels * 0.2);
        float speedKmh = calculateSpeed(smoothedDisplacement, realWorldWidth, inputWidth, frameTime * numFrames);

        if (smoothedDisplacement != lastDisplacement || fabs(speedKmh - lastSpeed) > 0.01f) {
            printf("Displacement (pixels/sec): %.2f\n", smoothedDisplacement * 30.0f);
            printf("Speed: %.2f km/h\n", speedKmh);
            lastDisplacement = smoothedDisplacement;
            lastSpeed = speedKmh;
        }

        char overlayText[256];
        sprintf(overlayText, "Speed: %.2f km/h ", speedKmh);
        font->OverlayText(imgOutput, inputWidth, inputHeight, overlayText, 10, 10, make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));

        output->Render(imgOutput, inputWidth, inputHeight);
        cudaMemcpy(imgPrevious, imgCurrent, frameSize, cudaMemcpyDeviceToDevice);

        if (!output->IsStreaming()) break;
    }

    cudaFree(imgPrevious);
    cudaFree(imgOutput);
    cudaFree(d_bbox);
    delete input;
    delete output;

    return 0;
}