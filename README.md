# Vehicle Speed Measurement using CUDA

This project implements a vehicle speed measurement system using CUDA and image processing techniques on the NVIDIA Jetson Nano. The system detects moving vehicles, calculates their speed in km/h, and displays the results in real-time.

## **1. Overview**
The program processes video frames captured from a camera, detects moving vehicles using a CUDA kernel, and calculates their speed based on pixel displacement across frames. CUDA's parallel processing capabilities allow efficient real-time operation.

---

## **2. Key Features**
- Real-time motion detection using CUDA.
- Bounding box calculation with atomic operations.
- Speed calculation in km/h based on real-world dimensions.
- Displacement smoothing using exponential moving averages.
- Real-time video output with speed overlay.

---

## **3. Code Breakdown**

### **Libraries Used:**
- `videoSource` and `videoOutput`: Camera input and video output.
- `cudaMappedMemory` and `cudaFont`: Memory management and text overlay.
- `cuda_runtime.h`: Core CUDA functionalities.

### **CUDA Kernel: Motion Detection**
The kernel compares consecutive frames and calculates pixel differences. If a significant difference is detected, it updates the bounding box using atomic operations:

```cpp
__global__ void motionDetectionKernel(const uchar3* current, const uchar3* previous, uchar3* output, int* bbox, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 curPixel = current[idx];
        uchar3 prevPixel = previous[idx];
        int diff = abs(curPixel.x - prevPixel.x) + abs(curPixel.y - prevPixel.y) + abs(curPixel.z - prevPixel.z);

        if (diff > 50) {  // Motion threshold
            output[idx] = curPixel;
            atomicMin(&bbox[0], x); atomicMax(&bbox[1], x); // Update min/max X
            atomicMin(&bbox[2], y); atomicMax(&bbox[3], y); // Update min/max Y
        } else {
            output[idx] = curPixel;
        }
    }
}
```

### **Speed Calculation Function:**
The vehicle's speed is calculated using the following equation:

```cpp
float calculateSpeed(int displacementPixels, float realWorldWidth, int frameWidth, float elapsedTime) {
    if (elapsedTime <= 0.0f) return 0.0f;
    float speedMps = (displacementPixels * realWorldWidth) / (frameWidth * elapsedTime);
    return speedMps * 3.6;  // Convert m/s to km/h
}
```

### **Displacement Smoothing:**
To reduce noise, the displacement is smoothed using an exponential moving average:

```cpp
smoothedDisplacement = (smoothedDisplacement * 0.8) + (displacementPixels * 0.2);
```

---

## **4. Main Workflow**
1. Initialize video input/output.
2. Allocate GPU memory.
3. Capture video frames.
4. Launch the CUDA kernel for motion detection.
5. Update bounding box and calculate displacement.
6. Smooth displacement and calculate speed in km/h.
7. Display speed overlay on the video output.
8. Free allocated memory and cleanup.

---

## **5. Running the Code**
To run the code:

```bash
./vehicle_speed_detector input.mp4 output.mp4
```

---

## **6. Acknowledgments**
This project demonstrates CUDA’s potential for real-time video processing and embedded system applications like traffic monitoring.

---

## **7. License**
MIT License - Feel free to use and modify the code.

