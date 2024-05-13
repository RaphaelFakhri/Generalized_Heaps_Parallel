all: main.cu kernel1.cu
	nvcc main.cu kernel1.cu -Xptxas -O3 -std=c++17 -o main

clean:
	rm -rf a.out kernel1 main
