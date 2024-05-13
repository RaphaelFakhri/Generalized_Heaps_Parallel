all: main.cu kernel3.cu
	nvcc main.cu kernel3.cu -Xptxas -O3 -std=c++17 -o main

clean:
	rm -rf a.out kernel3 main
