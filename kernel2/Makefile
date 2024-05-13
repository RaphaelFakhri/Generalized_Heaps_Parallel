all: main.cu kernel2.cu
	nvcc main.cu kernel2.cu -Xptxas -O3 -std=c++17 -o main

clean:
	rm -rf a.out kernel2 main
