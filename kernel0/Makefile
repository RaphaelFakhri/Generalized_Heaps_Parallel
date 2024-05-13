all: main.cu kernel0.cu
	nvcc main.cu kernel0.cu -Xptxas -O3 -std=c++17 -o main

clean:
	rm -rf a.out kernel0 main
