CC = gcc
CFLAGS = -Wall -Wextra -I./src

all: main

main: main.o heap.o
	$(CC) $(CFLAGS) -o main main.o heap.o

main.o: src/main.c src/heap.h
	$(CC) $(CFLAGS) -c src/main.c

heap.o: src/heap.c src/heap.h
	$(CC) $(CFLAGS) -c src/heap.c

clean:
	rm -f main *.o
