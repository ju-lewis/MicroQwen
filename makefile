
build := build
src   := src
flags := -Wall -Wconversion -Wpedantic -Wextra


inference:
	

test: linalg.o
	gcc ${flags} ${src}/test.c ${build}/linalg.o -o test

linalg.o:
	gcc ${flags} -c ${src}/linalg.c -o ${build}/linalg.o

clean:
	rm ${build}/*.o
