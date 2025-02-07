
build := build
src   := src
flags := -Wall -Wconversion -Wpedantic -Wextra -g -std=c2x -lm


#inference:
	

test: linalg.o safetensor.o util.o
	gcc ${flags} ${src}/test.c ${build}/linalg.o ${build}/safetensor.o ${build}/util.o -o test 

linalg.o:
	gcc ${flags} -c ${src}/linalg.c -o ${build}/linalg.o -O3


safetensor.o: util.o
	gcc ${flags} -c ${src}/safetensor.c -o ${build}/safetensor.o


util.o:
	gcc ${flags} -c ${src}/util.c -o ${build}/util.o


clean:
	rm ${build}/*.o
