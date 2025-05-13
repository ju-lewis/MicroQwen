
BUILD    := build
SRC      := src
FLAGS    := -Wall -Wconversion -Wpedantic -Wextra -g -std=c2x -lm
COMPILER := cc

OBJS := linalg.o safetensor.o nn.o util.o


PREFIXED_OBJS = $(addprefix $(BUILD)/, $(OBJS))

test: $(OBJS)
	$(COMPILER) $(FLAGS) -o test $(SRC)/test.c $(PREFIXED_OBJS)

	
# Object files
%.o: $(SRC)/%.c
	$(COMPILER) $(FLAGS) -c -o $(BUILD)/$@ $<

	


clean:
	rm $(BUILD)/*.o
