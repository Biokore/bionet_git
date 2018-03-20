
GCCPARAMS = -W -Wall -Wextra -Werror -O6 -fPIC
LDPARAMS = -lm -O6 -shared

OBJECTS = layer.o layer_get.o nnet.o nnet_train.o nnet_get.o fsys.o
LIB = libnnet.so



all: $(LIB)

%.o: %.cpp
	g++ $(GCCPARAMS) -c $^ -o $@


$(LIB): $(OBJECTS)
	g++ -o $@ $^ $(LDPARAMS)

clean:
	rm -rf $(OBJECTS)

install:
	cp *.hpp ./lib/headers/
	mv libnnet.so ./lib/shared/
	make clean




