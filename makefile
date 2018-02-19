
GCCPARAMS = -W -Wall -Wextra -Werror -O3
LDPARAMS = -lm -O3
LDSFML = -lsfml-graphics -lsfml-window -lsfml-system

OBJECTS = layer.o brain.o getlayer.o getbrain.o fsys.o main.o
EXEC = brain.ex



all: $(EXEC)

%.o: %.cpp
	g++ $(GCCPARAMS) -c $^ -o $@


$(EXEC): $(OBJECTS)
	g++ -o $@ $^ $(LDPARAMS)	# $(LDSFML) --> in case of using SFML


run: $(EXEC)
	./$<
	
clean:
	rm -rf $(OBJECTS)

remove:
	rm -rf $(OBJECTS) $(EXEC)




