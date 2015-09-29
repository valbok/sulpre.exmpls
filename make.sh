g++ fd/fd.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv` -o fd/fd
g++ shapes/shapes.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv`-o shapes/shapes
g++ -std=c++11 tpl/tpl.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv` -o tpl/tpl