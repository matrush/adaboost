# all: weak_classifiers read_images

weak_classifiers: weak_classifiers.cpp

process_images: process_images.cpp

%: %.cpp
	@$(CXX) -Wall -O2 -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
