# all: weak_classifiers read_images

weak_classifiers: weak_classifiers.cpp

read_images: process_images.cpp

%: %.o
	@$(CXX) -Wall -O2 -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM

.PHONY: all clean
