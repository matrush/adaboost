read_images: process_images.cpp

%: %.cpp
	@$(CXX) -Wall -O2 -o $@ $<

clean:
	@rm -f *.o

.PHONY: clean
