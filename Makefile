face_detection: face_detection.cpp compute_feature.cpp
	@g++ -Wall -O2 `pkg-config --cflags --libs opencv` -o $@ $+

topk: topk.cpp

realboost: realboost.cpp compute_threshold.cpp compute_error.cpp compute_feature.cpp
	@$(CXX) -Wall -O2 -o $@ $+

adaboost: adaboost.cpp compute_threshold.cpp compute_error.cpp compute_feature.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -o $@ $+

find_errors: find_errors.cpp compute_threshold.cpp compute_error.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -o $@ $+

precompute_feature_values: precompute_feature_values.cpp compute_feature.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -o $@ $+

weak_classifiers: weak_classifiers.cpp

process_images: process_images.cpp

%: %.cpp
	@echo "CXX $<"
	@$(CXX) -Wall -O2 -o $@ $<

clean:
	@rm -f  *.o
	@rm -rf *.dSYM
	@rm -f top10 find_errors precompute_feature_values weak_classifiers process_images topk adaboost face_detection

.PHONY: all clean
