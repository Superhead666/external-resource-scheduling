CXX ?= c++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20

GUROBI_HOME ?= /Library/gurobi1302/macos_universal2
GUROBI_VERSION ?= 130
GUROBI_INC ?= -I"$(GUROBI_HOME)/include"
GUROBI_LIB ?= -L"$(GUROBI_HOME)/lib" -lgurobi_c++ -lgurobi$(GUROBI_VERSION) -Wl,-rpath,"$(GUROBI_HOME)/lib"

CPLEX_HOME ?= /Applications/CPLEX_Studio2211
CP_PLATFORM ?= arm64_osx
CP_INC ?= -I"$(CPLEX_HOME)/cpoptimizer/include" -I"$(CPLEX_HOME)/concert/include"
CP_LIB ?= -L"$(CPLEX_HOME)/cpoptimizer/lib/$(CP_PLATFORM)/static_pic" \
	-L"$(CPLEX_HOME)/cplex/lib/$(CP_PLATFORM)/static_pic" \
	-L"$(CPLEX_HOME)/concert/lib/$(CP_PLATFORM)/static_pic" \
	-lcp -lcplex -lconcert -lpthread -ldl

.PHONY: all baselines clean

all: bin/generate_instances bin/proposed_solver bin/component_solver bin/empty_set_lower_bound

baselines: bin/gurobi_baselines bin/cp_baseline bin/integer_lshaped

bin:
	mkdir -p bin

bin/generate_instances: src/generate_instances.cpp src/instance.hpp | bin
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

bin/proposed_solver: src/proposed_solver.cpp src/instance.hpp src/frontier_labeling_core.inc | bin
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

bin/component_solver: src/component_solver.cpp src/instance.hpp src/frontier_labeling_core.inc | bin
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

bin/empty_set_lower_bound: src/empty_set_lower_bound.cpp src/instance.hpp src/scenario_lower_bound.hpp | bin
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

bin/price_sensitivity: sensitivity/price_sensitivity.cpp src/instance.hpp src/frontier_labeling_core.inc | bin
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

bin/gurobi_baselines: src/gurobi_baselines.cpp src/instance.hpp src/scenario_lower_bound.hpp | bin
	$(CXX) $(CXXFLAGS) $(GUROBI_INC) -Isrc $< $(GUROBI_LIB) -o $@

bin/cp_baseline: src/cp_baseline.cpp src/instance.hpp src/scenario_lower_bound.hpp src/cp_structure.hpp | bin
	$(CXX) $(CXXFLAGS) -Wno-deprecated-declarations $(CP_INC) -Isrc $< $(CP_LIB) -o $@

bin/integer_lshaped: src/integer_lshaped.cpp src/instance.hpp src/scenario_lower_bound.hpp src/cp_structure.hpp | bin
	$(CXX) $(CXXFLAGS) -Wno-deprecated-declarations $(GUROBI_INC) $(CP_INC) -Isrc $< $(GUROBI_LIB) $(CP_LIB) -o $@

clean:
	rm -rf bin
