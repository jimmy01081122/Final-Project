#!/usr/bin/env python3
"""
BMC Verification for TSP Verilog Module
Reads tsp.v file and verifies property output using Z3 solver
"""

import sys
import re
from z3 import *
import time

class VerilogTSPVerifier:
    def __init__(self, verilog_file="tsp.v"):
        self.verilog_file = verilog_file
        self.costs = {}
        self.threshold = 0
        self.parse_verilog()
        
    def parse_verilog(self):
        """Parse Verilog file to extract cost constants and threshold"""
        try:
            with open(self.verilog_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract cost definitions
            cost_pattern = r'`define\s+COST_(\d)_(\d)\s+8\'d(\d+)'
            for match in re.finditer(cost_pattern, content):
                from_city, to_city, cost = int(match.group(1)), int(match.group(2)), int(match.group(3))
                self.costs[(from_city, to_city)] = cost
            
            # Extract threshold
            threshold_match = re.search(r'`define\s+THRESHOLD\s+8\'d(\d+)', content)
            if threshold_match:
                self.threshold = int(threshold_match.group(1))
            
            print(f"Parsed {len(self.costs)} cost definitions, THRESHOLD={self.threshold}")
            
        except FileNotFoundError:
            print(f"Error: {self.verilog_file} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error parsing Verilog file: {e}")
            sys.exit(1)
    
    def get_cost(self, from_city, to_city):
        """Get cost between two cities"""
        return self.costs.get((from_city, to_city), 0)
    
    def calculate_tour_cost(self, j, k, l):
        """Calculate total cost for tour 0→j→k→l→0"""
        return (self.get_cost(0, j) + self.get_cost(j, k) + 
                self.get_cost(k, l) + self.get_cost(l, 0))
    
    def enumerate_all_tours(self):
        """Generate all valid 4-city tours starting from city 0"""
        tours = []
        for j in [1, 2, 3]:
            for k in [1, 2, 3]:
                for l in [1, 2, 3]:
                    if j != k and j != l and k != l:  # All cities must be different
                        cost = self.calculate_tour_cost(j, k, l)
                        tours.append((0, j, k, l, 0, cost))
        return tours
    
    def verify_property(self):
        """Verify the property: true if no tour cost < THRESHOLD"""
        tours = self.enumerate_all_tours()
        
        print(f"\nAll possible tours from city 0:")
        min_cost = float('inf')
        tours_below_threshold = []
        
        for i, tour in enumerate(tours, 1):
            path_str = f"{tour[0]}→{tour[1]}→{tour[2]}→{tour[3]}→{tour[4]}"
            cost = tour[5]
            min_cost = min(min_cost, cost)
            
            print(f"{i}. {path_str}: cost={cost}")
            
            if cost < self.threshold:
                tours_below_threshold.append(tour)
        
        # Property is true if NO tours have cost < THRESHOLD
        property_result = len(tours_below_threshold) == 0
        
        print(f"\nThreshold: {self.threshold}")
        print(f"Minimum tour cost: {min_cost}")
        print(f"Tours with cost < {self.threshold}: {len(tours_below_threshold)}")
        
        if tours_below_threshold:
            print("Tours below threshold:")
            for tour in tours_below_threshold:
                path_str = f"{tour[0]}→{tour[1]}→{tour[2]}→{tour[3]}→{tour[4]}"
                print(f"  {path_str}: cost={tour[5]}")
        
        return property_result
    
    def bmc_verification(self, max_steps=100):
        """BMC verification using Z3 solver with frame-by-frame execution"""
        print(f"\nRunning BMC verification (max_steps={max_steps})...")
        
        solver = Solver()
        
        # State variables for each time step
        state = [BitVec(f'state_{t}', 2) for t in range(max_steps)]
        j = [BitVec(f'j_{t}', 2) for t in range(max_steps)]
        k = [BitVec(f'k_{t}', 2) for t in range(max_steps)]
        l = [BitVec(f'l_{t}', 2) for t in range(max_steps)]
        foundLower = [Bool(f'foundLower_{t}') for t in range(max_steps)]
        initialized = [Bool(f'initialized_{t}') for t in range(max_steps)]
        property_out = [Bool(f'property_{t}') for t in range(max_steps)]
        
        # State encoding
        S_INIT, S_COMPUTE, S_DONE = 0, 1, 2
        state_names = {0: 'INIT', 1: 'COMPUTE', 2: 'DONE'}
        
        print("\nFrame-by-frame BMC execution:")
        print("Frame | Time(s) | State | j | k | l | foundLower | Property")
        print("-" * 60)
        
        total_time = 0
        
        for frame in range(max_steps):
            frame_start = time.time()
            
            # Build constraints incrementally
            if frame == 0:
                # Initial conditions
                solver.add(state[0] == S_INIT)
                solver.add(foundLower[0] == False)
                solver.add(initialized[0] == False)
            else:
                # Add transition constraints for this frame
                t = frame - 1
                solver.add(initialized[t+1] == True)
                
                solver.add(
                    If(state[t] == S_INIT,
                       And(state[t+1] == S_COMPUTE,
                           j[t+1] == 1, k[t+1] == 2, l[t+1] == 3),
                       If(state[t] == S_COMPUTE,
                          self.add_compute_transition(solver, t, j, k, l, foundLower, state),
                          # S_DONE: stay in done state
                          And(state[t+1] == S_DONE,
                              foundLower[t+1] == foundLower[t]))))
            
            # Property definition for this frame
            solver.add(property_out[frame] == And(initialized[frame], 
                                                state[frame] == S_DONE, 
                                                Not(foundLower[frame])))
            
            # Try to solve up to current frame
            solver.push()  # Save state
            
            # Check if we can reach current frame with valid states
            check_result = solver.check()
            frame_time = time.time() - frame_start
            total_time += frame_time
            
            if check_result == sat:
                model = solver.model()
                
                # Extract values for current frame
                curr_state = model.eval(state[frame], model_completion=True).as_long()
                curr_j = model.eval(j[frame], model_completion=True).as_long() if j[frame] in model else 0
                curr_k = model.eval(k[frame], model_completion=True).as_long() if k[frame] in model else 0
                curr_l = model.eval(l[frame], model_completion=True).as_long() if l[frame] in model else 0
                curr_foundLower = model.eval(foundLower[frame], model_completion=True)
                curr_property = model.eval(property_out[frame], model_completion=True)
                
                print(f"{frame:5d} | {frame_time:6.3f} | {state_names[curr_state]:7s} | {curr_j} | {curr_k} | {curr_l} | {str(curr_foundLower):10s} | {curr_property}")
                
                # Check if we reached DONE state with stable property
                if curr_state == S_DONE:
                    # Verify property is stable for a few more frames
                    stable_frames = 3
                    if frame + stable_frames < max_steps:
                        # Check stability
                        for future_frame in range(frame + 1, min(frame + stable_frames + 1, max_steps)):
                            solver.add(state[future_frame] == S_DONE)
                            solver.add(property_out[future_frame] == property_out[frame])
                        
                        stability_check = solver.check()
                        if stability_check == sat:
                            print(f"\nProperty stabilized at frame {frame}")
                            print(f"Total BMC time: {total_time:.3f}s")
                            print(f"Final property value: {curr_property}")
                            solver.pop()
                            return str(curr_property).lower() == 'true'
            else:
                print(f"{frame:5d} | {frame_time:6.3f} | UNSAT   | - | - | - | -          | -")
            
            solver.pop()  # Restore state
            
            # Early termination if taking too long
            if total_time > 30:  # 30 second timeout
                print(f"\nTimeout after {total_time:.3f}s at frame {frame}")
                break
        
        print(f"\nBMC completed {frame + 1} frames in {total_time:.3f}s")
        print("Unable to determine final property value")
        return None
    
    def add_compute_transition(self, solver, t, j, k, l, foundLower, state):
        """Add constraints for S_COMPUTE state transitions"""
        # Calculate cost for current permutation and update foundLower
        cost_checks = []
        for j_val in [1, 2, 3]:
            for k_val in [1, 2, 3]:
                for l_val in [1, 2, 3]:
                    if j_val != k_val and j_val != l_val and k_val != l_val:
                        tour_cost = self.calculate_tour_cost(j_val, k_val, l_val)
                        cost_checks.append(
                            If(And(j[t] == j_val, k[t] == k_val, l[t] == l_val),
                               foundLower[t+1] == Or(foundLower[t], tour_cost < self.threshold),
                               True))
        
        # Simplified permutation advancement - check if we've covered all permutations
        last_permutation = And(j[t] == 3, k[t] == 2, l[t] == 1)
        
        return And(
            Or(cost_checks),
            If(last_permutation,
               state[t+1] == 2,  # S_DONE
               And(state[t+1] == 1,  # S_COMPUTE
                   # Simplified next permutation logic
                   Or(j[t+1] > j[t], 
                      And(j[t+1] == j[t], k[t+1] > k[t]),
                      And(j[t+1] == j[t], k[t+1] == k[t], l[t+1] > l[t]))))
        )
    
    def run_verification(self):
        """Run complete verification"""
        print("TSP Verilog Module Verification")
        print("=" * 40)
        
        # Analytical verification
        analytical_result = self.verify_property()
        print(f"\nAnalytical result: Property = {analytical_result}")
        
        # BMC verification
        bmc_result = self.bmc_verification()
        
        # Results
        print(f"\n{'='*40}")
        print(f"VERIFICATION RESULTS:")
        print(f"Analytical: Property = {analytical_result}")
        if bmc_result is not None:
            print(f"BMC:        Property = {bmc_result}")
            match = analytical_result == bmc_result
            print(f"Results match: {match}")
            return match
        else:
            print("BMC: Unable to verify")
            return False

def main():
    if len(sys.argv) > 1:
        verilog_file = sys.argv[1]
    else:
        verilog_file = "tsp.v"
    
    verifier = VerilogTSPVerifier(verilog_file)
    result = verifier.run_verification()
    
    if result:
        print("\n✓ Verification PASSED")
    else:
        print("\n✗ Verification FAILED")
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()