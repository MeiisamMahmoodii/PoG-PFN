
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.data.scm_generator import SCMGenerator, MechanismType, GraphType
import numpy as np

def test_stability():
    print("Testing Nonlinear Stability...")
    print("-" * 50)
    
    max_val = 0.0
    n_trials = 100
    
    for i in range(n_trials):
        scm = SCMGenerator(
            n_vars=10,
            graph_type=GraphType.ERDOS_RENYI,
            mechanism_type=MechanismType.NONLINEAR_ADDITIVE,
            density=0.2,
            seed=i
        )
        
        data, _ = scm.sample(n_samples=500)
        curr_max = np.abs(data).max()
        max_val = max(max_val, curr_max)
        
        if curr_max > 1000:
            print(f"❌ Trial {i}: Value explosion detected! Max: {curr_max}")
            return False
            
    print(f"✅ Passed {n_trials} trials.")
    print(f"Maximum value observed: {max_val:.2f}")
    if max_val < 500:
        print("Verdict: STABLE")
        return True
    else:
        print("Verdict: UNSTABLE (Still too high)")
        return False

if __name__ == "__main__":
    test_stability()
